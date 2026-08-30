import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from llm_lab.generation import (
    GenerationResponse,
    GenerationTiming,
    RuntimeMetadata,
    TokenUsage,
)
from llm_lab.models import ModelSpec
from llm_lab.quantization import (
    ArtifactProvenance,
    QuantizationManifest,
    QuantizationVariant,
)
from llm_lab.runtimes import RuntimeConfig


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "experiments/exp_002-quantization_llama_cpp_gguf/runner.py"
SPEC = importlib.util.spec_from_file_location("exp_002_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class ByteTokenizer:
    name = "fixture-llama-tokenizer-v1"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


class FakeRuntime:
    name = "llama.cpp"
    instances: list["FakeRuntime"] = []

    def __init__(self) -> None:
        self.loaded: tuple[ModelSpec, RuntimeConfig] | None = None
        self.closed = False
        self.__class__.instances.append(self)

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        self.loaded = (model, config)

    def get_tokenizer(self) -> ByteTokenizer:
        return ByteTokenizer()

    def generate(self, request: Any) -> GenerationResponse:
        if "Project Aurora" in request.prompt:
            output = "ZX-4817"
        elif "nightly build" in request.prompt:
            output = "Reliability Engineering"
        else:
            output = "8392"
        prompt_tokens = len(ByteTokenizer().encode(request.prompt))
        return GenerationResponse(
            output_text=output,
            usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=1),
            timing=GenerationTiming(
                ttft_seconds=0.01,
                post_first_chunk_seconds=0.02,
                total_seconds=0.03,
            ),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version="fixture",
                model_id=request.model.model_id,
                tokenizer_id=request.model.tokenizer_id,
                config={"model_path": self.loaded[1].options["model_path"]},
            ),
        )

    def close(self) -> None:
        self.closed = True


def _manifest_file(directory: Path) -> Path:
    variants = []
    for index, (condition_id, label, quantization_type, bits) in enumerate(
        (
            ("q8_0", "Q8_0", "Q8_0", 8),
            ("q6_k", "Q6_K", "Q6_K", 6),
            ("q5_k_m", "Q5_K_M", "Q5_K_M", 5),
            ("q4_k_m", "Q4_K_M", "Q4_K_M", 4),
        ),
        start=1,
    ):
        artifact_path = directory / f"{condition_id}.gguf"
        artifact_path.write_bytes(f"fixture-{index}".encode("ascii"))
        variants.append(
            QuantizationVariant(
                condition_id=condition_id,
                label=label,
                format="GGUF",
                quantization_type=quantization_type,
                bits=bits,
                runtime_kernel="ggml",
                artifact=ArtifactProvenance(
                    source_uri="hf://Qwen/Qwen3.8-27B",
                    source_revision="model-sha",
                    conversion_command="fixture conversion",
                    converter_revision="llama.cpp-sha",
                    artifact_uri=artifact_path.as_uri(),
                    artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                    artifact_size_bytes=artifact_path.stat().st_size,
                ),
            )
        )
    manifest = QuantizationManifest(
        experiment_id="exp_002",
        model_id="Qwen/Qwen3.8-27B",
        model_revision="model-sha",
        tokenizer_id="Qwen/Qwen3.8-27B",
        tokenizer_revision="tokenizer-sha",
        runtime_name="llama.cpp",
        runtime_version="llama-cpp-python-fixture",
        runtime_options={"n_ctx": 33088, "n_batch": 512},
        prompt_id="prompt.qa.v001",
        task_ids=(
            "task.literal.000001",
            "task.semantic.000001",
            "task.multihop.000001",
        ),
        context_lengths=(8192, 32768),
        sampling={
            "max_new_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": None,
            "seed": 42,
        },
        variants=tuple(variants),
        repeats=5,
        context_length_semantics="input_tokens",
        context_overhead_tokens=256,
    )
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest.to_record()), encoding="utf-8")
    return path


class Exp002RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeRuntime.instances = []

    def test_runner_uses_manifest_task_catalog_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            catalog_path = root / "tasks.jsonl"
            catalog_path.write_text(
                '{"schema_version": 1, "id": "task.fixture", '
                '"type": "literal_retrieval", "version": 1, '
                '"question": "What is the code?", '
                '"expected": {"type": "exact", "value": "A-1"}, '
                '"evidence": [{"id": "evidence", "text": "The code is A-1."}], '
                '"metadata": {"seed": 1, "source": "fixture", "license": "CC0"}}\n',
                encoding="utf-8",
            )
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["controls"]["task_ids"] = ["task.fixture"]
            record["controls"]["task_catalog"] = str(catalog_path)
            record["controls"]["task_catalog_sha256"] = hashlib.sha256(
                catalog_path.read_bytes()
            ).hexdigest()
            record["controls"]["scorer_version"] = "calibrated.v1"
            manifest_path.write_text(json.dumps(record), encoding="utf-8")
            output_path = root / "raw" / "trials.jsonl"
            summary_path = root / "processed" / "summary.csv"

            result = runner.run_experiment(
                manifest_path=manifest_path,
                output_path=output_path,
                processed_path=summary_path,
                condition_ids=("q8_0",),
                context_lengths=(8192,),
                repeats=1,
                runtime_factory=FakeRuntime,
            )

            self.assertEqual(1, result["actual_trial_n"])
            [trial] = runner.load_trial_results(output_path)
            self.assertEqual(str(catalog_path), trial.input["task_catalog"])
            self.assertEqual(
                record["controls"]["task_catalog_sha256"],
                trial.input["task_catalog_sha256"],
            )
            self.assertEqual("calibrated.v1", trial.input["scorer_version"])
            self.assertEqual(64, len(trial.input["context_sha256"]))
            self.assertIn("context/synthetic.py", trial.input["source_revisions"])
            self.assertIn("evaluation/contracts.py", trial.input["source_revisions"])
            self.assertIn("generation/types.py", trial.input["source_revisions"])
            self.assertIn("runtimes/llama_cpp.py", trial.input["source_revisions"])

    def test_run_fingerprint_binds_source_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _manifest_file(Path(directory))
            manifest = runner.load_manifest(manifest_path)

            first = runner._run_fingerprint(
                manifest, source_revisions={"context/synthetic.py": "revision-a"}
            )
            second = runner._run_fingerprint(
                manifest, source_revisions={"context/synthetic.py": "revision-b"}
            )

            self.assertNotEqual(first, second)

    def test_runner_rejects_a_task_catalog_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["controls"]["task_catalog"] = str(ROOT / "data/tasks/core.v002.jsonl")
            record["controls"]["task_catalog_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(record), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "task catalog SHA-256 mismatch"):
                runner.run_experiment(
                    manifest_path=manifest_path,
                    output_path=root / "raw" / "trials.jsonl",
                    processed_path=root / "processed" / "summary.csv",
                    condition_ids=("q8_0",),
                    context_lengths=(8192,),
                    repeats=1,
                    runtime_factory=FakeRuntime,
                )

    def test_pilot_runs_q8_at_8k_for_all_tasks_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            output_path = root / "raw" / "pilot-trials.jsonl"
            summary_path = root / "processed" / "pilot-summary.csv"

            result = runner.run_experiment(
                manifest_path=manifest_path,
                output_path=output_path,
                processed_path=summary_path,
                condition_ids=("q8_0",),
                context_lengths=(8192,),
                repeats=1,
                runtime_factory=FakeRuntime,
            )

            self.assertEqual(3, result["actual_trial_n"])
            trials = runner.load_trial_results(output_path)
            self.assertEqual(3, len(trials))
            self.assertTrue(all(item.score["scorer"] == "calibrated.v1" for item in trials))
            self.assertTrue(all("format_valid" in item.score for item in trials))
            self.assertTrue(summary_path.is_file())
            self.assertTrue(all(instance.closed for instance in FakeRuntime.instances))

    def test_timing_probes_are_written_separately_from_capability_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["controls"]["capability_repeats"] = 1
            record["controls"]["timing_repeats"] = 3
            manifest_path.write_text(json.dumps(record), encoding="utf-8")
            capability_output = root / "raw" / "capability.jsonl"
            capability_summary = root / "processed" / "summary.csv"
            timing_output = root / "raw" / "timing.jsonl"
            timing_summary = root / "processed" / "timing-summary.csv"

            result = runner.run_experiment(
                manifest_path=manifest_path,
                output_path=capability_output,
                processed_path=capability_summary,
                timing_output_path=timing_output,
                timing_processed_path=timing_summary,
                condition_ids=("q8_0",),
                context_lengths=(8192,),
                repeats=1,
                timing_repeats=3,
                runtime_factory=FakeRuntime,
            )

            self.assertEqual(3, result["actual_trial_n"])
            self.assertEqual(9, result["actual_timing_trial_n"])
            self.assertEqual(3, len(runner.load_trial_results(capability_output)))
            timing_trials = runner.load_trial_results(timing_output)
            self.assertEqual(9, len(timing_trials))
            self.assertTrue(
                all(trial.input.get("sample_role") == "timing" for trial in timing_trials)
            )
            self.assertTrue(timing_summary.is_file())
            capability_rows = capability_summary.read_text(encoding="utf-8").splitlines()
            timing_rows = timing_summary.read_text(encoding="utf-8").splitlines()
            self.assertIn("raw_results_sha256", capability_rows[0].split(","))
            self.assertIn("timing_raw_results_sha256", timing_rows[0].split(","))
            self.assertEqual(
                hashlib.sha256(capability_output.read_bytes()).hexdigest(),
                result["raw_results_sha256"],
            )
            self.assertEqual(
                hashlib.sha256(timing_output.read_bytes()).hexdigest(),
                result["timing_raw_results_sha256"],
            )

    def test_timing_probes_require_explicit_manifest_repeat_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            with self.assertRaisesRegex(ValueError, "explicit timing_repeats"):
                runner.run_experiment(
                    manifest_path=manifest_path,
                    output_path=root / "raw" / "capability.jsonl",
                    processed_path=root / "processed" / "summary.csv",
                    timing_output_path=root / "raw" / "timing.jsonl",
                    condition_ids=("q8_0",),
                    context_lengths=(8192,),
                    repeats=1,
                    runtime_factory=FakeRuntime,
                )

    def test_task_level_summary_keeps_distinct_tasks_in_one_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["controls"]["task_ids"] = [
                "task.literal.000001",
                "task.literal.000002",
            ]
            manifest_path.write_text(json.dumps(record), encoding="utf-8")
            output_path = root / "raw" / "trials.jsonl"
            summary_path = root / "processed" / "summary.csv"

            result = runner.run_experiment(
                manifest_path=manifest_path,
                output_path=output_path,
                processed_path=summary_path,
                condition_ids=("q8_0",),
                context_lengths=(8192,),
                repeats=1,
                runtime_factory=FakeRuntime,
            )

            self.assertEqual(2, result["summary_row_n"])
            rows = summary_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("task_id", rows[0].split(","))

    def test_full_selection_has_120_expected_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _manifest_file(Path(directory))
            manifest = runner.load_manifest(manifest_path)

            self.assertEqual(120, runner.expected_trial_count(manifest))

    def test_v002_template_declares_1200_trial_matrix(self) -> None:
        template = json.loads(
            (
                ROOT
                / "experiments/exp_002-quantization_llama_cpp_gguf/manifest.template.json"
            ).read_text(encoding="utf-8")
        )

        controls = template["controls"]
        self.assertEqual(30, len(controls["task_ids"]))
        self.assertEqual(1200, 4 * 2 * len(controls["task_ids"]) * controls["repeats"])

    def test_checked_in_manifest_is_resolved_v002(self) -> None:
        manifest_path = (
            ROOT
            / "experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json"
        )
        record = json.loads(manifest_path.read_text(encoding="utf-8"))

        controls = record["controls"]
        self.assertFalse(record.get("template", True))
        self.assertEqual(30, len(controls["task_ids"]))
        self.assertEqual("data/tasks/core.v002.jsonl", controls["task_catalog"])
        self.assertEqual("calibrated.v1", controls["scorer_version"])
        self.assertEqual(64, len(controls["task_catalog_sha256"]))

        manifest = runner.load_manifest(manifest_path)
        self.assertEqual(1200, runner.expected_trial_count(manifest))

    def test_checked_in_v002_pilot_summary_has_task_level_coverage(self) -> None:
        summary_path = (
            ROOT
            / "experiments/exp_002-quantization_llama_cpp_gguf/results/processed/pilot-v002-summary.csv"
        )
        rows = summary_path.read_text(encoding="utf-8").splitlines()

        self.assertIn("task_id", rows[0].split(","))
        self.assertEqual(30, len(rows) - 1)

    def test_analysis_notebook_rejects_legacy_summaries(self) -> None:
        notebook = (ROOT / "experiments/exp_002-quantization_llama_cpp_gguf/analysis.ipynb").read_text(
            encoding="utf-8"
        )

        self.assertIn("EXPECTED_SCORER = 'calibrated.v1'", notebook)
        self.assertIn("TIMING_SUMMARY_PATH", notebook)
        self.assertIn("separate timing summary is required", notebook)

    def test_resume_reuses_completed_pilot_trials_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            output_path = root / "raw" / "pilot-trials.jsonl"
            summary_path = root / "processed" / "pilot-summary.csv"
            kwargs = {
                "manifest_path": manifest_path,
                "output_path": output_path,
                "processed_path": summary_path,
                "condition_ids": ("q8_0",),
                "context_lengths": (8192,),
                "repeats": 1,
                "runtime_factory": FakeRuntime,
            }

            runner.run_experiment(**kwargs)
            result = runner.run_experiment(**kwargs)

            self.assertEqual(3, result["actual_trial_n"])
            self.assertEqual(3, len(runner.load_trial_results(output_path)))

    def test_resume_rejects_existing_trials_from_another_scorer_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            output_path = root / "raw" / "pilot-trials.jsonl"
            summary_path = root / "processed" / "pilot-summary.csv"
            kwargs = {
                "manifest_path": manifest_path,
                "output_path": output_path,
                "processed_path": summary_path,
                "condition_ids": ("q8_0",),
                "context_lengths": (8192,),
                "repeats": 1,
                "runtime_factory": FakeRuntime,
            }

            runner.run_experiment(**kwargs)
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["score"]["scorer"] = "expected.v1"
            output_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "scorer version"):
                runner.run_experiment(**kwargs)

    def test_resume_rejects_changed_source_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            output_path = root / "raw" / "pilot-trials.jsonl"
            summary_path = root / "processed" / "pilot-summary.csv"
            kwargs = {
                "manifest_path": manifest_path,
                "output_path": output_path,
                "processed_path": summary_path,
                "condition_ids": ("q8_0",),
                "context_lengths": (8192,),
                "repeats": 1,
                "runtime_factory": FakeRuntime,
            }

            runner.run_experiment(**kwargs)
            original_source_revisions = runner._source_revisions
            runner._source_revisions = lambda: {
                **original_source_revisions(),
                "context/synthetic.py": "changed-source",
            }
            try:
                with self.assertRaisesRegex(ValueError, "selected manifest run"):
                    runner.run_experiment(**kwargs)
            finally:
                runner._source_revisions = original_source_revisions

    def test_resume_rejects_changed_runtime_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            output_path = root / "raw" / "pilot-trials.jsonl"
            summary_path = root / "processed" / "pilot-summary.csv"
            kwargs = {
                "manifest_path": manifest_path,
                "output_path": output_path,
                "processed_path": summary_path,
                "condition_ids": ("q8_0",),
                "context_lengths": (8192,),
                "repeats": 1,
                "runtime_factory": FakeRuntime,
            }

            runner.run_experiment(**kwargs)
            original_source_revisions = runner._source_revisions
            runner._source_revisions = lambda: {
                **original_source_revisions(),
                "runtimes/llama_cpp.py": "changed-runtime-source",
            }
            try:
                with self.assertRaisesRegex(ValueError, "selected manifest run"):
                    runner.run_experiment(**kwargs)
            finally:
                runner._source_revisions = original_source_revisions

    def test_full_run_can_resume_from_a_pilot_same_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = _manifest_file(root)
            output_path = root / "raw" / "trials.jsonl"
            summary_path = root / "processed" / "summary.csv"

            runner.run_experiment(
                manifest_path=manifest_path,
                output_path=output_path,
                processed_path=summary_path,
                condition_ids=("q8_0",),
                context_lengths=(8192,),
                repeats=1,
                runtime_factory=FakeRuntime,
            )
            result = runner.run_experiment(
                manifest_path=manifest_path,
                output_path=output_path,
                processed_path=summary_path,
                runtime_factory=FakeRuntime,
            )

            self.assertEqual(120, result["expected_trial_n"])
            self.assertEqual(120, result["actual_trial_n"])
            self.assertEqual(3, result["skipped_trial_n"])
            self.assertEqual(120, len(runner.load_trial_results(output_path)))


if __name__ == "__main__":
    unittest.main()
