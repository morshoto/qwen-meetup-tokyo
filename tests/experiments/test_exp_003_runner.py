import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_lab.evaluation import load_trial_results
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
RUNNER_PATH = ROOT / "experiments/exp_003-context_x_quantization/runner.py"
SPEC = importlib.util.spec_from_file_location("exp_003_runner", RUNNER_PATH)
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
        answers = {
            "task.literal.000001": "ZX-4817",
            "task.semantic.000001": "Reliability Engineering",
            "task.multihop.000001": "8392",
        }
        output = answers[str(request.metadata["task_id"])]
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
                config={
                    "model_path": (
                        self.loaded[1].options["model_path"]
                        if self.loaded is not None
                        and "model_path" in self.loaded[1].options
                        else None
                    )
                },
            ),
        )

    def close(self) -> None:
        self.closed = True


def _manifest_file(directory: Path) -> Path:
    variants = []
    for index, (condition_id, label, quantization_type, bits) in enumerate(
        (
            ("q8_0", "Q8_0", "Q8_0", 8),
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
                    artifact_sha256=hashlib.sha256(
                        artifact_path.read_bytes()
                    ).hexdigest(),
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
        repeats=1,
        context_length_semantics="input_tokens",
        context_overhead_tokens=256,
    )
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest.to_record()), encoding="utf-8")
    return path


class Exp003RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeRuntime.instances = []

    def test_runner_uses_config_for_phase_and_variant_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(
                """experiment:
  default_phase: smoke
quantization:
  variants: [q4_k_m]
phases:
  smoke:
    lengths: [123]
    evidence_positions: [0.25]
    repeats: 2
    backend: fixture
""",
                encoding="utf-8",
            )
            source_manifest = _manifest_file(root)

            result = runner.run_experiment(
                source_manifest_path=source_manifest,
                output_path=root / "raw" / "smoke-trials.jsonl",
                manifest_output_path=root / "manifests" / "smoke.json",
                processed_path=root / "processed" / "summary.csv",
                phase="smoke",
                runtime_factory=FakeRuntime,
                config_path=config_path,
            )

            self.assertEqual(6, result["expected_trial_n"])
            self.assertEqual(6, result["actual_trial_n"])
            self.assertEqual(
                123,
                json.loads(
                    (root / "manifests" / "smoke.json").read_text(encoding="utf-8")
                )["context_lengths"][0],
            )
            self.assertEqual("Q4_K_M", FakeRuntime.instances[0].loaded[1].options["quantization_type"])

    def test_smoke_plans_context_and_position_cells(self) -> None:
        conditions = runner.planned_conditions("smoke")

        self.assertEqual(4, len(conditions))
        self.assertEqual("ctx008192:p005", conditions[0].condition_id)
        self.assertEqual("ctx032768:p050", conditions[-1].condition_id)

    def test_runner_reuses_context_instance_across_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _manifest_file(root)
            output_path = root / "raw" / "smoke-trials.jsonl"
            run_manifest_path = root / "manifests" / "smoke.json"
            summary_path = root / "processed" / "summary.csv"

            result = runner.run_experiment(
                source_manifest_path=source_manifest,
                output_path=output_path,
                manifest_output_path=run_manifest_path,
                processed_path=summary_path,
                phase="smoke",
                backend="fixture",
                condition_ids=("q8_0", "q4_k_m"),
                runtime_factory=FakeRuntime,
            )

            self.assertEqual(24, result["expected_trial_n"])
            self.assertEqual(24, result["actual_trial_n"])
            self.assertEqual(24, result["summary_row_n"])
            self.assertTrue(summary_path.is_file())
            self.assertTrue(all(instance.closed for instance in FakeRuntime.instances))

            records = load_trial_results(output_path)
            grouped: dict[tuple[str, int, float], list[Any]] = defaultdict(list)
            for record in records:
                grouped[
                    (
                        record.task_id,
                        int(record.input["target_context_tokens"]),
                        float(record.input["requested_evidence_position"]),
                    )
                ].append(record)

            self.assertEqual(12, len(grouped))
            for matched_records in grouped.values():
                self.assertEqual(2, len(matched_records))
                self.assertEqual(
                    1,
                    len({record.input["context_instance_id"] for record in matched_records}),
                )
                self.assertEqual(
                    1,
                    len({record.input["context_sha256"] for record in matched_records}),
                )
                self.assertEqual(
                    {"q8_0", "q4_k_m"},
                    {record.input["variant_condition_id"] for record in matched_records},
                )
                for record in matched_records:
                    self.assertEqual("fixture", record.runtime["version"])
                    self.assertEqual(64, len(record.input["artifact_sha256"]))

            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("exp_003", manifest["experiment_id"])
            self.assertEqual(24, manifest["planned_trial_n"])
            self.assertEqual(24, manifest["actual_trial_n"])
            self.assertEqual(24, len(manifest["coverage"]))
            self.assertEqual(0, len(manifest["excluded_cells"]))
            self.assertEqual([8192, 32768], manifest["context_lengths"])
            self.assertEqual([0.05, 0.50], manifest["evidence_positions"])
            self.assertEqual(
                ["literal_retrieval", "semantic_retrieval", "multi_hop"],
                manifest["task_types"],
            )
            self.assertEqual(
                runner._sha256(source_manifest), manifest["source_manifest_sha256"]
            )

    def test_runner_resumes_without_duplicate_trial_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _manifest_file(root)
            kwargs = {
                "source_manifest_path": source_manifest,
                "output_path": root / "raw" / "smoke-trials.jsonl",
                "manifest_output_path": root / "manifests" / "smoke.json",
                "processed_path": root / "processed" / "summary.csv",
                "phase": "smoke",
                "backend": "fixture",
                "condition_ids": ("q8_0", "q4_k_m"),
                "runtime_factory": FakeRuntime,
            }

            runner.run_experiment(**kwargs)
            result = runner.run_experiment(**kwargs)

            self.assertEqual(24, result["actual_trial_n"])
            self.assertEqual(24, result["skipped_trial_n"])
            records = load_trial_results(kwargs["output_path"])
            self.assertEqual(24, len(records))
            self.assertEqual(24, len({record.trial_id for record in records}))

    def test_fixture_and_measured_runs_have_distinct_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = runner.load_manifest(_manifest_file(Path(directory)))
            conditions = runner.planned_conditions("smoke")

            fixture_fingerprint = runner._run_fingerprint(
                manifest,
                phase="smoke",
                backend="fixture",
                variant_ids=("q8_0",),
                conditions=conditions,
                repeats=1,
                fixture_seed=42,
            )
            measured_fingerprint = runner._run_fingerprint(
                manifest,
                phase="smoke",
                backend="llama.cpp",
                variant_ids=("q8_0",),
                conditions=conditions,
                repeats=1,
                fixture_seed=42,
            )

            self.assertNotEqual(fixture_fingerprint, measured_fingerprint)


if __name__ == "__main__":
    unittest.main()
