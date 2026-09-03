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
RUNNER_PATH = ROOT / "experiments/exp_004-agent_context_growth/runner.py"
SPEC = importlib.util.spec_from_file_location("exp_004_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class FakeRuntime:
    name = "fixture-agent"
    instances: list["FakeRuntime"] = []
    answers = {
        "task.agent.000001": "middleware/auth.ts",
        "task.agent.000002": "src/cache/redis.py",
        "task.agent.000003": "deploy/kubernetes/production.yaml",
        "task.agent.000004": "src/telemetry/exporter.py",
        "task.agent.000005": "config/flags/registry.yaml",
        "task.agent.000006": "services/payments/retry_policy.go",
        "task.agent.000007": "workers/notifications/dispatcher.py",
        "task.agent.000008": "packages/audit/logger.ts",
        "task.agent.000009": "db/migrations/2026_08_add_events.sql",
        "task.agent.000010": "app/routes/settings.tsx",
    }

    def __init__(self) -> None:
        self.loaded: tuple[ModelSpec, RuntimeConfig] | None = None
        self.closed = False
        self.__class__.instances.append(self)

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        self.loaded = (model, config)

    def generate(self, request: Any) -> GenerationResponse:
        task_id = str(request.metadata["task_id"])
        if request.metadata["agent_stage"] == "discovery":
            output = '{"action":"tool","name":"discover_fact","arguments":{}}'
        else:
            output = json.dumps(
                {"action": "answer", "value": self.answers[task_id]}
            )
        return GenerationResponse(
            output_text=output,
            usage=TokenUsage(
                prompt_tokens=len(request.prompt.split()),
                completion_tokens=len(output.split()),
            ),
            timing=GenerationTiming(
                ttft_seconds=0.01,
                prefill_seconds=0.02,
                decode_seconds=0.01,
                total_seconds=0.03,
            ),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version="fixture",
                model_id=request.model.model_id,
                tokenizer_id=request.model.tokenizer_id,
                config={"purpose": "harness-smoke-only"},
            ),
        )

    def close(self) -> None:
        self.closed = True


class EmptyAnswerRuntime(FakeRuntime):
    def generate(self, request: Any) -> GenerationResponse:
        if request.metadata["agent_stage"] != "answer":
            return super().generate(request)
        return GenerationResponse(
            output_text="",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=0),
            timing=GenerationTiming(total_seconds=0.01),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version="fixture",
                model_id=request.model.model_id,
                tokenizer_id=request.model.tokenizer_id,
                config={"purpose": "harness-smoke-only"},
            ),
        )


def _source_manifest(directory: Path) -> Path:
    variants = []
    for condition_id, label, quantization_type, bits in (
        ("q8_0", "Q8_0", "Q8_0", 8),
        ("q4_k_m", "Q4_K_M", "Q4_K_M", 4),
    ):
        artifact_path = directory / f"{condition_id}.gguf"
        artifact_path.write_bytes(f"fixture-{condition_id}".encode("ascii"))
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
        experiment_id="exp_003",
        model_id="Qwen/Qwen3.8-27B",
        model_revision="model-sha",
        tokenizer_id="Qwen/Qwen3.8-27B",
        tokenizer_revision="tokenizer-sha",
        runtime_name="llama.cpp",
        runtime_version="llama-cpp-python-fixture",
        runtime_options={"n_ctx": 33088, "n_batch": 512},
        prompt_id="prompt.agent.v001",
        task_ids=(
            "task.literal.000001",
            "task.semantic.000001",
            "task.multihop.000001",
        ),
        context_lengths=(8192,),
        sampling={"max_new_tokens": 64, "temperature": 0.0, "seed": 42},
        variants=tuple(variants),
        repeats=5,
    )
    path = directory / "source-manifest.json"
    path.write_text(json.dumps(manifest.to_record()), encoding="utf-8")
    return path


def _source_run_manifest(directory: Path) -> Path:
    source = _source_manifest(directory)
    source_record = json.loads(source.read_text(encoding="utf-8"))
    source_record["experiment_id"] = "exp_002"
    resolved_source = directory / "resolved-exp002.json"
    resolved_source.write_text(json.dumps(source_record), encoding="utf-8")
    source_sha256 = hashlib.sha256(resolved_source.read_bytes()).hexdigest()
    run_record = {
        "experiment_id": "exp_003",
        "backend": "llama.cpp",
        "source_manifest": str(resolved_source),
        "source_manifest_sha256": source_sha256,
        "model": source_record["model"],
        "runtime": {
            "name": "llama.cpp",
            "version": "llama-cpp-python-fixture",
            "source_options": {"n_ctx": 33088, "n_batch": 512},
        },
        "quantization_variants": source_record["variants"],
        "task_ids": [
            "task.literal.000001",
            "task.semantic.000001",
            "task.multihop.000001",
        ],
        "context_lengths": [8192, 32768],
        "prompt_id": "prompt.qa.v001",
        "repeats": 1,
    }
    path = directory / "exp003-run-manifest.json"
    path.write_text(json.dumps(run_record), encoding="utf-8")
    return path


class Exp004RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeRuntime.instances = []

    def test_runner_reuses_controlled_observations_across_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _source_manifest(root)
            output_path = root / "raw" / "smoke-trials.jsonl"
            manifest_path = root / "manifests" / "smoke.json"
            summary_path = root / "processed" / "smoke-summary.csv"

            result = runner.run_experiment(
                source_manifest_path=source_manifest,
                output_path=output_path,
                manifest_output_path=manifest_path,
                processed_path=summary_path,
                phase="smoke",
                backend="fixture",
                condition_ids=("q8_0", "q4_k_m"),
                trajectory_lengths=(4,),
                critical_positions=(0.25, 0.75),
                repeats=1,
                runtime_factory=FakeRuntime,
            )

            trials = runner.load_trial_results(output_path)
            by_context = {}
            for trial in trials:
                by_context.setdefault(
                    trial.input["context_instance_id"], []
                ).append(trial)

            self.assertEqual(40, result["actual_trial_n"])
            self.assertEqual(20, len(by_context))
            for matched_trials in by_context.values():
                self.assertEqual(2, len(matched_trials))
                self.assertEqual(
                    {item.input["environment_fingerprint"] for item in matched_trials},
                    {matched_trials[0].input["environment_fingerprint"]},
                )
                self.assertEqual(
                    matched_trials[0].input["trajectory"],
                    matched_trials[1].input["trajectory"],
                )
                self.assertTrue(matched_trials[0].score["correct"])
                self.assertGreater(len(matched_trials[0].input["trajectory"]), 0)
            self.assertTrue(summary_path.is_file())
            summary_header = summary_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("final_task_success", summary_header)
            self.assertIn("failure_category_counts", summary_header)
            self.assertTrue(all(instance.closed for instance in FakeRuntime.instances))

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["fixture_only"])
            self.assertEqual("exp_003", manifest["source_manifest"]["experiment_id"])
            self.assertEqual(2, len(manifest["source_manifest"]["variants"]))
            self.assertEqual(10, manifest["protocol"]["independent_task_n"])
            self.assertEqual("fixture-agent", manifest["effective_runtime"]["name"])
            self.assertEqual("model-sha", trials[0].model["revision"])
            self.assertEqual(
                "fixture-agent",
                trials[0].runtime["name"],
            )
            self.assertEqual(128, trials[0].input["sampling"]["max_new_tokens"])
            self.assertEqual(
                "single_json_object", trials[0].input["output_policy"]["format"]
            )
            self.assertEqual(3, trials[0].input["retry_policy"]["max_attempts"])
            self.assertEqual(
                {"trajectory_length": 1, "critical_position": 0.5},
                {
                    key: manifest["protocol"]["one_turn_control"][key]
                    for key in ("trajectory_length", "critical_position")
                },
            )

    def test_recheck_protocol_includes_one_turn_control_and_repeats(self) -> None:
        controls = runner.planned_conditions("recheck")
        self.assertEqual([1, 4, 8, 16, 32], [item.trajectory_length for item in controls])
        self.assertEqual([0.5] * 5, [item.critical_position for item in controls])
        self.assertEqual(0, controls[0].pre_discovery_steps)
        self.assertEqual(0, controls[0].post_discovery_steps)
        self.assertEqual(0.0, controls[0].actual_critical_position)
        with tempfile.TemporaryDirectory() as directory:
            source_manifest = _source_manifest(Path(directory))
            self.assertEqual(
                300,
                runner.expected_trial_count(
                    runner.load_source_manifest(source_manifest),
                    runner.load_tasks(),
                    phase="recheck",
                ),
            )

    def test_runner_loads_exp003_run_manifest_with_exp002_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _source_run_manifest(root)
            result = runner.run_experiment(
                source_manifest_path=source_manifest,
                output_path=root / "raw" / "trials.jsonl",
                manifest_output_path=root / "manifests" / "run.json",
                processed_path=root / "processed" / "summary.csv",
                phase="smoke",
                backend="fixture",
                condition_ids=("q8_0",),
                trajectory_lengths=(4,),
                critical_positions=(0.5,),
                repeats=1,
                runtime_factory=FakeRuntime,
            )

            self.assertEqual(10, result["actual_trial_n"])
            manifest = json.loads(
                (root / "manifests" / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual("exp_003", manifest["source_manifest"]["experiment_id"])
            self.assertTrue(Path(manifest["source_manifest"]["variants"][0]["artifact"]["artifact_uri"]).is_absolute())

    def test_runner_resumes_without_duplicate_trial_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _source_manifest(root)
            paths = {
                "source_manifest_path": source_manifest,
                "output_path": root / "raw" / "trials.jsonl",
                "manifest_output_path": root / "manifests" / "run.json",
                "processed_path": root / "processed" / "summary.csv",
                "phase": "smoke",
                "backend": "fixture",
                "condition_ids": ("q8_0",),
                "trajectory_lengths": (4,),
                "critical_positions": (0.5,),
                "repeats": 1,
                "runtime_factory": FakeRuntime,
            }

            runner.run_experiment(**paths)
            second = runner.run_experiment(**paths)

            self.assertEqual(10, second["expected_trial_n"])
            self.assertEqual(10, second["skipped_trial_n"])
            self.assertEqual(10, len(runner.load_trial_results(paths["output_path"])))
            manifest = json.loads(paths["manifest_output_path"].read_text(encoding="utf-8"))
            self.assertEqual("fixture-agent", manifest["effective_runtime"]["name"])

    def test_empty_model_response_is_recorded_as_invalid_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _source_manifest(root)
            output_path = root / "raw" / "trials.jsonl"
            runner.run_experiment(
                source_manifest_path=source_manifest,
                output_path=output_path,
                manifest_output_path=root / "manifests" / "run.json",
                processed_path=root / "processed" / "summary.csv",
                phase="smoke",
                backend="fixture",
                condition_ids=("q8_0",),
                trajectory_lengths=(4,),
                critical_positions=(0.5,),
                repeats=1,
                runtime_factory=EmptyAnswerRuntime,
            )

            trials = runner.load_trial_results(output_path)
            self.assertEqual(10, len(trials))
            self.assertTrue(all(trial.status.value == "invalid_output" for trial in trials))
            self.assertTrue(
                all(
                    any(
                        event["content"] == "[empty model response]"
                        for event in trial.input["trajectory"]
                    )
                    for trial in trials
                )
            )

    def test_real_backend_rejects_changed_selected_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = _source_manifest(root)
            (root / "q8_0.gguf").write_bytes(b"changed-after-manifest")

            with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
                runner.run_experiment(
                    source_manifest_path=source_manifest,
                    output_path=root / "raw" / "trials.jsonl",
                    manifest_output_path=root / "manifests" / "run.json",
                    processed_path=root / "processed" / "summary.csv",
                    phase="pilot",
                    backend="llama.cpp",
                    condition_ids=("q8_0",),
                    trajectory_lengths=(4,),
                    critical_positions=(0.5,),
                    repeats=1,
                )


if __name__ == "__main__":
    unittest.main()
