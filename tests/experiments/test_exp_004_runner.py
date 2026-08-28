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
        task_ids=("task.agent.000001", "task.agent.000002"),
        context_lengths=(8192,),
        sampling={"max_new_tokens": 64, "temperature": 0.0, "seed": 42},
        variants=tuple(variants),
        repeats=5,
    )
    path = directory / "source-manifest.json"
    path.write_text(json.dumps(manifest.to_record()), encoding="utf-8")
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

            self.assertEqual(8, result["actual_trial_n"])
            self.assertEqual(4, len(by_context))
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
            self.assertTrue(all(instance.closed for instance in FakeRuntime.instances))

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["fixture_only"])
            self.assertEqual("exp_003", manifest["source_manifest"]["experiment_id"])
            self.assertEqual(2, len(manifest["source_manifest"]["variants"]))
            self.assertEqual("fixture-agent", manifest["effective_runtime"]["name"])
            self.assertEqual("model-sha", trials[0].model["revision"])
            self.assertEqual(
                "fixture-agent",
                trials[0].runtime["name"],
            )

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

            self.assertEqual(2, second["expected_trial_n"])
            self.assertEqual(2, second["skipped_trial_n"])
            self.assertEqual(2, len(runner.load_trial_results(paths["output_path"])))


if __name__ == "__main__":
    unittest.main()
