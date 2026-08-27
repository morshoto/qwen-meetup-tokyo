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
            self.assertEqual(3, len(runner.load_trial_results(output_path)))
            self.assertTrue(summary_path.is_file())
            self.assertTrue(all(instance.closed for instance in FakeRuntime.instances))

    def test_full_selection_has_120_expected_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = _manifest_file(Path(directory))
            manifest = runner.load_manifest(manifest_path)

            self.assertEqual(120, runner.expected_trial_count(manifest))

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
