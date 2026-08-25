import importlib.util
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

from llm_lab.datasets import TaskCatalog
from llm_lab.evaluation import load_trial_results


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "experiments/exp_001-context_measurement/runner.py"
SPEC = importlib.util.spec_from_file_location("exp_001_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Exp001RunnerTests(unittest.TestCase):
    def test_planned_conditions_match_smoke_and_main_matrix(self) -> None:
        smoke = runner.planned_conditions("smoke")
        main = runner.planned_conditions("main")

        self.assertEqual(6, len(smoke))
        self.assertEqual(25, len(main))
        self.assertEqual("baseline:ctx008192:p005", smoke[0].condition_id)
        self.assertEqual("baseline:ctx262144:p095", main[-1].condition_id)

    def test_build_tasks_records_context_provenance_and_evidence_offsets(self) -> None:
        catalog = TaskCatalog.from_jsonl(ROOT / "data/tasks/core.v001.jsonl")
        condition = runner.planned_conditions("smoke")[0]

        tasks = runner.build_tasks(
            catalog,
            condition,
            fixture_seed=42,
        )

        self.assertEqual(3, len(tasks))
        self.assertEqual("literal_retrieval", tasks[0].task_type)
        request = tasks[0].build_request(
            runner.qwen38_model_spec(),
            runner.SamplingConfig(max_new_tokens=8),
        )
        self.assertEqual(8192, request.metadata["target_context_tokens"])
        self.assertEqual(0.05, request.metadata["requested_evidence_position"])
        self.assertEqual("whitespace-v1", request.metadata["context_tokenization"])
        self.assertEqual(42, request.metadata["fixture_seed"])
        self.assertEqual(
            "aurora-access",
            request.metadata["evidence_spans"][0]["id"],
        )
        self.assertLess(
            request.metadata["evidence_spans"][0]["token_start"],
            request.metadata["evidence_spans"][0]["token_end"],
        )

    def test_build_tasks_accepts_the_inference_tokenizer_for_model_runs(self) -> None:
        class FakeTokenizer:
            name = "fixture-inference-tokenizer"

            def encode(self, text: str) -> list[int]:
                return list(text.encode("utf-8"))

            def decode(self, tokens: list[int]) -> str:
                return bytes(tokens).decode("utf-8")

        catalog = TaskCatalog.from_jsonl(ROOT / "data/tasks/core.v001.jsonl")
        condition = runner.planned_conditions("smoke")[0]

        tasks = runner.build_tasks(
            catalog,
            condition,
            fixture_seed=42,
            tokenizer=FakeTokenizer(),
        )

        request = tasks[0].build_request(
            runner.qwen38_model_spec(),
            runner.SamplingConfig(max_new_tokens=8),
        )
        self.assertEqual(
            "fixture-inference-tokenizer",
            request.metadata["context_tokenization"],
        )
        self.assertEqual("tokenizer", request.metadata["context_tokenization_mode"])
        self.assertEqual(8192, request.metadata["actual_context_tokens"])

    def test_fixture_smoke_can_be_regenerated_without_duplicate_trial_ids(self) -> None:
        results_root = ROOT / "experiments/exp_001-context_measurement/results"
        with TemporaryDirectory(dir=results_root) as temporary_directory:
            directory = Path(temporary_directory)
            output_path = directory / "smoke-trials.jsonl"
            manifest_path = directory / "smoke.json"

            first = runner.run_experiment(
                phase="smoke",
                backend="fixture",
                output_path=output_path,
                manifest_path=manifest_path,
                overwrite_smoke=True,
            )
            with self.assertRaises(FileExistsError):
                runner.run_experiment(
                    phase="smoke",
                    backend="fixture",
                    output_path=output_path,
                    manifest_path=manifest_path,
                )
            second = runner.run_experiment(
                phase="smoke",
                backend="fixture",
                output_path=output_path,
                manifest_path=manifest_path,
                overwrite_smoke=True,
            )

            self.assertEqual(18, first["actual_trial_n"])
            self.assertEqual(18, second["actual_trial_n"])
            persisted = load_trial_results(output_path)
            self.assertEqual(18, len(persisted))
            self.assertEqual(18, len({result.trial_id for result in persisted}))

if __name__ == "__main__":
    unittest.main()
