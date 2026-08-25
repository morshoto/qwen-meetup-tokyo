import importlib.util
import sys
import unittest
from pathlib import Path

from llm_lab.datasets import TaskCatalog


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


if __name__ == "__main__":
    unittest.main()
