import csv
import tempfile
import unittest
from pathlib import Path

from llm_lab.analysis.aggregation import (
    aggregate_jsonl,
    aggregate_trials,
    write_summary_csv,
)
from llm_lab.evaluation import TrialResult, TrialStatus
from llm_lab.evaluation.storage import JsonlResultWriter


def trial(
    trial_id: str,
    *,
    task_type: str,
    condition_id: str,
    status: TrialStatus,
    correct: bool | None,
    total_s: float | None,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        experiment_id="exp_fixture",
        task_id=trial_id,
        status=status,
        input={"task_type": task_type, "condition_id": condition_id},
        score={} if correct is None else {"correct": correct, "value": float(correct)},
        timing={} if total_s is None else {"total_s": total_s},
    )


class AggregationTests(unittest.TestCase):
    def test_aggregation_groups_trials_and_excludes_unscored_failures(self) -> None:
        summaries = aggregate_trials(
            [
                trial(
                    "one",
                    task_type="literal_retrieval",
                    condition_id="q8",
                    status=TrialStatus.COMPLETED,
                    correct=True,
                    total_s=1.0,
                ),
                trial(
                    "two",
                    task_type="literal_retrieval",
                    condition_id="q8",
                    status=TrialStatus.COMPLETED,
                    correct=False,
                    total_s=3.0,
                ),
                trial(
                    "three",
                    task_type="literal_retrieval",
                    condition_id="q8",
                    status=TrialStatus.RUNTIME_ERROR,
                    correct=None,
                    total_s=None,
                ),
            ]
        )

        self.assertEqual(1, len(summaries))
        self.assertEqual(
            {
                "experiment_id": "exp_fixture",
                "task_type": "literal_retrieval",
                "condition_id": "q8",
                "n": 3,
                "completed_n": 2,
                "error_n": 1,
                "scored_n": 2,
                "accuracy": 0.5,
                "median_total_s": 2.0,
            },
            {key: summaries[0][key] for key in (
                "experiment_id",
                "task_type",
                "condition_id",
                "n",
                "completed_n",
                "error_n",
                "scored_n",
                "accuracy",
                "median_total_s",
            )},
        )

    def test_aggregation_can_load_jsonl_and_write_notebook_friendly_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "trials.jsonl"
            summary_path = Path(directory) / "processed" / "summary.csv"
            writer = JsonlResultWriter(raw_path)
            writer.append(
                trial(
                    "one",
                    task_type="semantic_retrieval",
                    condition_id="q4",
                    status=TrialStatus.COMPLETED,
                    correct=True,
                    total_s=2.0,
                )
            )

            summaries = aggregate_jsonl(raw_path)
            write_summary_csv(summary_path, summaries)
            with summary_path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))

        self.assertEqual("semantic_retrieval", rows[0]["task_type"])
        self.assertEqual("1.0", rows[0]["accuracy"])
        self.assertEqual("1", rows[0]["scored_n"])


if __name__ == "__main__":
    unittest.main()
