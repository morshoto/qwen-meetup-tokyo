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
    metadata: dict[str, object] | None = None,
) -> TrialResult:
    input_data = {"task_type": task_type, "condition_id": condition_id}
    input_data.update(metadata or {})
    return TrialResult(
        trial_id=trial_id,
        experiment_id="exp_fixture",
        task_id=trial_id,
        status=status,
        input=input_data,
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
        self.assertEqual(3, summaries[0]["attempted_n"])
        self.assertEqual(1, summaries[0]["correct_n"])
        self.assertEqual(1, summaries[0]["failure_n"])
        self.assertEqual(1 / 3, summaries[0]["end_to_end_success"])
        self.assertEqual(1 / 3, summaries[0]["failure_rate"])

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
            self.assertNotIn(b"\r", summary_path.read_bytes())

        self.assertEqual("semantic_retrieval", rows[0]["task_type"])
        self.assertEqual("1.0", rows[0]["accuracy"])
        self.assertEqual("1", rows[0]["scored_n"])

    def test_aggregation_preserves_context_dimensions_for_notebook_analysis(self) -> None:
        summaries = aggregate_trials(
            [
                trial(
                    "one",
                    task_type="literal_retrieval",
                    condition_id="ctx8192:p005",
                    status=TrialStatus.COMPLETED,
                    correct=True,
                    total_s=1.0,
                    metadata={
                        "target_context_tokens": 8192,
                        "requested_evidence_position": 0.05,
                        "actual_evidence_position": 0.047,
                    },
                ),
                trial(
                    "two",
                    task_type="literal_retrieval",
                    condition_id="ctx8192:p005",
                    status=TrialStatus.COMPLETED,
                    correct=False,
                    total_s=1.0,
                    metadata={
                        "target_context_tokens": 8192,
                        "requested_evidence_position": 0.05,
                        "actual_evidence_position": 0.053,
                    },
                ),
            ]
        )

        self.assertEqual(8192, summaries[0]["target_context_tokens"])
        self.assertEqual(0.05, summaries[0]["requested_evidence_position"])
        self.assertAlmostEqual(0.05, summaries[0]["actual_evidence_position"])

    def test_aggregation_preserves_variant_id_when_execution_condition_is_scoped(self) -> None:
        summaries = aggregate_trials(
            [
                trial(
                    "one",
                    task_type="literal_retrieval",
                    condition_id="q8_0:ctx8192",
                    status=TrialStatus.COMPLETED,
                    correct=True,
                    total_s=1.0,
                    metadata={"variant_condition_id": "q8_0"},
                )
            ]
        )

        self.assertEqual("q8_0", summaries[0]["variant_condition_id"])

    def test_aggregation_reports_calibrated_metrics_and_failure_kinds(self) -> None:
        calibrated_one = TrialResult(
            trial_id="calibrated-one",
            experiment_id="exp_fixture",
            task_id="calibrated-one",
            status=TrialStatus.COMPLETED,
            input={"task_type": "literal_retrieval", "condition_id": "calibrated"},
            score={
                "correct": True,
                "value": 1.0,
                "scorer": "calibrated.v1",
                "exact_correct": True,
                "answer_bearing_correct": True,
                "format_valid": True,
            },
        )
        calibrated_two = TrialResult(
            trial_id="calibrated-two",
            experiment_id="exp_fixture",
            task_id="calibrated-two",
            status=TrialStatus.COMPLETED,
            input={"task_type": "literal_retrieval", "condition_id": "calibrated"},
            score={
                "correct": False,
                "value": 0.0,
                "scorer": "calibrated.v1",
                "exact_correct": False,
                "answer_bearing_correct": True,
                "format_valid": False,
            },
        )
        runtime_failure = TrialResult(
            trial_id="calibrated-runtime",
            experiment_id="exp_fixture",
            task_id="calibrated-runtime",
            status=TrialStatus.RUNTIME_ERROR,
            input={"task_type": "literal_retrieval", "condition_id": "calibrated"},
            score={"scorer": "calibrated.v1"},
        )

        [summary] = aggregate_trials([calibrated_one, calibrated_two, runtime_failure])

        self.assertEqual("calibrated.v1", summary["scorer_version"])
        self.assertEqual(1, summary["exact_correct_n"])
        self.assertEqual(2, summary["exact_scored_n"])
        self.assertEqual(0.5, summary["exact_accuracy"])
        self.assertEqual(2, summary["answer_bearing_correct_n"])
        self.assertEqual(2, summary["answer_bearing_scored_n"])
        self.assertEqual(1.0, summary["answer_bearing_accuracy"])
        self.assertEqual(1, summary["format_valid_n"])
        self.assertEqual(2, summary["format_scored_n"])
        self.assertEqual(0.5, summary["format_validity"])
        self.assertEqual(1, summary["runtime_error_n"])
        self.assertEqual(0, summary["scorer_error_n"])
        self.assertEqual(0, summary["invalid_output_n"])


if __name__ == "__main__":
    unittest.main()
