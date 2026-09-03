import unittest

from llm_lab.analysis.feasibility import classify_feasibility
from llm_lab.evaluation import TrialResult, TrialStatus


def trial(
    length: int,
    task_id: str,
    *,
    status: TrialStatus = TrialStatus.COMPLETED,
    answer_bearing: bool | None = True,
) -> TrialResult:
    score = {"scorer": "calibrated.v1"}
    if answer_bearing is not None:
        score.update(
            {
                "correct": answer_bearing,
                "answer_bearing_correct": answer_bearing,
                "exact_correct": answer_bearing,
                "format_valid": True,
            }
        )
    return TrialResult(
        trial_id=f"exp_001:{task_id}:ctx{length}:run01",
        experiment_id="exp_001",
        task_id=task_id,
        status=status,
        input={
            "task_type": "literal_retrieval",
            "condition_id": f"feasibility:ctx{length:06d}:p050",
            "target_context_tokens": length,
            "requested_evidence_position": 0.50,
        },
        score=score,
    )


class FeasibilityAnalysisTests(unittest.TestCase):
    def test_classifies_all_completed_answer_bearing_trials_as_useful(self) -> None:
        result = classify_feasibility(
            [
                trial(65536, "task.literal.000001"),
                trial(65536, "task.semantic.000001"),
            ],
            expected_task_ids=("task.literal.000001", "task.semantic.000001"),
        )

        self.assertEqual("accepted_and_useful", result["classification"])
        self.assertEqual(2, result["attempted_n"])
        self.assertEqual(2, result["answer_bearing_n"])

    def test_classifies_completed_but_not_answer_bearing_as_not_useful(self) -> None:
        result = classify_feasibility(
            [trial(131072, "task.literal.000001", answer_bearing=False)],
            expected_task_ids=("task.literal.000001",),
        )

        self.assertEqual("accepted_but_not_useful", result["classification"])
        self.assertEqual(0, result["answer_bearing_n"])

    def test_runtime_failure_takes_precedence_and_remains_attempted(self) -> None:
        result = classify_feasibility(
            [
                trial(262144, "task.literal.000001"),
                trial(
                    262144,
                    "task.semantic.000001",
                    status=TrialStatus.TIMEOUT,
                    answer_bearing=None,
                ),
            ],
            expected_task_ids=("task.literal.000001", "task.semantic.000001"),
        )

        self.assertEqual("operational_failure", result["classification"])
        self.assertEqual(2, result["attempted_n"])
        self.assertEqual(1, result["timeout_n"])

    def test_missing_task_is_an_operational_failure(self) -> None:
        result = classify_feasibility(
            [trial(65536, "task.literal.000001")],
            expected_task_ids=("task.literal.000001", "task.semantic.000001"),
        )

        self.assertEqual("operational_failure", result["classification"])
        self.assertEqual(["task.semantic.000001"], result["missing_task_ids"])


if __name__ == "__main__":
    unittest.main()
