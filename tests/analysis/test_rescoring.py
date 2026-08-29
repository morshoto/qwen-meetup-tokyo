import unittest

from llm_lab.analysis.rescoring import rescore_trial
from llm_lab.evaluation import EvaluationTask, TrialResult, TrialStatus


def task(
    task_id: str,
    task_type: str,
    expected: dict[str, object],
) -> EvaluationTask:
    return EvaluationTask(
        task_id=task_id,
        task_type=task_type,
        question="What is the answer?",
        context="The answer is in the evidence.",
        expected=expected,
    )


def trial(
    task_id: str,
    output: str,
    *,
    old_correct: bool,
) -> TrialResult:
    return TrialResult(
        trial_id=f"exp_002:{task_id}:q8_0:ctx8192:run01",
        experiment_id="exp_002",
        task_id=task_id,
        status=TrialStatus.COMPLETED,
        input={
            "task_type": "literal_retrieval",
            "condition_id": "q8_0:ctx8192",
            "variant_condition_id": "q8_0",
            "target_context_tokens": 8192,
        },
        generation={"output_text": output},
        score={
            "correct": old_correct,
            "value": float(old_correct),
            "scorer": "expected.v1",
        },
    )


class RescoringTests(unittest.TestCase):
    def test_completed_trials_are_classified_by_calibrated_outcome(self) -> None:
        cases = (
            (
                "task.semantic.000001",
                task(
                    "task.semantic.000001",
                    "semantic_retrieval",
                    {
                        "type": "normalized_exact",
                        "value": "reliability engineering",
                        "accepted": ["reliability engineering", "reliability team"],
                    },
                ),
                "The Reliability Engineering group",
                "mismatch",
                False,
                True,
                True,
            ),
            (
                "task.literal.000001",
                task(
                    "task.literal.000001",
                    "literal_retrieval",
                    {"type": "exact", "value": "ZX-4817"},
                ),
                "ZX-4817.659",
                "format_failure",
                False,
                True,
                False,
            ),
            (
                "task.multihop.000001",
                task(
                    "task.multihop.000001",
                    "multi_hop",
                    {"type": "exact", "value": "8392"},
                ),
                "8392",
                "exact_match",
                True,
                True,
                True,
            ),
        )

        for (
            task_id,
            evaluation_task,
            output,
            expected_category,
            expected_exact,
            expected_answer_bearing,
            expected_format,
        ) in cases:
            with self.subTest(task_id=task_id):
                result = rescore_trial(
                    trial(task_id, output, old_correct=False),
                    evaluation_task,
                )

                self.assertEqual("expected.v1", result.trial.score["scorer"])
                self.assertFalse(result.legacy_correct)
                self.assertEqual("calibrated.v1", result.score.scorer)
                self.assertEqual(expected_category, result.category)
                self.assertEqual(expected_exact, result.score.exact_correct)
                self.assertEqual(
                    expected_answer_bearing,
                    result.score.answer_bearing_correct,
                )
                self.assertEqual(expected_format, result.score.format_valid)


if __name__ == "__main__":
    unittest.main()
