import unittest

from llm_lab.analysis.equivalence import (
    EquivalenceAnalysisError,
    paired_equivalence_report,
)
from llm_lab.evaluation import TrialResult, TrialStatus


def trial(
    variant: str,
    task_id: str,
    context_tokens: int,
    *,
    correct: bool = True,
    status: TrialStatus = TrialStatus.COMPLETED,
) -> TrialResult:
    return TrialResult(
        trial_id=f"exp:{variant}:{task_id}:{context_tokens}",
        experiment_id="exp",
        task_id=task_id,
        status=status,
        input={
            "variant_label": variant,
            "target_context_tokens": context_tokens,
            "requested_evidence_position": 0.50,
        },
        score={
            "correct": correct,
            "exact_correct": correct,
            "answer_bearing_correct": True,
            "format_valid": correct,
        },
    )


class EquivalenceTests(unittest.TestCase):
    def test_all_metrics_are_equivalent_when_pairs_match(self) -> None:
        rows = []
        for task_id in ("task.1", "task.2", "task.3"):
            for context_tokens in (8192, 32768):
                rows.extend(
                    [
                        trial("Q8_0", task_id, context_tokens),
                        trial("Q4_K_M", task_id, context_tokens),
                    ]
                )

        report = paired_equivalence_report(
            rows,
            reference_variant="Q8_0",
            candidate_variant="Q4_K_M",
            bootstrap_repeats=100,
            seed=7,
        )

        self.assertTrue(report["complete_quality_equivalent"])
        self.assertEqual(6, report["pair_n"])
        self.assertTrue(all(item["decision"] == "equivalent" for item in report["metrics"]))

    def test_report_distinguishes_answer_bearing_from_end_to_end(self) -> None:
        rows = []
        for index in range(20):
            task_id = f"task.{index:02d}"
            q8 = trial("Q8_0", task_id, 8192)
            q4 = trial("Q4_K_M", task_id, 8192, correct=index >= 2)
            rows.extend([q8, q4])

        report = paired_equivalence_report(
            rows,
            reference_variant="Q8_0",
            candidate_variant="Q4_K_M",
            margin=0.10,
            bootstrap_repeats=1000,
        )
        metrics = {item["metric"]: item for item in report["metrics"]}

        self.assertEqual("equivalent", metrics["answer_bearing"]["decision"])
        self.assertNotEqual("equivalent", metrics["end_to_end"]["decision"])
        self.assertFalse(report["complete_quality_equivalent"])

    def test_missing_pairs_and_runtime_failures_fail_closed(self) -> None:
        with self.assertRaisesRegex(EquivalenceAnalysisError, "both variants"):
            paired_equivalence_report(
                [trial("Q8_0", "task.1", 8192)],
                reference_variant="Q8_0",
                candidate_variant="Q4_K_M",
                bootstrap_repeats=10,
            )
        with self.assertRaisesRegex(EquivalenceAnalysisError, "completed trials"):
            paired_equivalence_report(
                [
                    trial("Q8_0", "task.1", 8192),
                    trial(
                        "Q4_K_M",
                        "task.1",
                        8192,
                        status=TrialStatus.TIMEOUT,
                    ),
                ],
                reference_variant="Q8_0",
                candidate_variant="Q4_K_M",
                bootstrap_repeats=10,
            )

    def test_bootstrap_is_deterministic_for_fixed_seed(self) -> None:
        rows = [
            value
            for index in range(4)
            for value in (
                trial("Q8_0", f"task.{index}", 8192),
                trial("Q4_K_M", f"task.{index}", 8192, correct=index != 0),
            )
        ]
        first = paired_equivalence_report(
            rows,
            reference_variant="Q8_0",
            candidate_variant="Q4_K_M",
            bootstrap_repeats=100,
            seed=123,
        )
        second = paired_equivalence_report(
            rows,
            reference_variant="Q8_0",
            candidate_variant="Q4_K_M",
            bootstrap_repeats=100,
            seed=123,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
