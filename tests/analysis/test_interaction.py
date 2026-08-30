import unittest

from llm_lab.analysis.interaction import (
    InteractionAnalysisError,
    effective_context_by_variant_and_task,
    interaction_report,
    matched_cell_rows,
    relative_degradation_rows,
)


VARIANTS = ("q8_0", "q4_k_m")
CONTEXT_LENGTHS = (8192, 32768, 65536)
POSITIONS = (0.05, 0.50)
TASK_TYPES = ("literal_retrieval",)
TASK_IDS = ("task.literal.000001", "task.literal.000002")


def summary(
    variant: str,
    context_tokens: int,
    position: float,
    accuracy: float,
    *,
    task_type: str = "literal_retrieval",
    scored_n: int = 10,
) -> dict[str, object]:
    instance_id = (
        f"task.literal.000001:seed1234:ctx{context_tokens}:"
        f"p{int(position * 100):03d}"
    )
    return {
        "experiment_id": "exp_003",
        "task_type": task_type,
        "condition_id": f"{variant}:ctx{context_tokens}:p{int(position * 100):03d}",
        "variant_condition_id": variant,
        "target_context_tokens": context_tokens,
        "requested_evidence_position": position,
        "actual_evidence_position": position,
        "context_instance_id": instance_id,
        "context_sha256": "a" * 64,
        "n": scored_n,
        "attempted_n": scored_n,
        "completed_n": scored_n,
        "error_n": 0,
        "scored_n": scored_n,
        "correct_n": int(accuracy * scored_n),
        "accuracy": accuracy,
    }


def complete_rows(
    q8_accuracies: dict[int, float],
    q4_accuracies: dict[int, float],
    *,
    positions: tuple[float, ...] = POSITIONS,
) -> list[dict[str, object]]:
    rows = []
    for context_tokens in q8_accuracies:
        for position in positions:
            rows.extend(
                (
                    summary("q8_0", context_tokens, position, q8_accuracies[context_tokens]),
                    summary("q4_k_m", context_tokens, position, q4_accuracies[context_tokens]),
                )
            )
    return rows


def independent_rows(
    q8_accuracies: dict[str, dict[int, float]],
    q4_accuracies: dict[str, dict[int, float]],
) -> list[dict[str, object]]:
    rows = []
    for task_id in TASK_IDS:
        for context_tokens in q8_accuracies[task_id]:
            for position in (0.50,):
                for variant, accuracies in (
                    ("q8_0", q8_accuracies),
                    ("q4_k_m", q4_accuracies),
                ):
                    row = summary(
                        variant,
                        context_tokens,
                        position,
                        accuracies[task_id][context_tokens],
                    )
                    row["task_id"] = task_id
                    row["context_instance_id"] = (
                        f"{task_id}:seed1234:ctx{context_tokens}:p050"
                    )
                    rows.append(row)
    return rows


class InteractionAnalysisTests(unittest.TestCase):
    def test_matched_cells_require_complete_dimensions_and_shared_context_identity(self) -> None:
        rows = complete_rows(
            {8192: 1.0, 32768: 0.9, 65536: 0.8},
            {8192: 0.9, 32768: 0.7, 65536: 0.2},
        )

        matched = matched_cell_rows(
            rows,
            variant_ids=VARIANTS,
            context_lengths=CONTEXT_LENGTHS,
            evidence_positions=POSITIONS,
            task_types=TASK_TYPES,
        )

        self.assertEqual(12, len(matched))
        self.assertEqual(
            1,
            len({
                row["context_instance_id"]
                for row in matched
                if row["target_context_tokens"] == 32768
                and row["requested_evidence_position"] == 0.50
            }),
        )
        with self.assertRaisesRegex(InteractionAnalysisError, "missing matched cells"):
            matched_cell_rows(
                rows[:-1],
                variant_ids=VARIANTS,
                context_lengths=CONTEXT_LENGTHS,
                evidence_positions=POSITIONS,
                task_types=TASK_TYPES,
            )

    def test_relative_degradation_uses_each_variant_short_context_baseline(self) -> None:
        rows = complete_rows(
            {8192: 1.0, 32768: 0.9, 65536: 0.8},
            {8192: 0.9, 32768: 0.7, 65536: 0.2},
            positions=(0.50,),
        )
        degradation = relative_degradation_rows(
            matched_cell_rows(
                rows,
                variant_ids=VARIANTS,
                context_lengths=CONTEXT_LENGTHS,
                evidence_positions=(0.50,),
                task_types=TASK_TYPES,
            ),
            baseline_context_tokens=8192,
        )

        q4_long = next(
            row
            for row in degradation
            if row["variant_condition_id"] == "q4_k_m"
            and row["target_context_tokens"] == 65536
        )
        self.assertEqual(0.9, q4_long["short_context_baseline_accuracy"])
        self.assertAlmostEqual(0.7, q4_long["accuracy_degradation"])
        self.assertAlmostEqual(0.7 / 0.9, q4_long["relative_degradation"])

    def test_interaction_report_identifies_context_dependent_gap(self) -> None:
        rows = complete_rows(
            {8192: 1.0, 32768: 1.0, 65536: 1.0},
            {8192: 0.9, 32768: 0.8, 65536: 0.2},
            positions=(0.50,),
        )

        [report] = interaction_report(
            matched_cell_rows(
                rows,
                variant_ids=VARIANTS,
                context_lengths=CONTEXT_LENGTHS,
                evidence_positions=(0.50,),
                task_types=TASK_TYPES,
            ),
            reference_variant="q8_0",
            approx_constant_gap_tolerance=0.10,
        )

        self.assertEqual("q4_k_m", report["variant_condition_id"])
        self.assertEqual("context_dependent", report["classification"])
        self.assertAlmostEqual(0.1, report["shortest_context_gap"])
        self.assertAlmostEqual(0.8, report["largest_context_gap"])
        self.assertAlmostEqual(0.7, report["gap_change"])
        self.assertEqual(3 * 10, report["matched_n"])

    def test_effective_context_is_grouped_by_variant_and_task(self) -> None:
        rows = complete_rows(
            {8192: 1.0, 32768: 1.0, 65536: 1.0},
            {8192: 1.0, 32768: 0.5, 65536: 0.4},
        )

        results = effective_context_by_variant_and_task(
            matched_cell_rows(
                rows,
                variant_ids=VARIANTS,
                context_lengths=CONTEXT_LENGTHS,
                evidence_positions=POSITIONS,
                task_types=TASK_TYPES,
            )
        )
        q4 = next(row for row in results if row["variant_condition_id"] == "q4_k_m")
        q8 = next(row for row in results if row["variant_condition_id"] == "q8_0")

        self.assertEqual("literal_retrieval", q4["task_type"])
        self.assertEqual("estimated", q4["status"])
        self.assertEqual(8192, q4["effective_context_tokens"])
        self.assertEqual("right_censored", q8["status"])

    def test_task_aware_matching_preserves_independent_tasks(self) -> None:
        rows = independent_rows(
            {
                TASK_IDS[0]: {8192: 1.0, 32768: 1.0, 65536: 1.0},
                TASK_IDS[1]: {8192: 1.0, 32768: 1.0, 65536: 1.0},
            },
            {
                TASK_IDS[0]: {8192: 0.9, 32768: 0.8, 65536: 0.2},
                TASK_IDS[1]: {8192: 0.8, 32768: 0.7, 65536: 0.0},
            },
        )

        matched = matched_cell_rows(
            rows,
            variant_ids=VARIANTS,
            context_lengths=CONTEXT_LENGTHS,
            evidence_positions=(0.50,),
            task_types=TASK_TYPES,
            task_ids=TASK_IDS,
        )

        self.assertEqual(12, len(matched))
        degradation = relative_degradation_rows(
            matched,
            baseline_context_tokens=8192,
        )
        q4_task_two_long = next(
            row
            for row in degradation
            if row["task_id"] == TASK_IDS[1]
            and row["variant_condition_id"] == "q4_k_m"
            and row["target_context_tokens"] == 65536
        )
        self.assertAlmostEqual(0.8, q4_task_two_long["accuracy_degradation"])

        [report] = interaction_report(matched, reference_variant="q8_0")
        self.assertEqual("context_dependent", report["classification"])
        self.assertAlmostEqual(0.15, report["shortest_context_gap"])
        self.assertAlmostEqual(0.90, report["largest_context_gap"])
        self.assertEqual(20, report["context_points"][-1]["matched_n"])

    def test_task_aware_matching_rejects_missing_independent_task_cell(self) -> None:
        rows = independent_rows(
            {task_id: {8192: 1.0, 32768: 1.0, 65536: 1.0} for task_id in TASK_IDS},
            {task_id: {8192: 1.0, 32768: 1.0, 65536: 1.0} for task_id in TASK_IDS},
        )

        with self.assertRaisesRegex(InteractionAnalysisError, "missing matched cells"):
            matched_cell_rows(
                rows[:-1],
                variant_ids=VARIANTS,
                context_lengths=CONTEXT_LENGTHS,
                evidence_positions=(0.50,),
                task_types=TASK_TYPES,
                task_ids=TASK_IDS,
            )

    def test_interaction_report_classifies_insufficient_data(self) -> None:
        rows = complete_rows(
            {8192: 1.0},
            {8192: 0.9},
            positions=(0.50,),
        )

        [report] = interaction_report(
            matched_cell_rows(
                rows,
                variant_ids=VARIANTS,
                context_lengths=(8192,),
                evidence_positions=(0.50,),
                task_types=TASK_TYPES,
            ),
            reference_variant="q8_0",
        )

        self.assertEqual("insufficient_data", report["classification"])


if __name__ == "__main__":
    unittest.main()
