import unittest

from llm_lab.analysis.effective_context import (
    effective_context_by_task,
    effective_context_by_task_and_position,
    missing_context_cells,
    position_gap_rows,
    position_curve_rows,
)


def summary(
    task_type: str,
    context_tokens: int,
    position: float,
    accuracy: float,
    *,
    scored_n: int = 10,
) -> dict[str, object]:
    return {
        "experiment_id": "exp_001",
        "task_type": task_type,
        "condition_id": f"ctx{context_tokens}:p{int(position * 1000):03d}",
        "target_context_tokens": context_tokens,
        "requested_evidence_position": position,
        "actual_evidence_position": position,
        "n": scored_n,
        "completed_n": scored_n,
        "error_n": 0,
        "scored_n": scored_n,
        "accuracy": accuracy,
    }


class EffectiveContextTests(unittest.TestCase):
    def test_position_gap_uses_edge_minus_middle_accuracy(self) -> None:
        rows = [
            summary("literal_retrieval", 8192, 0.05, 0.8, scored_n=5),
            summary("literal_retrieval", 8192, 0.50, 0.4, scored_n=10),
            summary("literal_retrieval", 8192, 0.95, 0.6, scored_n=15),
        ]

        [gap] = position_gap_rows(rows)

        self.assertEqual("literal_retrieval", gap["task_type"])
        self.assertEqual(8192, gap["target_context_tokens"])
        self.assertAlmostEqual(0.65, gap["edge_accuracy"])
        self.assertAlmostEqual(0.4, gap["middle_accuracy"])
        self.assertAlmostEqual(0.25, gap["position_gap"])
        self.assertEqual(20, gap["edge_scored_n"])
        self.assertEqual(10, gap["middle_scored_n"])

    def test_position_gap_requires_beginning_middle_and_end_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing position cells"):
            position_gap_rows(
                [summary("literal_retrieval", 8192, 0.50, 1.0)]
            )

    def test_position_gap_keeps_runtime_exclusion_as_unavailable(self) -> None:
        rows = [
            summary("literal_retrieval", 8192, 0.05, 1.0),
            summary("literal_retrieval", 8192, 0.50, 0.0, scored_n=0),
            summary("literal_retrieval", 8192, 0.95, 1.0),
        ]

        [gap] = position_gap_rows(rows)

        self.assertEqual("insufficient_data", gap["status"])
        self.assertIsNone(gap["position_gap"])
        self.assertEqual(20, gap["edge_scored_n"])
        self.assertEqual(0, gap["middle_scored_n"])

    def test_position_gap_requires_all_required_cells_to_be_available(self) -> None:
        rows = [
            summary("literal_retrieval", 8192, 0.05, 1.0),
            summary("literal_retrieval", 8192, 0.50, 0.0),
            summary("literal_retrieval", 8192, 0.95, 1.0),
        ]
        rows[0]["analysis_status"] = "unavailable"

        [gap] = position_gap_rows(rows)

        self.assertEqual("insufficient_data", gap["status"])
        self.assertIsNone(gap["position_gap"])
        self.assertEqual(10, gap["edge_scored_n"])
        self.assertEqual(10, gap["middle_scored_n"])

    def test_missing_context_cells_reports_task_length_and_position(self) -> None:
        rows = [summary("literal_retrieval", 8192, 0.05, 1.0)]

        missing = missing_context_cells(
            rows,
            context_lengths=[8192, 32768],
            evidence_positions=[0.05, 0.50],
            task_types=["literal_retrieval"],
        )

        self.assertEqual(
            [
                ("literal_retrieval", 8192, 0.50),
                ("literal_retrieval", 32768, 0.05),
                ("literal_retrieval", 32768, 0.50),
            ],
            missing,
        )

    def test_position_curve_rows_keep_context_and_position_dimensions(self) -> None:
        rows = [
            summary("literal_retrieval", 32768, 0.50, 0.7),
            summary("literal_retrieval", 8192, 0.05, 1.0),
        ]

        curve = position_curve_rows(rows)

        self.assertEqual(
            [
                (8192, 0.05, 1.0),
                (32768, 0.50, 0.7),
            ],
            [
                (
                    row["target_context_tokens"],
                    row["requested_evidence_position"],
                    row["accuracy"],
                )
                for row in curve
            ],
        )

    def test_effective_context_uses_sustained_crossing_and_baseline_gate(self) -> None:
        rows = []
        for context_tokens, accuracy in (
            (8192, 1.0),
            (32768, 0.95),
            (65536, 0.5),
            (131072, 0.4),
        ):
            rows.extend(
                summary("literal_retrieval", context_tokens, position, accuracy)
                for position in (0.05, 0.50, 0.95)
            )
        rows.extend(
            summary("semantic_retrieval", 8192, position, 0.6)
            for position in (0.05, 0.50, 0.95)
        )

        results = effective_context_by_task(rows)
        literal = next(row for row in results if row["task_type"] == "literal_retrieval")
        semantic = next(row for row in results if row["task_type"] == "semantic_retrieval")

        self.assertEqual("estimated", literal["status"])
        self.assertEqual(32768, literal["effective_context_tokens"])
        self.assertEqual(65536, literal["crossing_context_tokens"])
        self.assertEqual(1.0, literal["baseline_accuracy"])
        self.assertEqual("baseline_limited", semantic["status"])
        self.assertIsNone(semantic["effective_context_tokens"])

    def test_effective_context_marks_no_crossing_as_right_censored(self) -> None:
        rows = [
            summary("multi_hop", context_tokens, position, 1.0)
            for context_tokens in (8192, 32768, 65536)
            for position in (0.05, 0.50, 0.95)
        ]

        [result] = effective_context_by_task(rows)

        self.assertEqual("right_censored", result["status"])
        self.assertIsNone(result["crossing_context_tokens"])
        self.assertEqual(65536, result["largest_tested_context_tokens"])

    def test_position_sensitive_effective_context_exposes_middle_collapse(self) -> None:
        rows = []
        for position in (0.05, 0.50, 0.95):
            rows.append(summary("literal_retrieval", 8192, position, 1.0))
            rows.append(
                summary(
                    "literal_retrieval",
                    32768,
                    position,
                    0.5 if position == 0.50 else 1.0,
                )
            )
            rows.append(
                summary(
                    "literal_retrieval",
                    65536,
                    position,
                    0.4 if position == 0.50 else 1.0,
                )
            )

        results = effective_context_by_task_and_position(rows)
        middle = next(row for row in results if row["evidence_position"] == 0.50)
        edge = next(row for row in results if row["evidence_position"] == 0.05)

        self.assertEqual("estimated", middle["status"])
        self.assertEqual(8192, middle["effective_context_tokens"])
        self.assertEqual(32768, middle["crossing_context_tokens"])
        self.assertEqual("right_censored", edge["status"])


if __name__ == "__main__":
    unittest.main()
