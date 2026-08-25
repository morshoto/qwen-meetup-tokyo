import unittest

from llm_lab.analysis.effective_context import (
    effective_context_by_task,
    missing_context_cells,
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


if __name__ == "__main__":
    unittest.main()
