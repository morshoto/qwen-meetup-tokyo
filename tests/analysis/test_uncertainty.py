import unittest

from llm_lab.analysis import (
    UncertaintyAnalysisError,
    task_level_wilson,
    wilson_interval,
)


class UncertaintyTests(unittest.TestCase):
    def test_wilson_interval_is_bounded_and_contains_rate(self) -> None:
        low, high = wilson_interval(8, 10)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLessEqual(low, 0.8)
        self.assertGreaterEqual(high, 0.8)

    def test_task_level_wilson_uses_independent_tasks(self) -> None:
        rows = [
            {
                "variant": "q8",
                "context": 8192,
                "task_id": "task-1",
                "attempted_n": 1,
                "exact": 1,
                "answer": 1,
                "format": 1,
            },
            {
                "variant": "q8",
                "context": 8192,
                "task_id": "task-2",
                "attempted_n": 1,
                "exact": 0,
                "answer": 1,
                "format": 0,
            },
        ]
        [summary] = task_level_wilson(
            rows,
            group_keys=("variant", "context"),
            metric_fields={
                "exact": "exact",
                "answer_bearing": "answer",
                "format_valid": "format",
            },
        )
        self.assertEqual(2, summary["task_n"])
        self.assertEqual(1, summary["exact_success_n"])
        self.assertEqual(0.5, summary["exact_rate"])
        self.assertEqual(2, summary["answer_bearing_success_n"])
        self.assertEqual(0.5, summary["format_valid_rate"])

    def test_duplicate_task_is_not_counted_as_independent_sample(self) -> None:
        rows = [
            {"group": "g", "task_id": "task-1", "attempted_n": 1, "correct": 1},
            {"group": "g", "task_id": "task-1", "attempted_n": 1, "correct": 1},
        ]
        with self.assertRaisesRegex(UncertaintyAnalysisError, "not independent"):
            task_level_wilson(
                rows,
                group_keys=("group",),
                metric_fields={"correct": "correct"},
            )

    def test_runtime_failure_can_remain_a_failed_attempt(self) -> None:
        [summary] = task_level_wilson(
            [
                {"group": "g", "task_id": "ok", "attempted_n": 1, "correct": 1},
                {"group": "g", "task_id": "runtime-error", "attempted_n": 1, "correct": None},
            ],
            group_keys=("group",),
            metric_fields={"correct": "correct"},
        )
        self.assertEqual(2, summary["attempted_n"])
        self.assertEqual(1, summary["correct_success_n"])
        self.assertEqual(0.5, summary["correct_rate"])

    def test_pandas_numeric_scalars_are_accepted(self) -> None:
        [summary] = task_level_wilson(
            [
                {"group": "g", "task_id": "task-1", "attempted_n": 1.0, "correct": 1.0},
                {"group": "g", "task_id": "task-2", "attempted_n": 1.0, "correct": 0.0},
            ],
            group_keys=("group",),
            metric_fields={"correct": "correct"},
        )
        self.assertEqual(2, summary["task_n"])
        self.assertEqual(1, summary["correct_success_n"])


if __name__ == "__main__":
    unittest.main()
