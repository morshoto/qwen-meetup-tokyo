import hashlib
import tempfile
import unittest
from pathlib import Path

from llm_lab.analysis.rescoring import (
    comparison_rows,
    render_report,
    rescore_trial,
    rescore_trials,
    sha256_file,
)
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
    repeat_index: int = 1,
    task_type: str = "literal_retrieval",
) -> TrialResult:
    return TrialResult(
        trial_id=f"exp_002:{task_id}:q8_0:ctx8192:run{repeat_index:02d}",
        experiment_id="exp_002",
        task_id=task_id,
        status=TrialStatus.COMPLETED,
        input={
            "task_type": task_type,
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


def runtime_trial(task_id: str) -> TrialResult:
    return TrialResult(
        trial_id=f"exp_002:{task_id}:q8_0:ctx8192:run03",
        experiment_id="exp_002",
        task_id=task_id,
        status=TrialStatus.RUNTIME_ERROR,
        input={
            "task_type": "literal_retrieval",
            "condition_id": "q8_0:ctx8192",
            "variant_condition_id": "q8_0",
            "target_context_tokens": 8192,
        },
        score={},
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

    def test_runtime_failure_is_not_counted_as_a_calibrated_output(self) -> None:
        evaluation_task = task(
            "task.literal.000001",
            "literal_retrieval",
            {"type": "exact", "value": "ZX-4817"},
        )

        [result] = rescore_trials(
            [runtime_trial(evaluation_task.task_id)],
            {evaluation_task.task_id: evaluation_task},
        )

        self.assertEqual("runtime_failure", result.category)
        self.assertIsNone(result.score)
        self.assertIsNone(result.legacy_correct)

    def test_comparison_rows_group_by_task_variant_and_context(self) -> None:
        literal_task = task(
            "task.literal.000001",
            "literal_retrieval",
            {"type": "exact", "value": "ZX-4817"},
        )
        semantic_task = task(
            "task.semantic.000001",
            "semantic_retrieval",
            {
                "type": "normalized_exact",
                "value": "reliability engineering",
                "accepted": ["reliability engineering", "reliability team"],
            },
        )
        trials = [
            trial(literal_task.task_id, "ZX-4817", old_correct=True),
            trial(
                literal_task.task_id,
                "ZX-4817.659",
                old_correct=False,
                repeat_index=2,
            ),
            runtime_trial(literal_task.task_id),
            trial(
                semantic_task.task_id,
                "Reliability Engineering group",
                old_correct=False,
                task_type="semantic_retrieval",
            ),
        ]
        trials[-1] = TrialResult(
            trial_id="exp_002:task.semantic.000001:q4_k_m:ctx32768:run01",
            experiment_id=trials[-1].experiment_id,
            task_id=trials[-1].task_id,
            status=trials[-1].status,
            input={
                **trials[-1].input,
                "condition_id": "q4_k_m:ctx32768",
                "variant_condition_id": "q4_k_m",
                "target_context_tokens": 32768,
            },
            generation=trials[-1].generation,
            score=trials[-1].score,
        )

        rows = comparison_rows(
            rescore_trials(
                trials,
                {
                    literal_task.task_id: literal_task,
                    semantic_task.task_id: semantic_task,
                },
            )
        )

        self.assertEqual(2, len(rows))
        literal = rows[0]
        self.assertEqual("task.literal.000001", literal["task_id"])
        self.assertEqual("q8_0", literal["variant_condition_id"])
        self.assertEqual(8192, literal["target_context_tokens"])
        self.assertEqual(3, literal["attempted_n"])
        self.assertEqual(2, literal["old_scored_n"])
        self.assertEqual(1, literal["old_correct_n"])
        self.assertEqual(2, literal["new_exact_scored_n"])
        self.assertEqual(1, literal["new_exact_correct_n"])
        self.assertEqual(2, literal["new_answer_bearing_correct_n"])
        self.assertEqual(1, literal["new_format_valid_n"])
        self.assertEqual(1, literal["exact_match_n"])
        self.assertEqual(1, literal["format_failure_n"])
        self.assertEqual(1, literal["runtime_failure_n"])

        semantic = rows[1]
        self.assertEqual("task.semantic.000001", semantic["task_id"])
        self.assertEqual("q4_k_m", semantic["variant_condition_id"])
        self.assertEqual(32768, semantic["target_context_tokens"])
        self.assertEqual(1, semantic["mismatch_n"])

    def test_report_records_raw_sha256_and_diagnostic_caveat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "trials.jsonl"
            raw_path.write_text('{"trial_id":"one"}\n', encoding="utf-8")
            expected_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            catalog_path = Path("data/tasks/core.v001.jsonl")
            expected_catalog_hash = sha256_file(catalog_path)

            report = render_report(
                raw_path=raw_path,
                task_catalog_path=catalog_path,
                raw_trial_n=1,
                rows=[
                    {
                        "variant_condition_id": "q8_0",
                        "variant_label": "Q8_0",
                        "old_scored_n": 1,
                        "old_correct_n": 0,
                        "new_exact_scored_n": 1,
                        "new_exact_correct_n": 0,
                        "new_answer_bearing_correct_n": 1,
                        "new_answer_bearing_scored_n": 1,
                        "new_format_valid_n": 0,
                        "new_format_scored_n": 1,
                        "mismatch_n": 0,
                        "format_failure_n": 1,
                        "runtime_failure_n": 0,
                    }
                ],
            )

            self.assertIn(expected_hash, report)
            self.assertIn(expected_catalog_hash, report)
            self.assertIn("calibrated.v1", report)
            self.assertIn("Diagnostic re-scoring only", report)
            self.assertIn("must not be used for a final quantization claim", report)
            self.assertIn("rescored-summary.csv", report)


if __name__ == "__main__":
    unittest.main()
