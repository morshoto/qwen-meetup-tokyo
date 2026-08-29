import json
import unittest
from pathlib import Path

from llm_lab.evaluation import CalibratedAnswerScorer, EvaluationTask
from llm_lab.generation import GenerationResponse


CALIBRATION_PATH = Path("data/fixtures/scoring_calibration.v001.json")


class ScoringCalibrationTests(unittest.TestCase):
    def test_calibration_examples_cover_literal_semantic_and_multi_hop_tasks(self) -> None:
        examples = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        scorer = CalibratedAnswerScorer()

        self.assertEqual(
            {"literal", "semantic", "multi_hop"},
            {example["id"] for example in examples},
        )
        for example in examples:
            task = EvaluationTask(**example["task"])
            for case in example["cases"]:
                with self.subTest(example=example["id"], case=case["name"]):
                    result = scorer.score(task, GenerationResponse(output_text=case["output"]))

                    self.assertEqual(case["exact_correct"], result.exact_correct)
                    self.assertEqual(
                        case["answer_bearing_correct"],
                        result.answer_bearing_correct,
                    )
                    self.assertEqual(case["format_valid"], result.format_valid)
                    self.assertEqual(case["exact_correct"], result.correct)
                    self.assertEqual(
                        1.0 if case["exact_correct"] else 0.0,
                        result.value,
                    )

    def test_scorer_records_a_stable_policy_version(self) -> None:
        task = EvaluationTask(
            task_id="task.literal.calibration",
            task_type="literal_retrieval",
            question="What is the access code?",
            context="The access code is ZX-4817.",
            expected={"type": "exact", "value": "ZX-4817", "format": "identifier"},
        )

        result = CalibratedAnswerScorer().score(
            task,
            GenerationResponse(output_text="ZX-4817"),
        )

        self.assertEqual("calibrated.v1", result.scorer)

    def test_empty_output_is_invalid_without_claiming_model_incorrectness(self) -> None:
        task = EvaluationTask(
            task_id="task.literal.calibration",
            task_type="literal_retrieval",
            question="What is the access code?",
            context="The access code is ZX-4817.",
            expected={"type": "exact", "value": "ZX-4817", "format": "identifier"},
        )

        result = CalibratedAnswerScorer().score(task, GenerationResponse(output_text="  "))

        self.assertIsNone(result.correct)
        self.assertIsNone(result.exact_correct)
        self.assertIsNone(result.answer_bearing_correct)
        self.assertFalse(result.format_valid)
        self.assertEqual("invalid_output", result.details["reason"])

    def test_scoring_contract_documents_policy_dimensions_and_semantic_range(self) -> None:
        contract = Path("docs/data-and-result-contracts.md").read_text(encoding="utf-8")

        for term in (
            "calibrated.v1",
            "exact_correct",
            "answer_bearing_correct",
            "format_valid",
            "expected.accepted",
            "scorer_error",
            "runtime_error",
        ):
            with self.subTest(term=term):
                self.assertIn(term, contract)


if __name__ == "__main__":
    unittest.main()
