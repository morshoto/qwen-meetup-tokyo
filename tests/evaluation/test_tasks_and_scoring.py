import unittest
from pathlib import Path

from llm_lab.evaluation import (
    EvaluationTask,
    ExpectedAnswerScorer,
    ScoreResult,
    Scorer,
    Task,
)
from llm_lab.datasets import TaskCatalog
from llm_lab.generation import GenerationResponse, SamplingConfig
from llm_lab.models import ModelSpec


class TaskAndScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = EvaluationTask(
            task_id="task.literal.000001",
            task_type="literal_retrieval",
            question="What is the access code?",
            context="The access code is ZX-4817.",
            expected={"type": "exact", "value": "ZX-4817"},
        )
        self.model = ModelSpec(
            model_id="fixture/model",
            tokenizer_id="fixture/tokenizer",
        )

    def test_evaluation_task_builds_a_common_generation_request(self) -> None:
        self.assertIsInstance(self.task, Task)

        request = self.task.build_request(
            self.model,
            SamplingConfig(max_new_tokens=5),
        )

        self.assertEqual(self.model, request.model)
        self.assertIn("The access code is ZX-4817.", request.prompt)
        self.assertIn("What is the access code?", request.prompt)
        self.assertEqual(5, request.sampling.max_new_tokens)

    def test_catalog_definition_can_be_adapted_into_a_runnable_evaluation_task(self) -> None:
        catalog = TaskCatalog.from_jsonl(
            Path("data/tasks/core.v001.jsonl")
        )
        definition = catalog.get("task.literal.000001")

        adapted = EvaluationTask.from_definition(
            definition,
            context=definition.evidence[0]["text"],
        )

        self.assertEqual(definition.task_id, adapted.task_id)
        self.assertEqual(definition.expected, adapted.expected)
        self.assertIn("Project Aurora", adapted.question)

    def test_expected_answer_scorer_handles_exact_and_normalized_answers(self) -> None:
        scorer = ExpectedAnswerScorer()
        self.assertIsInstance(scorer, Scorer)

        exact = scorer.score(self.task, GenerationResponse(output_text=" ZX-4817\n"))
        semantic_task = EvaluationTask(
            task_id="task.semantic.000001",
            task_type="semantic_retrieval",
            question="Which team owns the build?",
            context="Reliability Engineering owns the build.",
            expected={
                "type": "normalized_exact",
                "value": "reliability engineering",
                "accepted": ["reliability engineering", "reliability team"],
            },
        )
        normalized = scorer.score(
            semantic_task,
            GenerationResponse(output_text="  Reliability   Team "),
        )

        self.assertEqual(ScoreResult(correct=True, value=1.0, scorer="expected.v1"), exact)
        self.assertTrue(normalized.correct)
        self.assertEqual(1.0, normalized.value)
        self.assertEqual("expected.v1", normalized.scorer)

    def test_scorer_returns_a_machine_checkable_mismatch_and_invalid_output(self) -> None:
        scorer = ExpectedAnswerScorer()

        wrong = scorer.score(self.task, GenerationResponse(output_text="not the code"))
        empty = scorer.score(self.task, GenerationResponse(output_text="  "))

        self.assertEqual(False, wrong.correct)
        self.assertEqual(0.0, wrong.value)
        self.assertEqual("mismatch", wrong.details["reason"])
        self.assertIsNone(empty.correct)
        self.assertIsNone(empty.value)
        self.assertEqual("invalid_output", empty.details["reason"])


if __name__ == "__main__":
    unittest.main()
