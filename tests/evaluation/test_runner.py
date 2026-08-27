import tempfile
import unittest
from pathlib import Path

from llm_lab.evaluation import (
    EvaluationTask,
    ExpectedAnswerScorer,
    ScoreResult,
    TrialStatus,
    load_trial_results,
)
from llm_lab.evaluation.runner import EvaluationRunner
from llm_lab.generation import (
    GenerationRequest,
    GenerationResponse,
    GenerationTiming,
    RuntimeMetadata,
    TokenUsage,
)
from llm_lab.models import ModelSpec


class FixtureRuntime:
    name = "fixture"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if "runtime failure" in request.prompt:
            raise RuntimeError("fixture backend unavailable")
        output = " " if "invalid output" in request.prompt else "ZX-4817"
        return GenerationResponse(
            output_text=output,
            usage=TokenUsage(prompt_tokens=4, completion_tokens=1),
            timing=GenerationTiming(
                ttft_seconds=0.1,
                prefill_seconds=0.2,
                post_first_chunk_seconds=0.05,
                total_seconds=0.25,
            ),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version="1.0",
                model_id=request.model.model_id,
                tokenizer_id=request.model.tokenizer_id,
                config={"device": "cpu"},
            ),
        )


class FailingScorer:
    name = "failing"

    def score(self, task: EvaluationTask, response: GenerationResponse) -> ScoreResult:
        raise ValueError("fixture scorer failed")


def task(
    task_id: str,
    context: str,
    *,
    metadata: dict[str, object] | None = None,
) -> EvaluationTask:
    return EvaluationTask(
        task_id=task_id,
        task_type="literal_retrieval",
        question="What is the code?",
        context=context,
        expected={"type": "exact", "value": "ZX-4817"},
        metadata=metadata or {},
    )


class EvaluationRunnerTests(unittest.TestCase):
    def test_runner_repeats_tasks_writes_results_and_preserves_metadata(self) -> None:
        model = ModelSpec(model_id="fixture/model", tokenizer_id="fixture/tokenizer")
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "results" / "raw" / "trials.jsonl"
            runner = EvaluationRunner(
                runtime=FixtureRuntime(),
                model=model,
                scorer=ExpectedAnswerScorer(),
                experiment_id="exp_fixture",
                output_path=output_path,
            )

            results = runner.run(
                [task("task.literal.000001", "The code is ZX-4817.")],
                repeats=2,
                condition_id="fixture",
            )
            persisted = load_trial_results(output_path)

        self.assertEqual(2, len(results))
        self.assertEqual(2, len(persisted))
        self.assertEqual(
            ["exp_fixture:task.literal.000001:fixture:run01", "exp_fixture:task.literal.000001:fixture:run02"],
            [result.trial_id for result in results],
        )
        self.assertTrue(all(result.status == TrialStatus.COMPLETED for result in results))
        self.assertEqual(True, results[0].score["correct"])
        self.assertEqual("fixture/tokenizer", results[0].runtime["tokenizer_id"])
        self.assertEqual(20.0, results[0].timing["prefill_tokens_per_second"])
        self.assertEqual(0.1, results[0].timing["stream_ttft_s"])
        self.assertEqual(40.0, results[0].timing["prompt_throughput_proxy_tok_s"])
        self.assertEqual(20.0, results[0].timing["post_first_chunk_output_tok_s"])

    def test_runner_records_runtime_and_invalid_output_failures(self) -> None:
        runner = EvaluationRunner(
            runtime=FixtureRuntime(),
            model=ModelSpec(model_id="fixture/model"),
            scorer=ExpectedAnswerScorer(),
            experiment_id="exp_fixture",
        )

        results = runner.run(
            [
                task("task.runtime", "runtime failure"),
                task("task.invalid", "invalid output"),
            ]
        )

        self.assertEqual([TrialStatus.RUNTIME_ERROR, TrialStatus.INVALID_OUTPUT], [item.status for item in results])
        self.assertEqual("fixture backend unavailable", results[0].error["message"])
        self.assertEqual("invalid_output", results[1].score["details"]["reason"])
        self.assertIsNone(results[1].score["correct"])

    def test_runner_preserves_task_metadata_in_trial_input(self) -> None:
        runner = EvaluationRunner(
            runtime=FixtureRuntime(),
            model=ModelSpec(model_id="fixture/model"),
            scorer=ExpectedAnswerScorer(),
            experiment_id="exp_fixture",
        )

        [result] = runner.run(
            [
                task(
                    "task.context",
                    "The code is ZX-4817.",
                    metadata={
                        "target_context_tokens": 8192,
                        "requested_evidence_position": 0.05,
                        "actual_evidence_position": 0.047,
                        "evidence_spans": [
                            {"id": "code", "token_start": 410, "token_end": 413}
                        ],
                    },
                )
            ],
            condition_id="baseline:ctx08192:p005",
        )

        self.assertEqual(8192, result.input["target_context_tokens"])
        self.assertEqual(0.05, result.input["requested_evidence_position"])
        self.assertEqual(0.047, result.input["actual_evidence_position"])
        self.assertEqual(
            [{"id": "code", "token_start": 410, "token_end": 413}],
            result.input["evidence_spans"],
        )
        self.assertEqual(4, result.input["prompt_tokens"])

    def test_runner_records_scorer_failures_after_generation(self) -> None:
        runner = EvaluationRunner(
            runtime=FixtureRuntime(),
            model=ModelSpec(model_id="fixture/model"),
            scorer=FailingScorer(),
            experiment_id="exp_fixture",
        )

        [result] = runner.run([task("task.scorer", "The code is ZX-4817.")])

        self.assertEqual(TrialStatus.SCORER_ERROR, result.status)
        self.assertEqual("fixture scorer failed", result.error["message"])
        self.assertEqual("ZX-4817", result.generation["output_text"])


if __name__ == "__main__":
    unittest.main()
