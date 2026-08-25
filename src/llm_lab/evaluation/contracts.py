"""Task and scorer contracts used by the benchmark runner."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from llm_lab.generation import GenerationRequest, GenerationResponse, SamplingConfig
from llm_lab.models import ModelSpec


@runtime_checkable
class Task(Protocol):
    task_id: str
    task_type: str

    def build_request(
        self,
        model: ModelSpec,
        sampling: SamplingConfig,
    ) -> GenerationRequest:
        """Build one backend-neutral request for this task."""


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    task_type: str
    question: str
    context: str
    expected: Mapping[str, Any]
    prompt_id: str = "prompt.qa.v001"

    def build_request(
        self,
        model: ModelSpec,
        sampling: SamplingConfig,
    ) -> GenerationRequest:
        prompt = (
            "Use only the evidence in the context. Return the shortest answer "
            "that satisfies the question. Do not explain your reasoning.\n\n"
            f"Context:\n{self.context}\n\n"
            f"Question: {self.question}\n"
            "Answer:"
        )
        return GenerationRequest(
            prompt=prompt,
            model=model,
            sampling=sampling,
            metadata={"task_id": self.task_id, "prompt_id": self.prompt_id},
        )


@dataclass(frozen=True)
class ScoreResult:
    correct: bool | None
    value: float | None
    scorer: str
    details: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, task: Task, response: GenerationResponse) -> ScoreResult:
        """Score a generated response without invoking the runtime."""


class ExpectedAnswerScorer:
    """Score exact and normalized-exact task answer declarations."""

    name = "expected.v1"

    def score(self, task: Task, response: GenerationResponse) -> ScoreResult:
        output = response.output_text.strip()
        if not output:
            return ScoreResult(
                correct=None,
                value=None,
                scorer=self.name,
                details={"reason": "invalid_output"},
            )

        expected = getattr(task, "expected")
        expected_type = expected.get("type")
        if expected_type == "exact":
            candidate = output
            accepted = [str(expected.get("value", "")).strip()]
        elif expected_type == "normalized_exact":
            candidate = _normalize(output)
            accepted = [
                _normalize(str(value))
                for value in expected.get("accepted", [expected.get("value", "")])
            ]
        else:
            raise ValueError(f"unsupported expected answer type: {expected_type!r}")

        correct = candidate in accepted
        details: dict[str, Any] = {}
        if not correct:
            details = {
                "reason": "mismatch",
                "normalized_output": candidate,
                "accepted": accepted,
            }
        return ScoreResult(
            correct=correct,
            value=1.0 if correct else 0.0,
            scorer=self.name,
            details=details,
        )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
