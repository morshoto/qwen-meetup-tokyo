"""Task and scorer contracts used by the benchmark runner."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from llm_lab.datasets import TaskDefinition
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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_definition(
        cls,
        definition: TaskDefinition,
        *,
        context: str,
        prompt_id: str = "prompt.qa.v001",
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvaluationTask":
        task_metadata = dict(definition.metadata)
        task_metadata.update(metadata or {})
        return cls(
            task_id=definition.task_id,
            task_type=definition.task_type,
            question=definition.question,
            context=context,
            expected=definition.expected,
            prompt_id=prompt_id,
            metadata=task_metadata,
        )

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
        request_metadata = dict(self.metadata)
        request_metadata.update({"task_id": self.task_id, "prompt_id": self.prompt_id})
        return GenerationRequest(
            prompt=prompt,
            model=model,
            sampling=sampling,
            metadata=request_metadata,
        )


@dataclass(frozen=True)
class ScoreResult:
    correct: bool | None
    value: float | None
    scorer: str
    details: Mapping[str, Any] = field(default_factory=dict)
    exact_correct: bool | None = None
    answer_bearing_correct: bool | None = None
    format_valid: bool | None = None


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


class CalibratedAnswerScorer:
    """Score exactness, answer presence, and output shape independently.

    ``correct`` and ``value`` intentionally retain exact-answer semantics for
    compatibility with existing aggregations.  The calibrated fields expose
    answer-bearing output and format validity without changing that legacy
    interpretation.
    """

    name = "calibrated.v1"

    def score(self, task: Task, response: GenerationResponse) -> ScoreResult:
        output = response.output_text.strip()
        if not output:
            return ScoreResult(
                correct=None,
                value=None,
                scorer=self.name,
                details={"reason": "invalid_output"},
                format_valid=False,
            )

        expected = getattr(task, "expected")
        expected_type = expected.get("type")
        canonical, accepted = _expected_answers(expected, expected_type)
        exact_correct = (
            output == canonical
            if expected_type == "exact"
            else _normalize(output) == _normalize(canonical)
        )
        answer_bearing_correct = any(
            _contains_answer(_normalize(output), _normalize(answer))
            for answer in accepted
        )
        expected_format = _expected_format(expected, canonical)
        format_valid = _is_valid_format(output, expected_format)
        return ScoreResult(
            correct=exact_correct,
            value=1.0 if exact_correct else 0.0,
            scorer=self.name,
            details={
                "expected_type": expected_type,
                "expected_format": expected_format,
                "normalized_output": _normalize(output),
                "accepted": list(accepted),
            },
            exact_correct=exact_correct,
            answer_bearing_correct=answer_bearing_correct,
            format_valid=format_valid,
        )


def _expected_answers(
    expected: Mapping[str, Any],
    expected_type: Any,
) -> tuple[str, tuple[str, ...]]:
    if expected_type == "exact":
        canonical = str(expected.get("value", "")).strip()
        return canonical, (canonical,)
    if expected_type == "normalized_exact":
        canonical = str(expected.get("value", "")).strip()
        accepted = tuple(
            str(value).strip()
            for value in expected.get("accepted", [canonical])
        )
        return canonical, accepted
    raise ValueError(f"unsupported expected answer type: {expected_type!r}")


def _expected_format(expected: Mapping[str, Any], canonical: str) -> str:
    declared = expected.get("format")
    if declared is not None:
        return str(declared)
    return "identifier" if _IDENTIFIER_PATTERN.fullmatch(canonical) else "phrase"


_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
_PHRASE_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[ _-][A-Za-z0-9]+)*")


def _is_valid_format(output: str, expected_format: str) -> bool:
    if expected_format == "identifier":
        return _IDENTIFIER_PATTERN.fullmatch(output) is not None
    if expected_format == "phrase":
        return _PHRASE_PATTERN.fullmatch(output) is not None
    raise ValueError(f"unsupported answer format: {expected_format!r}")


def _contains_answer(output: str, answer: str) -> bool:
    if not answer:
        return False
    return re.search(rf"(?<!\w){re.escape(answer)}(?!\w)", output) is not None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
