"""Diagnostic re-scoring for historical trial records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from llm_lab.evaluation import (
    CalibratedAnswerScorer,
    EvaluationTask,
    ScoreResult,
    Scorer,
    TrialResult,
    TrialStatus,
)
from llm_lab.generation import GenerationResponse


EXACT_MATCH = "exact_match"
MISMATCH = "mismatch"
FORMAT_FAILURE = "format_failure"
RUNTIME_FAILURE = "runtime_failure"


@dataclass(frozen=True)
class RescoredTrial:
    """A non-mutating comparison of one historical trial and new score."""

    trial: TrialResult
    score: ScoreResult | None
    category: str
    legacy_correct: bool | None


def rescore_trial(
    trial: TrialResult,
    task: EvaluationTask,
    *,
    scorer: Scorer | None = None,
) -> RescoredTrial:
    """Apply a calibrated scorer without changing the historical trial."""

    legacy_correct = trial.score.get("correct")
    if not isinstance(legacy_correct, bool):
        legacy_correct = None

    if trial.status != TrialStatus.COMPLETED:
        return RescoredTrial(
            trial=trial,
            score=None,
            category=RUNTIME_FAILURE,
            legacy_correct=legacy_correct,
        )

    output = trial.generation.get("output_text")
    response = GenerationResponse(output_text=output if isinstance(output, str) else "")
    calibrated_score = (scorer or CalibratedAnswerScorer()).score(task, response)
    return RescoredTrial(
        trial=trial,
        score=calibrated_score,
        category=_category_for(calibrated_score),
        legacy_correct=legacy_correct,
    )


def rescore_trials(
    trials: Iterable[TrialResult | Mapping[str, Any]],
    tasks: Mapping[str, EvaluationTask],
    *,
    scorer: Scorer | None = None,
) -> list[RescoredTrial]:
    """Re-score a batch using an explicitly supplied task catalog."""

    results: list[RescoredTrial] = []
    for value in trials:
        trial = value if isinstance(value, TrialResult) else TrialResult.from_record(value)
        task = tasks.get(trial.task_id)
        if task is None:
            raise ValueError(f"missing task definition for {trial.task_id!r}")
        results.append(rescore_trial(trial, task, scorer=scorer))
    return results


def comparison_rows(
    trials: Iterable[RescoredTrial],
) -> list[dict[str, Any]]:
    """Aggregate legacy/calibrated outcomes by task, variant, and context."""

    groups: dict[tuple[Any, ...], list[RescoredTrial]] = {}
    for result in trials:
        source = result.trial
        metadata = source.input
        key = (
            source.experiment_id,
            source.task_id,
            metadata.get("task_type", "unknown"),
            metadata.get("variant_condition_id"),
            metadata.get("target_context_tokens"),
            metadata.get("condition_id", "default"),
        )
        groups.setdefault(key, []).append(result)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=_group_sort_key):
        (
            experiment_id,
            task_id,
            task_type,
            variant_condition_id,
            target_context_tokens,
            condition_id,
        ) = key
        legacy_values = [
            result.legacy_correct
            for result in group
            if isinstance(result.legacy_correct, bool)
        ]
        exact_values = [
            result.score.exact_correct
            for result in group
            if result.score is not None and isinstance(result.score.exact_correct, bool)
        ]
        answer_bearing_values = [
            result.score.answer_bearing_correct
            for result in group
            if result.score is not None
            and isinstance(result.score.answer_bearing_correct, bool)
        ]
        format_values = [
            result.score.format_valid
            for result in group
            if result.score is not None and isinstance(result.score.format_valid, bool)
        ]
        rows.append(
            {
                "experiment_id": experiment_id,
                "task_id": task_id,
                "task_type": task_type,
                "condition_id": condition_id,
                "variant_condition_id": variant_condition_id,
                "variant_label": _common_input_value(group, "variant_label"),
                "quantization_type": _common_input_value(group, "quantization_type"),
                "target_context_tokens": target_context_tokens,
                "attempted_n": len(group),
                "completed_n": sum(
                    result.trial.status == TrialStatus.COMPLETED for result in group
                ),
                "old_scorer": _common_old_score_value(group, "scorer"),
                "old_scored_n": len(legacy_values),
                "old_correct_n": sum(legacy_values),
                "old_accuracy": _accuracy(legacy_values),
                "new_scorer": _common_new_score_value(group),
                "new_exact_scored_n": len(exact_values),
                "new_exact_correct_n": sum(exact_values),
                "new_exact_accuracy": _accuracy(exact_values),
                "new_answer_bearing_scored_n": len(answer_bearing_values),
                "new_answer_bearing_correct_n": sum(answer_bearing_values),
                "new_answer_bearing_accuracy": _accuracy(answer_bearing_values),
                "new_format_scored_n": len(format_values),
                "new_format_valid_n": sum(format_values),
                "new_format_validity": _accuracy(format_values),
                "score_changed_n": sum(
                    result.legacy_correct != result.score.exact_correct
                    for result in group
                    if isinstance(result.legacy_correct, bool)
                    and result.score is not None
                    and isinstance(result.score.exact_correct, bool)
                ),
                "exact_match_n": _category_count(group, EXACT_MATCH),
                "mismatch_n": _category_count(group, MISMATCH),
                "format_failure_n": _category_count(group, FORMAT_FAILURE),
                "runtime_failure_n": _category_count(group, RUNTIME_FAILURE),
            }
        )
    return rows


def _category_for(score: ScoreResult) -> str:
    if score.format_valid is False:
        return FORMAT_FAILURE
    if score.exact_correct is False:
        return MISMATCH
    if score.exact_correct is True:
        return EXACT_MATCH
    raise ValueError("calibrated score must declare exactness or format validity")


def _accuracy(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _category_count(group: Iterable[RescoredTrial], category: str) -> int:
    return sum(result.category == category for result in group)


def _common_input_value(
    group: Iterable[RescoredTrial],
    key: str,
) -> Any:
    values = [result.trial.input.get(key) for result in group]
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _common_old_score_value(
    group: Iterable[RescoredTrial],
    key: str,
) -> Any:
    values = [result.trial.score.get(key) for result in group]
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _common_new_score_value(group: Iterable[RescoredTrial]) -> str | None:
    values = [
        result.score.scorer
        for result in group
        if result.score is not None
    ]
    if not values or any(value != values[0] for value in values[1:]):
        return None
    return values[0]


def _group_sort_key(item: tuple[tuple[Any, ...], list[RescoredTrial]]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in item[0])
