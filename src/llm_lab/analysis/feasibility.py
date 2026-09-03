"""Conservative classification for exp_001 context feasibility probes."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from llm_lab.evaluation import TrialResult, TrialStatus


FEASIBILITY_CLASSIFICATIONS = frozenset(
    {"accepted_and_useful", "accepted_but_not_useful", "operational_failure"}
)
_OPERATIONAL_FAILURES = frozenset(
    {
        TrialStatus.RUNTIME_ERROR,
        TrialStatus.SCORER_ERROR,
        TrialStatus.INVALID_INPUT,
        TrialStatus.OUT_OF_MEMORY,
        TrialStatus.TIMEOUT,
        TrialStatus.CANCELLED,
    }
)


class FeasibilityAnalysisError(ValueError):
    """Raised when a feasibility probe cannot be classified safely."""


def classify_feasibility(
    trials: Iterable[TrialResult | Mapping[str, Any]],
    *,
    expected_task_ids: Iterable[str],
) -> dict[str, Any]:
    """Classify one context length using an all-task, fail-closed rule.

    A length is useful only when every selected task completed and was
    answer-bearing correct. Runtime, scorer, input, timeout, OOM, cancellation,
    or missing-task failures take precedence over capability scores. An
    invalid/empty model output is an accepted-but-not-useful observation because
    the process completed but did not yield useful capability.
    """

    expected = tuple(str(task_id) for task_id in expected_task_ids)
    if not expected or len(expected) > 3 or len(set(expected)) != len(expected):
        raise FeasibilityAnalysisError(
            "expected_task_ids must contain between one and three unique task IDs"
        )
    normalized: list[TrialResult] = []
    for value in trials:
        normalized.append(
            value if isinstance(value, TrialResult) else TrialResult.from_record(value)
        )
    lengths = {
        _context_length(trial)
        for trial in normalized
    }
    if len(lengths) > 1:
        raise FeasibilityAnalysisError(
            f"feasibility classification requires one context length, found {sorted(lengths)}"
        )
    context_length = next(iter(lengths), None)
    by_task: dict[str, TrialResult] = {}
    duplicate_task_ids: list[str] = []
    for trial in normalized:
        if trial.task_id in by_task:
            duplicate_task_ids.append(trial.task_id)
        by_task[trial.task_id] = trial
    missing_task_ids = [task_id for task_id in expected if task_id not in by_task]
    unexpected_task_ids = [task_id for task_id in by_task if task_id not in expected]
    statuses = Counter(trial.status.value for trial in normalized)
    operational_trials = [
        trial for trial in normalized if trial.status in _OPERATIONAL_FAILURES
    ]
    invalid_output_n = sum(
        trial.status == TrialStatus.INVALID_OUTPUT for trial in normalized
    )
    answer_bearing_n = sum(
        trial.score.get("answer_bearing_correct") is True for trial in normalized
    )
    scored_n = sum(
        isinstance(trial.score.get("answer_bearing_correct"), bool)
        for trial in normalized
    )
    reasons: list[str] = []
    if missing_task_ids:
        reasons.append(f"missing task IDs: {missing_task_ids}")
    if unexpected_task_ids:
        reasons.append(f"unexpected task IDs: {unexpected_task_ids}")
    if duplicate_task_ids:
        reasons.append(f"duplicate task IDs: {sorted(set(duplicate_task_ids))}")
    if operational_trials:
        reasons.append(
            "operational statuses: "
            + ", ".join(sorted({trial.status.value for trial in operational_trials}))
        )

    if reasons:
        classification = "operational_failure"
    elif (
        len(normalized) == len(expected)
        and all(trial.status == TrialStatus.COMPLETED for trial in normalized)
        and answer_bearing_n == len(expected)
    ):
        classification = "accepted_and_useful"
    else:
        classification = "accepted_but_not_useful"

    return {
        "target_context_tokens": context_length,
        "classification": classification,
        "expected_task_n": len(expected),
        "attempted_n": len(normalized),
        "completed_n": sum(
            trial.status == TrialStatus.COMPLETED for trial in normalized
        ),
        "scored_n": scored_n,
        "answer_bearing_n": answer_bearing_n,
        "invalid_output_n": invalid_output_n,
        "timeout_n": statuses.get(TrialStatus.TIMEOUT.value, 0),
        "runtime_failure_n": sum(
            statuses.get(status.value, 0) for status in _OPERATIONAL_FAILURES
        ),
        "status_counts": dict(sorted(statuses.items())),
        "missing_task_ids": missing_task_ids,
        "unexpected_task_ids": unexpected_task_ids,
        "reason": "; ".join(reasons) or None,
    }


def classify_feasibility_by_length(
    trials: Iterable[TrialResult | Mapping[str, Any]],
    *,
    expected_task_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Return deterministic feasibility rows sorted by target context length."""

    normalized = [
        value if isinstance(value, TrialResult) else TrialResult.from_record(value)
        for value in trials
    ]
    grouped: dict[int, list[TrialResult]] = {}
    for trial in normalized:
        grouped.setdefault(_context_length(trial), []).append(trial)
    return [
        classify_feasibility(group, expected_task_ids=expected_task_ids)
        for _, group in sorted(grouped.items())
    ]


def _context_length(trial: TrialResult) -> int:
    value = trial.input.get("target_context_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FeasibilityAnalysisError(
            f"trial {trial.trial_id} is missing target_context_tokens"
        )
    return value
