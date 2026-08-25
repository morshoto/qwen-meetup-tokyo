"""Analysis helpers for exp_001 context-length and position measurements."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


class ContextAnalysisError(ValueError):
    """Raised when context dimensions or required baseline data are missing."""


def missing_context_cells(
    summaries: Iterable[Mapping[str, Any]],
    *,
    context_lengths: Iterable[int],
    evidence_positions: Iterable[float],
    task_types: Iterable[str],
) -> list[tuple[str, int, float]]:
    """Return planned task × context × position cells absent from summaries."""

    present = {
        _cell_key(row)
        for row in summaries
        if _has_context_dimensions(row)
    }
    return [
        (task_type, context_tokens, position)
        for task_type in task_types
        for context_tokens in context_lengths
        for position in evidence_positions
        if (task_type, context_tokens, position) not in present
    ]


def position_curve_rows(
    summaries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return summary rows sorted for position curves, failing on missing dimensions."""

    rows = [dict(row) for row in summaries]
    for row in rows:
        _cell_key(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["task_type"]),
            int(row["target_context_tokens"]),
            float(row["requested_evidence_position"]),
        ),
    )


def effective_context_by_task(
    summaries: Iterable[Mapping[str, Any]],
    *,
    baseline_context_tokens: int = 8192,
    alpha: float = 0.90,
    minimum_baseline_accuracy: float = 0.80,
) -> list[dict[str, Any]]:
    """Calculate task-specific effective context using the project rule.

    Accuracy is weighted by scored trial count across evidence positions. A
    baseline below the declared gate is ``baseline_limited``. Otherwise the
    first below-threshold context with the next tested context also below the
    threshold is the sustained crossing. A final, unconfirmed drop is marked
    ``provisional``; no observed drop is ``right_censored``.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 < minimum_baseline_accuracy <= 1.0:
        raise ValueError("minimum_baseline_accuracy must be between 0 and 1")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        _cell_key(row)
        grouped[str(row["task_type"])].append(row)

    results: list[dict[str, Any]] = []
    for task_type, rows in sorted(grouped.items()):
        points = _weighted_context_points(rows)
        baseline = next(
            (point for point in points if point["context_tokens"] == baseline_context_tokens),
            None,
        )
        if baseline is None or baseline["accuracy"] is None:
            raise ContextAnalysisError(
                f"{task_type} is missing a scored {baseline_context_tokens}-token baseline"
            )
        baseline_accuracy = float(baseline["accuracy"])
        threshold = alpha * baseline_accuracy
        result: dict[str, Any] = {
            "task_type": task_type,
            "baseline_context_tokens": baseline_context_tokens,
            "baseline_accuracy": baseline_accuracy,
            "baseline_valid": baseline_accuracy >= minimum_baseline_accuracy,
            "minimum_baseline_accuracy": minimum_baseline_accuracy,
            "alpha": alpha,
            "threshold_accuracy": threshold,
            "largest_tested_context_tokens": points[-1]["context_tokens"],
            "crossing_context_tokens": None,
            "effective_context_tokens": None,
            "status": "baseline_limited",
            "points": points,
        }
        if not result["baseline_valid"]:
            results.append(result)
            continue

        crossing_index = next(
            (
                index
                for index, point in enumerate(points)
                if point["accuracy"] is not None
                and point["accuracy"] < threshold
                and index + 1 < len(points)
                and points[index + 1]["accuracy"] is not None
                and points[index + 1]["accuracy"] < threshold
            ),
            None,
        )
        if crossing_index is not None:
            result.update(
                {
                    "status": "estimated",
                    "crossing_context_tokens": points[crossing_index]["context_tokens"],
                    "effective_context_tokens": points[crossing_index - 1]["context_tokens"]
                    if crossing_index > 0
                    else None,
                }
            )
        elif (
            len(points) > 1
            and points[-1]["accuracy"] is not None
            and points[-1]["accuracy"] < threshold
            and points[-2]["accuracy"] >= threshold
        ):
            result.update(
                {
                    "status": "provisional",
                    "crossing_context_tokens": points[-1]["context_tokens"],
                    "effective_context_tokens": points[-2]["context_tokens"],
                }
            )
        else:
            result["status"] = "right_censored"
        results.append(result)
    return results


def _weighted_context_points(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, float]] = defaultdict(lambda: {"scored_n": 0.0, "successes": 0.0})
    for row in rows:
        context_tokens = int(row["target_context_tokens"])
        scored_n = int(row.get("scored_n", 0))
        accuracy = row.get("accuracy")
        if scored_n < 0:
            raise ContextAnalysisError("scored_n cannot be negative")
        if scored_n and accuracy is None:
            raise ContextAnalysisError(
                f"accuracy is required when scored_n is positive for context {context_tokens}"
            )
        if scored_n:
            grouped[context_tokens]["scored_n"] += scored_n
            grouped[context_tokens]["successes"] += float(accuracy) * scored_n

    return [
        {
            "context_tokens": context_tokens,
            "scored_n": int(values["scored_n"]),
            "accuracy": (
                values["successes"] / values["scored_n"]
                if values["scored_n"]
                else None
            ),
        }
        for context_tokens, values in sorted(grouped.items())
    ]


def _cell_key(row: Mapping[str, Any]) -> tuple[str, int, float]:
    if not _has_context_dimensions(row):
        raise ContextAnalysisError(
            "summary row must include task_type, target_context_tokens, "
            "and requested_evidence_position"
        )
    return (
        str(row["task_type"]),
        int(row["target_context_tokens"]),
        float(row["requested_evidence_position"]),
    )


def _has_context_dimensions(row: Mapping[str, Any]) -> bool:
    return (
        row.get("task_type") is not None
        and row.get("target_context_tokens") is not None
        and row.get("requested_evidence_position") is not None
    )
