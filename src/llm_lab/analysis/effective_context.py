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


def position_gap_rows(
    summaries: Iterable[Mapping[str, Any]],
    *,
    edge_positions: tuple[float, float] = (0.05, 0.95),
    middle_position: float = 0.50,
) -> list[dict[str, Any]]:
    """Calculate the declared edge-minus-middle position gap by context.

    ``A_edge`` is the scored accuracy pooled across the beginning and end
    positions. The pooling is weighted by scored trial count so a runtime
    exclusion cannot be mistaken for a zero-accuracy observation. A complete
    beginning/middle/end triplet is required for every task/context group.
    """

    if len(edge_positions) != 2 or edge_positions[0] >= edge_positions[1]:
        raise ValueError("edge_positions must contain two ordered positions")
    grouped: dict[tuple[str, int], dict[float, Mapping[str, Any]]] = defaultdict(dict)
    for row in summaries:
        key = _cell_key(row)
        cell_key = (key[0], key[1])
        position = key[2]
        if position in grouped[cell_key]:
            raise ContextAnalysisError(
                f"duplicate position cell for task={key[0]}, context={key[1]}, "
                f"position={position}"
            )
        grouped[cell_key][position] = row

    output: list[dict[str, Any]] = []
    required_positions = (*edge_positions, middle_position)
    for (task_type, context_tokens), cells in sorted(grouped.items()):
        missing = [position for position in required_positions if position not in cells]
        if missing:
            raise ContextAnalysisError(
                "missing position cells for "
                f"task={task_type}, context={context_tokens}: {missing}"
            )
        edge_scored_n = sum(_scored_n(cells[position]) for position in edge_positions)
        middle_scored_n = _scored_n(cells[middle_position])
        edge_accuracy = _weighted_accuracy(
            (cells[position] for position in edge_positions),
            edge_scored_n,
        )
        middle_accuracy = _weighted_accuracy(
            (cells[middle_position],),
            middle_scored_n,
        )
        if edge_accuracy is None or middle_accuracy is None:
            raise ContextAnalysisError(
                "position cell has no scored accuracy for "
                f"task={task_type}, context={context_tokens}"
            )
        output.append(
            {
                "task_type": task_type,
                "target_context_tokens": context_tokens,
                "edge_positions": list(edge_positions),
                "middle_position": middle_position,
                "edge_accuracy": edge_accuracy,
                "middle_accuracy": middle_accuracy,
                "position_gap": edge_accuracy - middle_accuracy,
                "edge_scored_n": edge_scored_n,
                "middle_scored_n": middle_scored_n,
            }
        )
    return output


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

    _validate_effective_context_parameters(alpha, minimum_baseline_accuracy)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        _cell_key(row)
        grouped[str(row["task_type"])].append(row)

    return [
        _effective_context_for_rows(
            task_type,
            rows,
            baseline_context_tokens=baseline_context_tokens,
            alpha=alpha,
            minimum_baseline_accuracy=minimum_baseline_accuracy,
        )
        for task_type, rows in sorted(grouped.items())
    ]


def effective_context_by_task_and_position(
    summaries: Iterable[Mapping[str, Any]],
    *,
    baseline_context_tokens: int = 8192,
    alpha: float = 0.90,
    minimum_baseline_accuracy: float = 0.80,
) -> list[dict[str, Any]]:
    """Calculate the same effective-context rule separately by evidence position.

    The aggregate task-level result remains available from
    :func:`effective_context_by_task`. This companion prevents a pooled average
    across positions from hiding a middle-position collapse.
    """

    _validate_effective_context_parameters(alpha, minimum_baseline_accuracy)
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        _cell_key(row)
        grouped[(str(row["task_type"]), float(row["requested_evidence_position"]))].append(row)

    results: list[dict[str, Any]] = []
    for (task_type, evidence_position), rows in sorted(grouped.items()):
        result = _effective_context_for_rows(
            task_type,
            rows,
            baseline_context_tokens=baseline_context_tokens,
            alpha=alpha,
            minimum_baseline_accuracy=minimum_baseline_accuracy,
        )
        result["evidence_position"] = evidence_position
        results.append(result)
    return results


def _validate_effective_context_parameters(
    alpha: float,
    minimum_baseline_accuracy: float,
) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 < minimum_baseline_accuracy <= 1.0:
        raise ValueError("minimum_baseline_accuracy must be between 0 and 1")


def _effective_context_for_rows(
    task_type: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_context_tokens: int,
    alpha: float,
    minimum_baseline_accuracy: float,
) -> dict[str, Any]:
    points = _weighted_context_points(rows)
    if not points:
        raise ContextAnalysisError(f"{task_type} has no scored context points")
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
        return result

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
    return result


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


def _scored_n(row: Mapping[str, Any]) -> int:
    value = row.get("scored_n", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContextAnalysisError("scored_n must be a non-negative integer")
    return value


def _weighted_accuracy(
    rows: Iterable[Mapping[str, Any]],
    scored_n: int,
) -> float | None:
    if scored_n == 0:
        return None
    successes = 0.0
    for row in rows:
        row_n = _scored_n(row)
        accuracy = row.get("accuracy")
        if row_n and accuracy is None:
            raise ContextAnalysisError("accuracy is required when scored_n is positive")
        if row_n:
            successes += float(accuracy) * row_n
    return successes / scored_n


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
