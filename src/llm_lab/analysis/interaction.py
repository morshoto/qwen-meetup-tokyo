"""Matched-cell analysis for context-length × quantization experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .effective_context import effective_context_by_task


class InteractionAnalysisError(ValueError):
    """Raised when matched interaction dimensions or metrics are incomplete."""


def matched_cell_rows(
    summaries: Iterable[Mapping[str, Any]],
    *,
    variant_ids: Iterable[str],
    context_lengths: Iterable[int],
    evidence_positions: Iterable[float],
    task_types: Iterable[str],
) -> list[dict[str, Any]]:
    """Validate and sort one summary row per matched task/context cell.

    The context instance ID and text hash must agree across variants for every
    task/context/position key. This makes a comparison fail closed when a
    runner accidentally regenerated a different context for one artifact.
    """

    rows = [dict(row) for row in summaries]
    expected = {
        (str(variant), str(task), int(context), float(position))
        for variant in variant_ids
        for task in task_types
        for context in context_lengths
        for position in evidence_positions
    }
    actual: set[tuple[str, str, int, float]] = set()
    context_identity: dict[tuple[str, int, float], tuple[str, str]] = {}
    for row in rows:
        key = _row_key(row)
        if key in actual:
            raise InteractionAnalysisError(f"duplicate matched cell: {key}")
        actual.add(key)
        instance_id = row.get("context_instance_id")
        context_sha256 = row.get("context_sha256")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise InteractionAnalysisError(
                f"matched cell {key} is missing context_instance_id"
            )
        if not isinstance(context_sha256, str) or not context_sha256.strip():
            raise InteractionAnalysisError(
                f"matched cell {key} is missing context_sha256"
            )
        base_key = key[1:]
        identity = (instance_id, context_sha256)
        previous = context_identity.setdefault(base_key, identity)
        if previous != identity:
            raise InteractionAnalysisError(
                f"matched cells do not share one context instance: {base_key}"
            )

    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise InteractionAnalysisError(
            f"missing matched cells: {sorted(missing)!r}"
        )
    if unexpected:
        raise InteractionAnalysisError(
            f"unexpected matched cells: {sorted(unexpected)!r}"
        )
    return sorted(
        rows,
        key=lambda row: _row_key(row),
    )


def relative_degradation_rows(
    summaries: Iterable[Mapping[str, Any]],
    *,
    baseline_context_tokens: int = 8192,
) -> list[dict[str, Any]]:
    """Compare each cell with its variant/task/position short-context baseline."""

    rows = [dict(row) for row in summaries]
    baselines: dict[tuple[str, str, float], Mapping[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key[2] == baseline_context_tokens:
            baselines[(key[0], key[1], key[3])] = row

    output: list[dict[str, Any]] = []
    for row in rows:
        key = _row_key(row)
        baseline = baselines.get((key[0], key[1], key[3]))
        if baseline is None:
            raise InteractionAnalysisError(
                "missing short-context baseline for "
                f"variant={key[0]}, task={key[1]}, position={key[3]}"
            )
        current_accuracy = _accuracy(row)
        baseline_accuracy = _accuracy(baseline)
        enriched = dict(row)
        enriched["short_context_baseline_accuracy"] = baseline_accuracy
        if current_accuracy is None or baseline_accuracy is None:
            enriched["accuracy_degradation"] = None
            enriched["relative_degradation"] = None
        else:
            enriched["accuracy_degradation"] = baseline_accuracy - current_accuracy
            enriched["relative_degradation"] = (
                (baseline_accuracy - current_accuracy) / baseline_accuracy
                if baseline_accuracy != 0
                else None
            )
        output.append(enriched)
    return sorted(output, key=_row_key)


def interaction_report(
    summaries: Iterable[Mapping[str, Any]],
    *,
    reference_variant: str,
    approx_constant_gap_tolerance: float = 0.10,
) -> list[dict[str, Any]]:
    """Describe whether each variant's gap from a reference changes by context.

    Gaps are weighted by the number of matched scored trials across evidence
    positions. The returned classification is descriptive and deliberately
    does not claim statistical significance.
    """

    if approx_constant_gap_tolerance < 0:
        raise ValueError("approx_constant_gap_tolerance cannot be negative")
    rows = [dict(row) for row in summaries]
    by_key = {_row_key(row): row for row in rows}
    reports: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        variant, task_type, context_tokens, position = _row_key(row)
        if variant == reference_variant:
            continue
        reference = by_key.get((reference_variant, task_type, context_tokens, position))
        if reference is None:
            raise InteractionAnalysisError(
                "missing reference cell for "
                f"variant={variant}, task={task_type}, context={context_tokens}, position={position}"
            )
        current_accuracy = _accuracy(row)
        reference_accuracy = _accuracy(reference)
        matched_n = min(
            _scored_n(row),
            _scored_n(reference),
        )
        if current_accuracy is None or reference_accuracy is None or matched_n < 1:
            continue
        grouped[(task_type, variant, context_tokens)].append(
            (reference_accuracy - current_accuracy, matched_n)
        )

    grouped_contexts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (task_type, variant, context_tokens), gaps in grouped.items():
        total_n = sum(weight for _, weight in gaps)
        weighted_gap = sum(gap * weight for gap, weight in gaps) / total_n
        grouped_contexts[(task_type, variant)].append(
            {
                "context_tokens": context_tokens,
                "quantization_gap": weighted_gap,
                "matched_n": total_n,
            }
        )

    for (task_type, variant), points in sorted(grouped_contexts.items()):
        points = sorted(points, key=lambda point: point["context_tokens"])
        gaps = [float(point["quantization_gap"]) for point in points]
        if len(gaps) < 2:
            classification = "insufficient_data"
            gap_range = None
            gap_change = None
        else:
            gap_range = max(gaps) - min(gaps)
            gap_change = gaps[-1] - gaps[0]
            classification = (
                "approximately_constant"
                if gap_range <= approx_constant_gap_tolerance
                else "context_dependent"
            )
        reports.append(
            {
                "task_type": task_type,
                "variant_condition_id": variant,
                "reference_variant": reference_variant,
                "context_points": points,
                "shortest_context_tokens": points[0]["context_tokens"] if points else None,
                "largest_context_tokens": points[-1]["context_tokens"] if points else None,
                "shortest_context_gap": gaps[0] if gaps else None,
                "largest_context_gap": gaps[-1] if gaps else None,
                "gap_range": gap_range,
                "gap_change": gap_change,
                "matched_n": sum(int(point["matched_n"]) for point in points),
                "approx_constant_gap_tolerance": approx_constant_gap_tolerance,
                "classification": classification,
            }
        )
    return reports


def effective_context_by_variant_and_task(
    summaries: Iterable[Mapping[str, Any]],
    *,
    baseline_context_tokens: int = 8192,
    alpha: float = 0.90,
    minimum_baseline_accuracy: float = 0.80,
) -> list[dict[str, Any]]:
    """Calculate the existing effective-context rule per variant and task."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        key = _row_key(row)
        grouped[(key[0], key[1])].append(row)

    output: list[dict[str, Any]] = []
    for (variant, task_type), rows in sorted(grouped.items()):
        [result] = effective_context_by_task(
            rows,
            baseline_context_tokens=baseline_context_tokens,
            alpha=alpha,
            minimum_baseline_accuracy=minimum_baseline_accuracy,
        )
        result = dict(result)
        result["variant_condition_id"] = variant
        result["task_type"] = task_type
        output.append(result)
    return output


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, int, float]:
    required = (
        row.get("variant_condition_id"),
        row.get("task_type"),
        row.get("target_context_tokens"),
        row.get("requested_evidence_position"),
    )
    if any(value is None for value in required):
        raise InteractionAnalysisError(
            "summary row must include variant_condition_id, task_type, "
            "target_context_tokens, and requested_evidence_position"
        )
    return (
        str(required[0]),
        str(required[1]),
        int(required[2]),
        float(required[3]),
    )


def _accuracy(row: Mapping[str, Any]) -> float | None:
    value = row.get("accuracy")
    return None if value is None else float(value)


def _scored_n(row: Mapping[str, Any]) -> int:
    value = row.get("scored_n", 0)
    if isinstance(value, bool):
        raise InteractionAnalysisError("scored_n must be an integer")
    return int(value)
