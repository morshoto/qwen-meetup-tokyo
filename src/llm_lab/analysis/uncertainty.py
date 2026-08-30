"""Uncertainty estimates whose resampling unit is an independent task."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite, sqrt
from typing import Any, Iterable, Mapping


class UncertaintyAnalysisError(ValueError):
    """Raised when task-level uncertainty cannot be computed safely."""


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binary rate.

    The interval is deliberately defined for integer task outcomes. Runtime
    and invalid-output failures should therefore be represented as attempted
    tasks with zero success, rather than removed from ``trials``.
    """

    if isinstance(successes, bool) or isinstance(trials, bool):
        raise UncertaintyAnalysisError("successes and trials must be integers")
    if not isinstance(successes, int) or not isinstance(trials, int):
        raise UncertaintyAnalysisError("successes and trials must be integers")
    if trials < 1 or successes < 0 or successes > trials:
        raise UncertaintyAnalysisError(
            f"invalid binary counts: successes={successes}, trials={trials}"
        )
    if not isinstance(z, (int, float)) or isinstance(z, bool) or not isfinite(float(z)) or z <= 0:
        raise UncertaintyAnalysisError("z must be a positive finite number")
    z_value = float(z)
    n = float(trials)
    p = successes / n
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / n
    center = (p + z_squared / (2.0 * n)) / denominator
    half_width = z_value * sqrt((p * (1.0 - p) + z_squared / (4.0 * n)) / n) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def task_level_wilson(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_keys: Iterable[str],
    task_key: str = "task_id",
    metric_fields: Mapping[str, str],
    attempted_key: str = "attempted_n",
) -> list[dict[str, Any]]:
    """Calculate Wilson intervals with one independent task per observation.

    ``rows`` must contain one capability row per task within each group. A
    repeated deterministic prompt is not silently treated as a larger sample:
    duplicate task IDs raise an error. Missing/false metric values count as a
    failed attempted task, preserving runtime and invalid-output failures in
    the denominator.
    """

    selected_group_keys = tuple(str(key) for key in group_keys)
    if not selected_group_keys:
        raise UncertaintyAnalysisError("at least one group key is required")
    if not metric_fields:
        raise UncertaintyAnalysisError("at least one metric field is required")
    groups: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        try:
            group = tuple(row[key] for key in selected_group_keys)
            task_id = row[task_key]
        except KeyError as error:
            raise UncertaintyAnalysisError(
                f"task-level uncertainty is missing required field: {error.args[0]}"
            ) from error
        if task_id is None or not str(task_id).strip():
            raise UncertaintyAnalysisError("task-level uncertainty requires task_id")
        task_id = str(task_id)
        if task_id in groups[group]:
            raise UncertaintyAnalysisError(
                "duplicate task within uncertainty group; repeated prompts are "
                f"not independent: group={group}, task_id={task_id}"
            )
        attempted = row.get(attempted_key, 1)
        if isinstance(attempted, bool) or not isinstance(attempted, int) or attempted != 1:
            raise UncertaintyAnalysisError(
                "task-level uncertainty requires exactly one capability observation "
                f"per task; task_id={task_id}, attempted_n={attempted!r}"
            )
        groups[group][task_id] = row

    output: list[dict[str, Any]] = []
    for group, task_rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result = dict(zip(selected_group_keys, group))
        result["task_n"] = len(task_rows)
        result["attempted_n"] = len(task_rows)
        for metric_name, field_name in metric_fields.items():
            successes = 0
            for row in task_rows.values():
                value = row.get(field_name)
                if isinstance(value, bool):
                    successes += int(value)
                elif isinstance(value, int) and value in (0, 1):
                    successes += value
                elif value is not None:
                    raise UncertaintyAnalysisError(
                        f"metric {field_name!r} must be boolean or 0/1, found {value!r}"
                    )
            low, high = wilson_interval(successes, len(task_rows))
            result.update(
                {
                    f"{metric_name}_success_n": successes,
                    f"{metric_name}_rate": successes / len(task_rows),
                    f"{metric_name}_ci_low": low,
                    f"{metric_name}_ci_high": high,
                }
            )
        output.append(result)
    return output
