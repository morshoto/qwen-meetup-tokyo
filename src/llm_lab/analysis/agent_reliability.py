"""Measured-data analysis helpers for exp_004 agent reliability."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from llm_lab.evaluation import TrialResult, TrialStatus


class AgentAnalysisError(ValueError):
    """Raised when agent dimensions or measurement provenance are incomplete."""


FAILURE_CATEGORIES = frozenset(
    {"retrieval", "state_tracking", "tool_planning", "runtime", "success"}
)


def aggregate_agent_trials(
    trials: Iterable[TrialResult | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate agent trials without dropping runtime failures."""

    groups: dict[tuple[str, str, int, float], list[TrialResult]] = defaultdict(list)
    for value in trials:
        trial = value if isinstance(value, TrialResult) else TrialResult.from_record(value)
        variant = trial.input.get("variant_condition_id")
        task_type = trial.input.get("task_type")
        length = trial.input.get("trajectory_length")
        position = trial.input.get("requested_critical_position")
        if variant is None or task_type is None or length is None or position is None:
            raise AgentAnalysisError(
                "agent trial must include variant_condition_id, task_type, "
                "trajectory_length, and requested_critical_position"
            )
        try:
            normalized_length = int(length)
            normalized_position = float(position)
        except (TypeError, ValueError) as error:
            raise AgentAnalysisError("agent trajectory dimensions must be numeric") from error
        if normalized_length < 1 or not 0.0 <= normalized_position <= 1.0:
            raise AgentAnalysisError("agent trajectory dimensions are out of range")
        groups[
            (str(variant), str(task_type), normalized_length, normalized_position)
        ].append(trial)

    rows: list[dict[str, Any]] = []
    for (variant, task_type, length, position), group in sorted(groups.items()):
        categories = Counter(_failure_category(trial) for trial in group)
        scored = [
            trial.score.get("correct")
            for trial in group
            if isinstance(trial.score.get("correct"), bool)
        ]
        metrics = [_metrics(trial) for trial in group]
        context_tokens = [
            int(item["max_input_tokens"])
            for item in metrics
            if isinstance(item.get("max_input_tokens"), int)
            and not isinstance(item.get("max_input_tokens"), bool)
        ]
        tool_calls = sum(
            _nonnegative_int(item.get("tool_call_n"), "tool_call_n")
            for item in metrics
        )
        valid_tool_calls = sum(
            _nonnegative_int(item.get("valid_tool_call_n"), "valid_tool_call_n")
            for item in metrics
        )
        attempted_n = len(group)
        correct_n = sum(value is True for value in scored)
        reused_n = sum(bool(item.get("critical_fact_reused")) for item in metrics)
        rows.append(
            {
                "experiment_id": group[0].experiment_id,
                "task_type": task_type,
                "variant_condition_id": variant,
                "trajectory_length": length,
                "requested_critical_position": position,
                "n": attempted_n,
                "attempted_n": attempted_n,
                "completed_n": sum(
                    trial.status == TrialStatus.COMPLETED for trial in group
                ),
                "error_n": sum(
                    trial.status != TrialStatus.COMPLETED for trial in group
                ),
                "scored_n": len(scored),
                "correct_n": correct_n,
                "scored_accuracy": correct_n / len(scored) if scored else None,
                "final_task_success": correct_n / attempted_n if attempted_n else None,
                "critical_fact_reused_n": reused_n,
                "critical_fact_reuse_rate": reused_n / attempted_n if attempted_n else None,
                "tool_call_n": tool_calls,
                "valid_tool_call_n": valid_tool_calls,
                "tool_call_validity": valid_tool_calls / tool_calls if tool_calls else None,
                "repeated_action_n": sum(
                    _nonnegative_int(item.get("repeated_action_n"), "repeated_action_n")
                    for item in metrics
                ),
                "recovery_n": sum(
                    _nonnegative_int(item.get("recovery_n"), "recovery_n")
                    for item in metrics
                ),
                "planning_error_n": sum(
                    _nonnegative_int(item.get("planning_error_n"), "planning_error_n")
                    for item in metrics
                ),
                "total_input_tokens": sum(
                    _nonnegative_int(item.get("total_input_tokens"), "total_input_tokens")
                    for item in metrics
                ),
                "trajectory_context_tokens": (
                    int(median(context_tokens)) if context_tokens else None
                ),
                "failure_category_counts": dict(sorted(categories.items())),
            }
        )
    return rows


def require_measured_trials(
    trials: Iterable[TrialResult | Mapping[str, Any]],
    *,
    output_directory: Path | None = None,
) -> list[TrialResult]:
    """Reject empty, fixture-only, or mixed fixture/model analysis inputs."""

    del output_directory
    materialized = [
        value if isinstance(value, TrialResult) else TrialResult.from_record(value)
        for value in trials
    ]
    if not materialized:
        raise AgentAnalysisError("measured agent trials are required")
    if any(_is_fixture(trial) for trial in materialized):
        raise AgentAnalysisError(
            "fixture-only smoke trials cannot be reported as model measurements"
        )
    return materialized


def missing_agent_cells(
    rows: Iterable[Mapping[str, Any]],
    *,
    variant_ids: Iterable[str],
    trajectory_lengths: Iterable[int],
    critical_positions: Iterable[float],
    task_types: Iterable[str],
) -> list[tuple[str, str, int, float]]:
    """Return declared variant/task/length/position cells absent from summaries."""

    present = {
        (
            str(row.get("variant_condition_id")),
            str(row.get("task_type")),
            int(row["trajectory_length"]),
            float(row["requested_critical_position"]),
        )
        for row in rows
    }
    return [
        (variant, task_type, length, position)
        for variant in variant_ids
        for task_type in task_types
        for length in trajectory_lengths
        for position in critical_positions
        if (variant, task_type, length, position) not in present
    ]


def validate_complete_matrix(
    rows: Iterable[Mapping[str, Any]],
    *,
    variant_ids: Iterable[str],
    trajectory_lengths: Iterable[int],
    critical_positions: Iterable[float],
    task_types: Iterable[str],
) -> None:
    missing = missing_agent_cells(
        rows,
        variant_ids=variant_ids,
        trajectory_lengths=trajectory_lengths,
        critical_positions=critical_positions,
        task_types=task_types,
    )
    if missing:
        raise AgentAnalysisError(f"missing agent measurement cells: {missing[:5]}")


def plot_reliability_by_length(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
) -> None:
    """Plot final task success and critical-fact reuse by trajectory length."""

    import matplotlib.pyplot as plt

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant_condition_id"])].append(row)
    figure, axis = plt.subplots()
    for variant, variant_rows in sorted(grouped.items()):
        ordered = sorted(variant_rows, key=lambda row: int(row["trajectory_length"]))
        x = [row["trajectory_length"] for row in ordered]
        axis.plot(
            x,
            [row["final_task_success"] for row in ordered],
            marker="o",
            label=f"{variant} success",
        )
        axis.plot(
            x,
            [row["critical_fact_reuse_rate"] for row in ordered],
            marker="x",
            linestyle="--",
            label=f"{variant} fact reuse",
        )
    axis.set_xlabel("Trajectory length (tool observations)")
    axis.set_ylabel("Rate")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    axis.grid(True, alpha=0.3)
    _save_figure(figure, output_path)


def plot_reliability_by_position(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
) -> None:
    """Plot final task success by critical-information position."""

    import matplotlib.pyplot as plt

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant_condition_id"])].append(row)
    figure, axis = plt.subplots()
    for variant, variant_rows in sorted(grouped.items()):
        by_length: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in variant_rows:
            by_length[int(row["trajectory_length"])].append(row)
        for length, length_rows in sorted(by_length.items()):
            ordered = sorted(
                length_rows,
                key=lambda row: float(row["requested_critical_position"]),
            )
            axis.plot(
                [row["requested_critical_position"] for row in ordered],
                [row["final_task_success"] for row in ordered],
                marker="o",
                label=f"{variant} traj{length}",
            )
    axis.set_xlabel("Critical observation position")
    axis.set_ylabel("Final task success")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    axis.grid(True, alpha=0.3)
    _save_figure(figure, output_path)


def _failure_category(trial: TrialResult) -> str:
    metrics = _metrics(trial)
    category = metrics.get("failure_category")
    if not isinstance(category, str) or category not in FAILURE_CATEGORIES:
        raise AgentAnalysisError(
            f"unsupported or missing failure category for {trial.trial_id}"
        )
    return category


def _metrics(trial: TrialResult) -> Mapping[str, Any]:
    value = trial.input.get("metrics", {})
    if not isinstance(value, Mapping):
        raise AgentAnalysisError(f"metrics must be an object for {trial.trial_id}")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AgentAnalysisError(f"{name} must be a non-negative integer")
    return value


def _is_fixture(trial: TrialResult) -> bool:
    return bool(trial.input.get("fixture_only")) or trial.environment.get("purpose") == "harness_smoke_only"


def _save_figure(figure: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.clf()
