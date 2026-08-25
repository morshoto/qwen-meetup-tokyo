"""Aggregation utilities for raw trial records and experiment notebooks."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from llm_lab.evaluation import TrialResult, TrialStatus
from llm_lab.evaluation.storage import load_trial_results


def aggregate_trials(
    trials: Iterable[TrialResult | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[TrialResult]] = {}
    for value in trials:
        trial = value if isinstance(value, TrialResult) else TrialResult.from_record(value)
        task_type = str(trial.input.get("task_type", "unknown"))
        condition_id = str(trial.input.get("condition_id", "default"))
        key = (trial.experiment_id, task_type, condition_id)
        groups.setdefault(key, []).append(trial)

    summaries: list[dict[str, Any]] = []
    for (experiment_id, task_type, condition_id), group in sorted(groups.items()):
        scored = [
            result.score["correct"]
            for result in group
            if isinstance(result.score.get("correct"), bool)
        ]
        total_seconds = _numeric_values(group, "timing", "total_s")
        ttft_seconds = _numeric_values(group, "timing", "ttft_s")
        prefill_rates = _numeric_values(
            group,
            "timing",
            "prefill_tokens_per_second",
        )
        decode_rates = _numeric_values(
            group,
            "timing",
            "decode_tokens_per_second",
        )
        peak_memory = _numeric_values(group, "memory", "peak_bytes")
        summaries.append(
            {
                "experiment_id": experiment_id,
                "task_type": task_type,
                "condition_id": condition_id,
                "n": len(group),
                "completed_n": sum(result.status == TrialStatus.COMPLETED for result in group),
                "error_n": sum(result.status != TrialStatus.COMPLETED for result in group),
                "scored_n": len(scored),
                "accuracy": sum(scored) / len(scored) if scored else None,
                "target_context_tokens": _common_input_value(
                    group, "target_context_tokens"
                ),
                "requested_evidence_position": _common_input_value(
                    group, "requested_evidence_position"
                ),
                "actual_evidence_position": _median(
                    _numeric_input_values(group, "actual_evidence_position")
                ),
                "median_total_s": _median(total_seconds),
                "median_ttft_s": _median(ttft_seconds),
                "median_prefill_tokens_per_second": _median(prefill_rates),
                "median_decode_tokens_per_second": _median(decode_rates),
                "median_peak_memory_bytes": _median(peak_memory),
            }
        )
    return summaries


def aggregate_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return aggregate_trials(load_trial_results(path))


def write_summary_csv(path: str | Path, summaries: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in summaries]
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _numeric_values(
    trials: Iterable[TrialResult],
    section: str,
    key: str,
) -> list[float]:
    values: list[float] = []
    for trial in trials:
        value = getattr(trial, section).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _median(values: list[float]) -> float | None:
    return median(values) if values else None


def _common_input_value(
    trials: Iterable[TrialResult],
    key: str,
) -> Any:
    values = [trial.input.get(key) for trial in trials]
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _numeric_input_values(
    trials: Iterable[TrialResult],
    key: str,
) -> list[float]:
    values: list[float] = []
    for trial in trials:
        value = trial.input.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values
