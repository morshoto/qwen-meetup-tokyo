"""Reusable aggregation, statistics, and plot preparation."""

from .aggregation import aggregate_jsonl, aggregate_trials, write_summary_csv
from .effective_context import (
    ContextAnalysisError,
    effective_context_by_task,
    effective_context_by_task_and_position,
    missing_context_cells,
    position_curve_rows,
)

__all__ = [
    "ContextAnalysisError",
    "aggregate_jsonl",
    "aggregate_trials",
    "effective_context_by_task",
    "effective_context_by_task_and_position",
    "missing_context_cells",
    "position_curve_rows",
    "write_summary_csv",
]
