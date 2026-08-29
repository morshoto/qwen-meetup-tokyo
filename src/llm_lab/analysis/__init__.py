"""Reusable aggregation, statistics, and plot preparation."""

from .aggregation import aggregate_jsonl, aggregate_trials, write_summary_csv
from .effective_context import (
    ContextAnalysisError,
    effective_context_by_task,
    effective_context_by_task_and_position,
    missing_context_cells,
    position_curve_rows,
)
from .quantization import (
    QuantizationAnalysisError,
    recommend_baseline,
    tradeoff_rows,
)
from .agent_reliability import (
    AgentAnalysisError,
    aggregate_agent_trials,
    missing_agent_cells,
    plot_reliability_by_length,
    plot_reliability_by_position,
    require_measured_trials,
    validate_complete_matrix,
)

__all__ = [
    "ContextAnalysisError",
    "QuantizationAnalysisError",
    "aggregate_jsonl",
    "aggregate_trials",
    "effective_context_by_task",
    "effective_context_by_task_and_position",
    "missing_context_cells",
    "position_curve_rows",
    "recommend_baseline",
    "tradeoff_rows",
    "write_summary_csv",
    "AgentAnalysisError",
    "aggregate_agent_trials",
    "missing_agent_cells",
    "plot_reliability_by_length",
    "plot_reliability_by_position",
    "require_measured_trials",
    "validate_complete_matrix",
]
