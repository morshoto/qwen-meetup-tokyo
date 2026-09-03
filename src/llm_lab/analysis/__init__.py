"""Reusable aggregation, statistics, and plot preparation."""

from .aggregation import aggregate_jsonl, aggregate_trials, write_summary_csv
from .effective_context import (
    ContextAnalysisError,
    effective_context_by_task,
    effective_context_by_task_and_position,
    missing_context_cells,
    position_gap_rows,
    position_curve_rows,
)
from .interaction import (
    InteractionAnalysisError,
    effective_context_by_variant_and_task,
    interaction_report,
    matched_cell_rows,
    relative_degradation_rows,
)
from .quantization import (
    QuantizationAnalysisError,
    recommend_baseline,
    tradeoff_rows,
)
from .rescoring import (
    EXACT_MATCH,
    FORMAT_FAILURE,
    MISMATCH,
    RUNTIME_FAILURE,
    RescoredTrial,
    comparison_rows,
    render_report,
    rescore_trial,
    rescore_trials,
    sha256_file,
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
from .uncertainty import UncertaintyAnalysisError, task_level_wilson, wilson_interval
from .feasibility import (
    FEASIBILITY_CLASSIFICATIONS,
    FeasibilityAnalysisError,
    classify_feasibility,
    classify_feasibility_by_length,
)

__all__ = [
    "ContextAnalysisError",
    "InteractionAnalysisError",
    "QuantizationAnalysisError",
    "aggregate_jsonl",
    "aggregate_trials",
    "effective_context_by_task",
    "effective_context_by_task_and_position",
    "effective_context_by_variant_and_task",
    "interaction_report",
    "matched_cell_rows",
    "missing_context_cells",
    "position_gap_rows",
    "position_curve_rows",
    "recommend_baseline",
    "relative_degradation_rows",
    "tradeoff_rows",
    "write_summary_csv",
    "EXACT_MATCH",
    "FORMAT_FAILURE",
    "MISMATCH",
    "RUNTIME_FAILURE",
    "RescoredTrial",
    "comparison_rows",
    "render_report",
    "rescore_trial",
    "rescore_trials",
    "sha256_file",
    "AgentAnalysisError",
    "aggregate_agent_trials",
    "missing_agent_cells",
    "plot_reliability_by_length",
    "plot_reliability_by_position",
    "require_measured_trials",
    "validate_complete_matrix",
    "UncertaintyAnalysisError",
    "task_level_wilson",
    "wilson_interval",
    "FEASIBILITY_CLASSIFICATIONS",
    "FeasibilityAnalysisError",
    "classify_feasibility",
    "classify_feasibility_by_length",
]
