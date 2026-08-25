"""Reusable aggregation, statistics, and plot preparation."""

from .aggregation import aggregate_jsonl, aggregate_trials, write_summary_csv

__all__ = ["aggregate_jsonl", "aggregate_trials", "write_summary_csv"]
