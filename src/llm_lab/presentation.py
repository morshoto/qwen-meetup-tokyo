"""Presentation-oriented plots for the measured experiment artifacts.

These helpers deliberately read named, committed processed artifacts instead
of relying on the notebook's exploratory phase selection.  The figures are
intended to make one bounded result legible on a presentation slide.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


PALETTE = {
    "q8_0": "#0072B2",
    "q6_k": "#009E73",
    "q5_k_m": "#E69F00",
    "q4_k_m": "#D55E00",
    "literal_retrieval": "#0072B2",
    "semantic_retrieval": "#E69F00",
    "multi_hop": "#009E73",
    "success": "#009E73",
    "timeout": "#D55E00",
    "reference": "#555555",
}

TASK_LABELS = {
    "literal_retrieval": "Literal",
    "semantic_retrieval": "Semantic",
    "multi_hop": "Multi-hop",
}

VARIANT_LABELS = {
    "q8_0": "Q8_0",
    "q6_k": "Q6_K",
    "q5_k_m": "Q5_K_M",
    "q4_k_m": "Q4_K_M",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _percent_axis(axis: plt.Axes) -> None:
    axis.set_ylim(0, 1.08)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))


def _annotate_bars(axis: plt.Axes, bars, labels: list[str], offset: float = 0.025) -> None:
    for bar, label in zip(bars, labels):
        height = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            max(height + offset, offset),
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def plot_exp001_presentation(results_dir: str | Path) -> list[str]:
    """Create the two slide-ready exp_001 figures."""

    _style()
    results = Path(results_dir)
    feasibility = pd.read_csv(results / "processed/feasibility-summary.csv")
    pilot = pd.read_csv(results / "processed/pilot-matched-summary.csv")
    output_dir = results / "figures"
    output_paths: list[str] = []

    labels = [f"{int(value) // 1024}K" for value in feasibility["target_context_tokens"]]
    completion = feasibility["completed_n"] / feasibility["attempted_n"]
    colors = [
        PALETTE["success"] if value > 0 else PALETTE["timeout"] for value in completion
    ]
    fig, axis = plt.subplots(figsize=(10, 5.6))
    bars = axis.bar(labels, completion, color=colors, width=0.58)
    _percent_axis(axis)
    axis.set_xlabel("Target context length")
    axis.set_ylabel("Completed trials")
    axis.set_title("")
    _annotate_bars(
        axis,
        bars,
        [
            f"{int(row.completed_n)}/{int(row.attempted_n)}\ncompleted"
            if row.completed_n
            else f"{int(row.attempted_n)}/{int(row.attempted_n)}\ntimeout"
            for row in feasibility.itertuples()
        ],
    )
    axis.text(
        0.98,
        0.08,
        "Bounded feasibility result\n—not a model hard limit",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color="#555555",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F4F4F4", "edgecolor": "none"},
    )
    fig.suptitle("64K completed; larger probes hit the 900 s timeout", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.925, "Qwen3.8-27B · Q8_0 · llama.cpp · evidence at 50% · 3 tasks per condition", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.11, right=0.98)
    output_paths.append(_save(fig, output_dir / "presentation-feasibility-boundary.png"))

    task_order = ["literal_retrieval", "semantic_retrieval", "multi_hop"]
    context_order = [8192, 32768]
    frame = pilot.copy()
    frame["task_type"] = pd.Categorical(frame["task_type"], task_order, ordered=True)
    frame = frame.sort_values(["task_type", "target_context_tokens"])
    x = np.arange(len(task_order))
    width = 0.34
    fig, axis = plt.subplots(figsize=(10, 5.6))
    for index, context in enumerate(context_order):
        subset = frame[frame["target_context_tokens"] == context].set_index("task_type").loc[task_order]
        bars = axis.bar(
            x + (index - 0.5) * width,
            subset["end_to_end_success"],
            width,
            label=f"{context // 1024}K",
            color=[PALETTE[task] for task in task_order],
            alpha=0.62 if index == 0 else 1.0,
            edgecolor="white",
            linewidth=0.8,
        )
        _annotate_bars(
            axis,
            bars,
            [f"{int(row.correct_n)}/{int(row.attempted_n)}" for row in subset.itertuples()],
            offset=0.018,
        )
    axis.axhline(0.8, color=PALETTE["reference"], linestyle="--", linewidth=1.3)
    axis.text(len(task_order) - 0.45, 0.815, "0.80 baseline gate", color="#555555", fontsize=10)
    _percent_axis(axis)
    axis.set_xticks(x, [TASK_LABELS[task] for task in task_order])
    axis.set_xlabel("Task family")
    axis.set_ylabel("End-to-end success")
    axis.set_title("")
    axis.legend(title="Input context", ncol=2, loc="upper right", frameon=False)
    fig.suptitle("Baseline quality is task-shaped—not one context curve", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.925, "Q8_0 · 10 independent tasks per family and context · p50 evidence position", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.11, right=0.98)
    output_paths.append(_save(fig, output_dir / "presentation-baseline-by-task.png"))
    return output_paths


def _exp002_frame(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_id, group in summary.groupby("variant_condition_id", sort=False):
        rows.append(
            {
                "variant_id": variant_id,
                "label": VARIANT_LABELS.get(variant_id, variant_id),
                "artifact_size_gb": float(group["artifact_size_bytes"].iloc[0]) / 1e9,
                "attempted_n": int(group["attempted_n"].sum()),
                "end_to_end": group["correct_n"].sum() / group["attempted_n"].sum(),
                "answer_bearing": group["answer_bearing_correct_n"].sum()
                / group["attempted_n"].sum(),
                "format_valid": group["format_valid_n"].sum() / group["attempted_n"].sum(),
            }
        )
    return pd.DataFrame(rows).set_index("variant_id").loc[["q8_0", "q6_k", "q5_k_m", "q4_k_m"]].reset_index()


def plot_exp002_presentation(results_dir: str | Path) -> list[str]:
    """Create quality/footprint and metric-specific equivalence figures."""

    _style()
    results = Path(results_dir)
    summary = pd.read_csv(results / "processed/summary.csv")
    frame = _exp002_frame(summary)
    output_dir = results / "figures"
    output_paths: list[str] = []

    fig, (size_axis, quality_axis) = plt.subplots(
        1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [0.9, 1.25]}
    )
    colors = [PALETTE[variant] for variant in frame["variant_id"]]
    bars = size_axis.barh(frame["label"], frame["artifact_size_gb"], color=colors, height=0.58)
    size_axis.invert_yaxis()
    size_axis.set_xlabel("GGUF artifact size (GB)")
    size_axis.set_title("Footprint", fontweight="bold", loc="left")
    size_axis.grid(axis="x")
    size_axis.grid(axis="y", visible=False)
    for bar, value in zip(bars, frame["artifact_size_gb"]):
        size_axis.text(value + 0.25, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontweight="bold")
    size_axis.set_xlim(0, max(frame["artifact_size_gb"]) * 1.18)

    metrics = [
        ("answer_bearing", "Answer-bearing", "o", "#009E73"),
        ("format_valid", "Format-valid", "s", "#E69F00"),
        ("end_to_end", "End-to-end", "D", "#D55E00"),
    ]
    y = np.arange(len(frame))
    for metric, label, marker, color in metrics:
        quality_axis.scatter(frame[metric], y, s=75, marker=marker, color=color, label=label, zorder=3)
    quality_axis.set_yticks(y, frame["label"])
    quality_axis.invert_yaxis()
    quality_axis.set_xlim(0, 1.08)
    quality_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    quality_axis.set_xlabel("Success rate")
    quality_axis.set_title("Quality depends on the metric", fontweight="bold", loc="left")
    quality_axis.legend(frameon=False, loc="lower left")
    for index, row in frame.iterrows():
        quality_axis.text(1.01, index, f"n={row.attempted_n}", va="center", fontsize=10, color="#555555")
    fig.suptitle("Quantization trades 42% less storage for metric-specific quality changes", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.035, "Qwen3.8-27B · llama.cpp · 240 capability trials · 30 independent tasks · p50 evidence position", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.08, right=0.98, wspace=0.30)
    output_paths.append(_save(fig, output_dir / "presentation-size-quality.png"))

    equivalence = json.loads((results / "processed/q4-q8-equivalence.json").read_text(encoding="utf-8"))
    metric_order = ["answer_bearing", "format_valid", "end_to_end"]
    metric_labels = {"answer_bearing": "Answer-bearing", "format_valid": "Format-valid", "end_to_end": "End-to-end"}
    records = {record["metric"]: record for record in equivalence["metrics"]}
    fig, axis = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(metric_order))
    observed = [records[metric]["observed_difference"] for metric in metric_order]
    lows = [records[metric]["ci_low"] for metric in metric_order]
    highs = [records[metric]["ci_high"] for metric in metric_order]
    axis.axvspan(-0.1, 0.1, color="#009E73", alpha=0.10, label="±10 pp practical margin")
    axis.axvline(0, color="#555555", linewidth=1.2)
    axis.errorbar(
        observed,
        y,
        xerr=[np.array(observed) - np.array(lows), np.array(highs) - np.array(observed)],
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    axis.set_yticks(y, [metric_labels[metric] for metric in metric_order])
    axis.set_xlim(-0.22, 0.08)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Q4_K_M − Q8_0 difference")
    axis.set_title("")
    for index, metric in enumerate(metric_order):
        decision = records[metric]["decision"]
        axis.text(0.075, index, decision, va="center", ha="right", fontsize=11, fontweight="bold", color="#009E73" if decision == "equivalent" else "#D55E00")
    fig.suptitle("Q4/Q8 equivalence is metric-specific", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.925, "Matched pairs · 60 pairs · 95% paired bootstrap CI · practical margin ±10 percentage points", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.16, right=0.98)
    output_paths.append(_save(fig, output_dir / "presentation-equivalence-by-metric.png"))
    return output_paths


def _exp003_aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby(["task_type", "variant_condition_id", "target_context_tokens"], as_index=False)[["correct_n", "attempted_n"]]
        .sum()
    )
    grouped["success"] = grouped["correct_n"] / grouped["attempted_n"]
    return grouped


def plot_exp003_presentation(results_dir: str | Path) -> list[str]:
    """Create context-gap and matched-cell heatmap figures from fast-matched data."""

    _style()
    results = Path(results_dir)
    summary = pd.read_csv(results / "processed/fast-matched-summary.csv")
    grouped = _exp003_aggregate(summary)
    output_dir = results / "figures"
    output_paths: list[str] = []

    contexts = [8192, 32768]
    task_order = ["literal_retrieval", "semantic_retrieval", "multi_hop"]
    fig, axis = plt.subplots(figsize=(10, 5.6))
    for task in task_order:
        points = []
        for context in contexts:
            q8 = grouped.query("task_type == @task and variant_condition_id == 'q8_0' and target_context_tokens == @context")["success"].iloc[0]
            q4 = grouped.query("task_type == @task and variant_condition_id == 'q4_k_m' and target_context_tokens == @context")["success"].iloc[0]
            points.append(q8 - q4)
        axis.plot(
            ["8K", "32K"],
            points,
            marker="o",
            linewidth=2.8,
            markersize=8,
            color=PALETTE[task],
            label=TASK_LABELS[task],
        )
        axis.annotate(f"{points[-1]:+.0%}", ("32K", points[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=PALETTE[task], fontweight="bold")
    axis.axhspan(-0.10, 0.10, color="#009E73", alpha=0.10)
    axis.axhline(0, color="#555555", linewidth=1.1)
    axis.axhline(0.10, color="#999999", linestyle=":", linewidth=1)
    axis.axhline(-0.10, color="#999999", linestyle=":", linewidth=1)
    axis.set_ylim(-0.28, 0.25)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Input context")
    axis.set_ylabel("Q8_0 − Q4_K_M end-to-end gap")
    axis.set_title("")
    axis.legend(frameon=False, ncol=3, loc="lower center")
    fig.suptitle("The Q4 disadvantage grows only for literal retrieval", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.925, "Matched pilot · p50 evidence position · n=10 per task family/cell · descriptive, not a significance test", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.14, right=0.98)
    output_paths.append(_save(fig, output_dir / "presentation-context-gap.png"))

    columns = [("q8_0", 8192), ("q8_0", 32768), ("q4_k_m", 8192), ("q4_k_m", 32768)]
    column_labels = ["Q8 · 8K", "Q8 · 32K", "Q4 · 8K", "Q4 · 32K"]
    matrix = np.zeros((len(task_order), len(columns)))
    annotations: list[list[str]] = [["" for _ in columns] for _ in task_order]
    for row_index, task in enumerate(task_order):
        for column_index, (variant, context) in enumerate(columns):
            record = grouped.query("task_type == @task and variant_condition_id == @variant and target_context_tokens == @context").iloc[0]
            matrix[row_index, column_index] = record.success
            annotations[row_index][column_index] = f"{int(record.correct_n)}/{int(record.attempted_n)}"
    fig, axis = plt.subplots(figsize=(10, 5.3))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(columns)), column_labels)
    axis.set_yticks(range(len(task_order)), [TASK_LABELS[task] for task in task_order])
    axis.set_xlabel("Matched model and context")
    axis.set_ylabel("Task family")
    axis.set_title("")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, annotations[row_index][column_index], ha="center", va="center", fontsize=13, fontweight="bold", color="white" if matrix[row_index, column_index] > 0.55 else "#222222")
    fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03, format=PercentFormatter(1.0), label="Success")
    fig.suptitle("Observed quality varies more by task than by one universal context rule", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.925, "120 matched trials · all completed · values are end-to-end success counts", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.15, right=0.93)
    output_paths.append(_save(fig, output_dir / "presentation-context-task-heatmap.png"))
    return output_paths


def _exp004_pilot_frame(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["trajectory_length"] = frame["condition_id"].str.extract(r"traj(\d+)")[0].astype(int)
    return (
        frame.groupby(["variant_condition_id", "trajectory_length"], as_index=False)[["correct_n", "attempted_n", "failure_n"]]
        .sum()
        .assign(success=lambda value: value["correct_n"] / value["attempted_n"])
    )


def plot_exp004_presentation(results_dir: str | Path) -> list[str]:
    """Create protocol-diagnostic and fixed-policy recheck figures."""

    _style()
    results = Path(results_dir)
    pilot = _exp004_pilot_frame(pd.read_csv(results / "processed/fast-matched-summary.csv"))
    recheck = pd.read_csv(results / "processed/recheck-summary.csv")
    output_dir = results / "figures"
    output_paths: list[str] = []

    fig, (pilot_axis, recheck_axis) = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for variant in ["q8_0", "q4_k_m"]:
        old = pilot[pilot["variant_condition_id"] == variant].sort_values("trajectory_length")
        new = recheck[recheck["variant_condition_id"] == variant].sort_values("trajectory_length")
        color = PALETTE[variant]
        pilot_axis.plot(old["trajectory_length"], old["success"], "--o", color=color, linewidth=2.2, label=VARIANT_LABELS[variant])
        recheck_axis.plot(new["trajectory_length"], new["final_task_success"], "-o", color=color, linewidth=2.2, label=VARIANT_LABELS[variant])
    for axis in [pilot_axis, recheck_axis]:
        _percent_axis(axis)
        axis.set_xlabel("Trajectory length (observations)")
        axis.set_xticks([1, 4, 8, 16, 32])
    pilot_axis.set_ylabel("Final task success")
    pilot_axis.set_title("Earlier pilot", loc="left", fontweight="bold")
    recheck_axis.set_title("Fixed-policy recheck", loc="left", fontweight="bold")
    pilot_axis.text(0.02, 0.08, "n=10/cell\n30 invalid-output trials", transform=pilot_axis.transAxes, fontsize=11, color="#D55E00", bbox={"boxstyle": "round,pad=0.3", "facecolor": "#FFF3EE", "edgecolor": "none"})
    recheck_axis.text(0.02, 0.08, "n=30/cell\n0 invalid outputs", transform=recheck_axis.transAxes, fontsize=11, color="#009E73", bbox={"boxstyle": "round,pad=0.3", "facecolor": "#EEFAF4", "edgecolor": "none"})
    recheck_axis.legend(frameon=False, loc="lower right")
    fig.suptitle("Observed agent failures changed with the output protocol", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.035, "Descriptive comparison: the recheck changed the output budget, JSON policy, and retry policy; this is not a causal ablation", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.08, right=0.98, wspace=0.08)
    output_paths.append(_save(fig, output_dir / "presentation-agent-protocol-diagnosis.png"))

    fig, (success_axis, tokens_axis) = plt.subplots(1, 2, figsize=(13, 5.6))
    for variant in ["q8_0", "q4_k_m"]:
        frame = recheck[recheck["variant_condition_id"] == variant].sort_values("trajectory_length")
        color = PALETTE[variant]
        success_axis.plot(frame["trajectory_length"], frame["final_task_success"], "-o", color=color, linewidth=2.5, label=VARIANT_LABELS[variant])
        tokens_axis.plot(frame["trajectory_length"], frame["total_input_tokens"] / 1000, "-o", color=color, linewidth=2.5, label=VARIANT_LABELS[variant])
    _percent_axis(success_axis)
    success_axis.set_xticks([1, 4, 8, 16, 32])
    success_axis.set_xlabel("Trajectory length")
    success_axis.set_ylabel("Final task success")
    success_axis.set_title("Reliability stays flat", loc="left", fontweight="bold")
    success_axis.legend(frameon=False, loc="lower right")
    tokens_axis.set_xticks([1, 4, 8, 16, 32])
    tokens_axis.set_xlabel("Trajectory length")
    tokens_axis.set_ylabel("Recorded total input (k tokens)")
    tokens_axis.set_title("While the input grows", loc="left", fontweight="bold")
    fig.suptitle("Under the fixed policy, longer agent context did not reduce success", x=0.02, y=0.98, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.02, 0.035, "Q8_0/Q4_K_M · 10 independent tasks · 3 greedy repeats per cell · one critical position (50%)", fontsize=11, color="#555555")
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.08, right=0.98, wspace=0.25)
    output_paths.append(_save(fig, output_dir / "presentation-agent-recheck-stability.png"))
    return output_paths
