"""Re-score historical exp_002 JSONL without rerunning model inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_lab.analysis import (  # noqa: E402
    comparison_rows,
    render_report,
    rescore_trials,
    write_summary_csv,
)
from llm_lab.datasets import TaskCatalog  # noqa: E402
from llm_lab.evaluation import EvaluationTask, load_trial_results  # noqa: E402


EXPERIMENT_DIR = ROOT / "experiments/exp_002-quantization_llama_cpp_gguf"
DEFAULT_RAW = EXPERIMENT_DIR / "results/raw/trials.jsonl"
DEFAULT_CATALOG = ROOT / "data/tasks/core.v001.jsonl"
DEFAULT_SUMMARY = EXPERIMENT_DIR / "results/processed/rescored-summary.csv"
DEFAULT_REPORT = EXPERIMENT_DIR / "results/processed/rescoring-report.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostic-only calibrated re-scoring of existing exp_002 raw trials"
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    raw_path = args.raw.resolve()
    catalog_path = args.catalog.resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw input does not exist: {raw_path}")
    if not catalog_path.is_file():
        raise FileNotFoundError(f"task catalog does not exist: {catalog_path}")

    trials = load_trial_results(raw_path)
    if not trials:
        raise ValueError(f"raw input is empty: {raw_path}")
    catalog = TaskCatalog.from_jsonl(catalog_path)
    tasks = {
        definition.task_id: EvaluationTask.from_definition(definition, context="")
        for definition in catalog.tasks
    }
    rescored = rescore_trials(trials, tasks)
    rows = comparison_rows(rescored)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(args.summary, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_report(
            raw_path=raw_path,
            task_catalog_path=catalog_path,
            raw_trial_n=len(trials),
            rows=rows,
        ),
        encoding="utf-8",
    )
    print(f"rescored {len(trials)} trials into {args.summary} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
