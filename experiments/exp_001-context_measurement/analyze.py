"""Regenerate exp_001 processed outputs from a verified run manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_ROOT = Path(__file__).resolve().parent
ROOT = EXPERIMENT_ROOT.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_lab.analysis import (  # noqa: E402
    aggregate_jsonl,
    effective_context_by_task,
    effective_context_by_task_and_position,
    missing_context_cells,
    position_gap_rows,
    write_summary_csv,
)
from llm_lab.evaluation import load_trial_results  # noqa: E402


class AnalysisInputError(ValueError):
    """Raised when a run manifest or its raw results are not analysis-ready."""


def regenerate(
    manifest_path: str | Path,
    *,
    raw_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    position_gap_path: str | Path | None = None,
    effective_context_path: str | Path | None = None,
    effective_context_by_position_path: str | Path | None = None,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Validate a run and regenerate all tabular effective-context outputs.

    The default is measured-data-only. ``allow_fixture`` exists solely for
    checking the harness regeneration path and must not be used for findings.
    All input validation completes before any output is written.
    """

    manifest_file = Path(manifest_path).resolve()
    manifest = _load_manifest(manifest_file)
    backend = manifest.get("backend")
    if backend == "fixture" and not allow_fixture:
        raise AnalysisInputError(
            "fixture results are harness-only; pass allow_fixture=True only for validation"
        )
    if backend != "fixture" and not isinstance(backend, str):
        raise AnalysisInputError("manifest backend must identify the execution backend")

    raw_file = (
        Path(raw_path).resolve()
        if raw_path is not None
        else _resolve_manifest_path(manifest_file, manifest.get("raw_results"))
    )
    if not raw_file.is_file():
        raise FileNotFoundError(f"raw results are required: {raw_file}")
    expected_hash = manifest.get("raw_results_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise AnalysisInputError("manifest must include raw_results_sha256")
    actual_hash = _sha256(raw_file)
    if actual_hash != expected_hash:
        raise AnalysisInputError(
            "raw results SHA-256 does not match the run manifest: "
            f"expected {expected_hash}, found {actual_hash}"
        )

    scorer_version = manifest.get("scorer_version")
    if not isinstance(scorer_version, str) or not scorer_version.strip():
        raise AnalysisInputError("manifest must include scorer_version")
    trials = load_trial_results(raw_file)
    summaries = aggregate_jsonl(raw_file, expected_scorer=scorer_version)
    context_lengths, evidence_positions, task_types = _manifest_dimensions(manifest)
    missing = missing_context_cells(
        summaries,
        context_lengths=context_lengths,
        evidence_positions=evidence_positions,
        task_types=task_types,
    )
    if missing:
        raise AnalysisInputError(f"missing planned cells: {missing[:10]}")

    outputs = _output_paths(
        manifest_file,
        summary_path=summary_path,
        position_gap_path=position_gap_path,
        effective_context_path=effective_context_path,
        effective_context_by_position_path=effective_context_by_position_path,
    )
    gaps = position_gap_rows(summaries)
    effective_options = manifest.get("effective_context", {})
    if not isinstance(effective_options, Mapping):
        raise AnalysisInputError("manifest effective_context must be an object")
    baseline_length = int(effective_options.get("baseline_length", 8192))
    baseline_gate = float(effective_options.get("baseline_accuracy_gate", 0.80))
    alpha = float(effective_options.get("alpha", 0.90))
    effective = effective_context_by_task(
        summaries,
        baseline_context_tokens=baseline_length,
        alpha=alpha,
        minimum_baseline_accuracy=baseline_gate,
    )
    effective_by_position = effective_context_by_task_and_position(
        summaries,
        baseline_context_tokens=baseline_length,
        alpha=alpha,
        minimum_baseline_accuracy=baseline_gate,
    )

    write_summary_csv(outputs["summary"], summaries)
    _write_rows_csv(outputs["position_gap"], gaps)
    _write_json(outputs["effective_context"], effective)
    _write_json(outputs["effective_context_by_position"], effective_by_position)
    return {
        "backend": backend,
        "phase": manifest.get("phase"),
        "raw_results": str(raw_file),
        "raw_results_sha256": actual_hash,
        "trial_n": len(trials),
        "summary_row_n": len(summaries),
        "position_gap_row_n": len(gaps),
        "baseline_limited_task_types": [
            row["task_type"] for row in effective if row["status"] == "baseline_limited"
        ],
        "excluded_cell_n": len(
            [row for row in manifest.get("coverage", []) if row.get("status") == "excluded"]
        ),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise AnalysisInputError(f"invalid manifest JSON: {path}") from error
    if not isinstance(value, dict):
        raise AnalysisInputError("run manifest must be a JSON object")
    if value.get("schema_version") != 1:
        raise AnalysisInputError("unsupported run manifest schema version")
    return value


def _manifest_dimensions(
    manifest: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[str, ...]]:
    dimensions = (
        manifest.get("context_lengths"),
        manifest.get("evidence_positions"),
        manifest.get("task_types"),
    )
    if any(value is None for value in dimensions):
        coverage = manifest.get("coverage")
        if not isinstance(coverage, list):
            raise AnalysisInputError(
                "manifest must declare dimensions or a coverage list"
            )
        try:
            dimensions = (
                sorted({int(row["target_context_tokens"]) for row in coverage}),
                sorted({float(row["requested_evidence_position"]) for row in coverage}),
                sorted({str(row["task_type"]) for row in coverage}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisInputError(
                "manifest coverage must declare context, position, and task dimensions"
            ) from error
    try:
        lengths = tuple(int(value) for value in dimensions[0])
        positions = tuple(float(value) for value in dimensions[1])
        task_types = tuple(str(value) for value in dimensions[2])
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisInputError(
            "manifest must declare context_lengths, evidence_positions, and task_types"
        ) from error
    if not lengths or not positions or not task_types:
        raise AnalysisInputError("manifest dimensions must not be empty")
    return lengths, positions, task_types


def _resolve_manifest_path(manifest_file: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisInputError("manifest must include raw_results")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (ROOT, manifest_file.parent, EXPERIMENT_ROOT):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (ROOT / candidate).resolve()


def _output_paths(
    manifest_file: Path,
    *,
    summary_path: str | Path | None,
    position_gap_path: str | Path | None,
    effective_context_path: str | Path | None,
    effective_context_by_position_path: str | Path | None,
) -> dict[str, Path]:
    results_root = manifest_file.parent.parent
    values = {
        "summary": summary_path or results_root / "processed/summary.csv",
        "position_gap": position_gap_path or results_root / "processed/position-gap.csv",
        "effective_context": effective_context_path
        or results_root / "processed/effective-context.json",
        "effective_context_by_position": effective_context_by_position_path
        or results_root / "processed/effective-context-by-position.json",
    }
    return {key: Path(value).resolve() for key, value in values.items()}


def _write_rows_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = []
    for row in rows:
        item = dict(row)
        if isinstance(item.get("edge_positions"), list):
            item["edge_positions"] = json.dumps(item["edge_positions"])
        materialized.append(item)
    if not materialized:
        raise AnalysisInputError(f"cannot write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args(argv)
    result = regenerate(args.manifest, raw_path=args.raw, allow_fixture=args.allow_fixture)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
