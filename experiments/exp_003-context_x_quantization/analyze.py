"""Regenerate verified exp_003 interaction-analysis outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse


EXPERIMENT_ROOT = Path(__file__).resolve().parent
ROOT = EXPERIMENT_ROOT.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_lab.analysis import (  # noqa: E402
    aggregate_jsonl,
    effective_context_by_variant_and_task,
    interaction_report,
    matched_cell_rows,
    relative_degradation_rows,
    write_summary_csv,
)
from llm_lab.datasets import TaskCatalog  # noqa: E402
from llm_lab.evaluation import load_trial_results  # noqa: E402


SCORER_VERSION = "calibrated.v1"
EXPERIMENT_ID = "exp_003"


class AnalysisInputError(ValueError):
    """Raised when measured exp_003 inputs are incomplete or inconsistent."""


def regenerate(
    manifest_path: str | Path,
    *,
    raw_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    degradation_path: str | Path | None = None,
    interaction_path: str | Path | None = None,
    effective_context_path: str | Path | None = None,
    allow_fixture: bool = False,
) -> dict[str, Any]:
    """Validate a run and regenerate all interaction-analysis outputs."""

    manifest_file = Path(manifest_path).resolve()
    manifest = _load_json(manifest_file, "run manifest")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise AnalysisInputError("run manifest must describe exp_003")
    backend = manifest.get("backend")
    if backend == "fixture" and not allow_fixture:
        raise AnalysisInputError(
            "fixture results are harness-only; pass allow_fixture=True only for validation"
        )
    if not isinstance(backend, str) or not backend.strip():
        raise AnalysisInputError("manifest backend must identify the execution backend")
    if manifest.get("scorer_version") != SCORER_VERSION:
        raise AnalysisInputError(
            f"exp_003 requires scorer version {SCORER_VERSION!r}"
        )

    source_manifest_file = _resolve_path(
        manifest_file, manifest.get("source_manifest"), "source manifest"
    )
    _verify_hash(
        source_manifest_file,
        manifest.get("source_manifest_sha256"),
        "source manifest SHA-256",
    )
    source_manifest = _load_json(source_manifest_file, "source manifest")
    source_controls = source_manifest.get("controls")
    if not isinstance(source_controls, Mapping):
        raise AnalysisInputError("source manifest controls must be an object")
    if source_controls.get("scorer_version") != SCORER_VERSION:
        raise AnalysisInputError(
            f"source manifest must use scorer version {SCORER_VERSION!r}"
        )

    raw_file = (
        Path(raw_path).resolve()
        if raw_path is not None
        else _resolve_path(manifest_file, manifest.get("raw_results"), "raw results")
    )
    _verify_hash(raw_file, manifest.get("raw_results_sha256"), "raw results SHA-256")

    catalog_file = _catalog_path(manifest, source_controls)
    expected_catalog_sha = manifest.get(
        "task_catalog_sha256", source_controls.get("task_catalog_sha256")
    )
    _verify_hash(catalog_file, expected_catalog_sha, "task catalog SHA-256")
    catalog = TaskCatalog.from_jsonl(catalog_file)

    task_ids = _required_strings(manifest, "task_ids")
    unknown_task_ids = set(task_ids) - set(catalog.ids)
    if unknown_task_ids:
        raise AnalysisInputError(
            f"task IDs are missing from the task catalog: {sorted(unknown_task_ids)}"
        )
    variants = _variant_ids(manifest)
    context_lengths = _required_ints(manifest, "context_lengths")
    evidence_positions = _required_floats(manifest, "evidence_positions")
    task_types = _required_strings(manifest, "task_types")
    repeats = _positive_int(manifest, "repeats")

    trials = load_trial_results(raw_file)
    summaries = aggregate_jsonl(
        raw_file,
        expected_scorer=SCORER_VERSION,
        group_by_task=True,
    )
    _validate_coverage(
        summaries,
        manifest.get("coverage"),
        task_ids=task_ids,
        task_types=task_types,
        variants=variants,
        context_lengths=context_lengths,
        evidence_positions=evidence_positions,
        repeats=repeats,
        catalog=catalog,
    )
    matched = matched_cell_rows(
        summaries,
        variant_ids=variants,
        context_lengths=context_lengths,
        evidence_positions=evidence_positions,
        task_types=task_types,
        task_ids=task_ids,
    )
    degradation = relative_degradation_rows(
        matched,
        baseline_context_tokens=min(context_lengths),
    )
    analysis_options = manifest.get("analysis", {})
    if not isinstance(analysis_options, Mapping):
        raise AnalysisInputError("manifest analysis must be an object")
    reference_variant = str(
        analysis_options.get("primary_gap_reference", variants[0])
    )
    if reference_variant not in variants:
        raise AnalysisInputError(
            f"primary gap reference is not selected: {reference_variant}"
        )
    tolerance = float(analysis_options.get("approx_constant_gap_tolerance", 0.10))
    reports = interaction_report(
        matched,
        reference_variant=reference_variant,
        approx_constant_gap_tolerance=tolerance,
    )

    effective_options = manifest.get("effective_context", {})
    if not isinstance(effective_options, Mapping):
        raise AnalysisInputError("manifest effective_context must be an object")
    effective = effective_context_by_variant_and_task(
        matched,
        baseline_context_tokens=int(
            effective_options.get("baseline_length", min(context_lengths))
        ),
        alpha=float(effective_options.get("alpha", 0.90)),
        minimum_baseline_accuracy=float(
            effective_options.get("baseline_accuracy_gate", 0.80)
        ),
    )

    outputs = _output_paths(
        manifest_file,
        summary_path=summary_path,
        degradation_path=degradation_path,
        interaction_path=interaction_path,
        effective_context_path=effective_context_path,
    )
    write_summary_csv(outputs["summary"], summaries)
    _write_rows_csv(outputs["degradation"], degradation)
    _write_json(outputs["interaction"], reports)
    _write_json(outputs["effective_context"], effective)
    return {
        "experiment_id": EXPERIMENT_ID,
        "phase": manifest.get("phase"),
        "backend": backend,
        "raw_results": str(raw_file),
        "raw_results_sha256": manifest["raw_results_sha256"],
        "trial_n": len(trials),
        "summary_row_n": len(summaries),
        "summary_rows": summaries,
        "matched_rows": matched,
        "degradation_rows": degradation,
        "interaction_reports": reports,
        "effective_context": effective,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def _validate_coverage(
    summaries: Iterable[Mapping[str, Any]],
    coverage_value: Any,
    *,
    task_ids: tuple[str, ...],
    task_types: tuple[str, ...],
    variants: tuple[str, ...],
    context_lengths: tuple[int, ...],
    evidence_positions: tuple[float, ...],
    repeats: int,
    catalog: TaskCatalog,
) -> None:
    if not isinstance(coverage_value, list) or not coverage_value:
        raise AnalysisInputError("manifest must include a non-empty coverage list")
    expected_keys = {
        (variant, task_type, context, position)
        for variant in variants
        for task_type in task_types
        for context in context_lengths
        for position in evidence_positions
    }
    coverage_by_key: dict[tuple[str, str, int, float], Mapping[str, Any]] = {}
    for item in coverage_value:
        if not isinstance(item, Mapping):
            raise AnalysisInputError("manifest coverage entries must be objects")
        key = _coverage_key(item)
        if key in coverage_by_key:
            raise AnalysisInputError(f"duplicate coverage cell: {key}")
        coverage_by_key[key] = item
    if set(coverage_by_key) != expected_keys:
        raise AnalysisInputError("manifest coverage does not match declared dimensions")

    summary_groups: dict[tuple[str, str, int, float], dict[str, int]] = defaultdict(
        lambda: {"attempted_n": 0, "scored_n": 0}
    )
    for row in summaries:
        key = _coverage_key(row)
        group = summary_groups[key]
        group["attempted_n"] += int(row.get("attempted_n", row.get("n", 0)))
        group["scored_n"] += int(row.get("scored_n", 0))

    task_type_by_id = {
        task.task_id: task.task_type
        for task in catalog.tasks
        if task.task_id in task_ids
    }
    for key, coverage in coverage_by_key.items():
        if coverage.get("status") != "valid":
            raise AnalysisInputError(
                "exp_003 has excluded cells; complete matched-cell analysis is unavailable"
            )
        independent_n = sum(
            task_type_by_id.get(task_id) == key[1] for task_id in task_ids
        )
        expected_trial_n = independent_n * repeats
        if coverage.get("expected_trial_n") != expected_trial_n:
            raise AnalysisInputError(f"coverage expected trial count mismatch: {key}")
        if coverage.get("trial_n") != expected_trial_n or coverage.get("scored_n") != expected_trial_n:
            raise AnalysisInputError(f"coverage cell is incomplete: {key}")
        actual = summary_groups.get(key)
        if actual is None or actual["attempted_n"] != expected_trial_n:
            raise AnalysisInputError(f"summary is missing coverage cell: {key}")
        if actual["scored_n"] != expected_trial_n:
            raise AnalysisInputError(f"summary coverage is not fully scored: {key}")


def _coverage_key(row: Mapping[str, Any]) -> tuple[str, str, int, float]:
    try:
        return (
            str(row["variant_condition_id"]),
            str(row["task_type"]),
            int(row["target_context_tokens"]),
            float(row["requested_evidence_position"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisInputError(
            "coverage and summary rows must declare variant, task, context, and position"
        ) from error


def _variant_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    value = manifest.get("quantization_variants")
    if not isinstance(value, list) or not value:
        raise AnalysisInputError("manifest quantization_variants must be non-empty")
    try:
        ids = tuple(str(item["condition_id"]) for item in value)
    except (KeyError, TypeError) as error:
        raise AnalysisInputError("manifest variants must include condition_id") from error
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise AnalysisInputError("manifest variant IDs must be unique and non-empty")
    return ids


def _required_strings(manifest: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = manifest.get(field)
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise AnalysisInputError(f"manifest {field} must be a non-empty string list")
    return tuple(value)


def _required_ints(manifest: Mapping[str, Any], field: str) -> tuple[int, ...]:
    value = manifest.get(field)
    if not isinstance(value, list) or not value:
        raise AnalysisInputError(f"manifest {field} must be a non-empty list")
    try:
        values = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise AnalysisInputError(f"manifest {field} must contain integers") from error
    if tuple(sorted(set(values))) != values or any(item < 1 for item in values):
        raise AnalysisInputError(f"manifest {field} must be strictly increasing")
    return values


def _required_floats(manifest: Mapping[str, Any], field: str) -> tuple[float, ...]:
    value = manifest.get(field)
    if not isinstance(value, list) or not value:
        raise AnalysisInputError(f"manifest {field} must be a non-empty list")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise AnalysisInputError(f"manifest {field} must contain numbers") from error
    if tuple(sorted(set(values))) != values or any(
        not 0.0 <= item <= 1.0 for item in values
    ):
        raise AnalysisInputError(f"manifest {field} must be ordered percentages")
    return values


def _positive_int(manifest: Mapping[str, Any], field: str) -> int:
    value = manifest.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AnalysisInputError(f"manifest {field} must be a positive integer")
    return value


def _catalog_path(
    manifest: Mapping[str, Any],
    source_controls: Mapping[str, Any],
) -> Path:
    value = manifest.get("task_catalog", source_controls.get("task_catalog"))
    if not isinstance(value, str) or not value.strip():
        raise AnalysisInputError("manifest must include task_catalog")
    parsed = urlparse(value)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise AnalysisInputError(f"unsupported task catalog URI host: {parsed.netloc}")
        return Path(unquote(parsed.path)).resolve()
    if parsed.scheme:
        raise AnalysisInputError("task catalog must be a local path or file URI")
    return (ROOT / Path(unquote(value))).resolve()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise AnalysisInputError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise AnalysisInputError(f"{description} must be a JSON object")
    return value


def _resolve_path(manifest_file: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisInputError(f"manifest must include {description}")
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (ROOT, manifest_file.parent, EXPERIMENT_ROOT):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (ROOT / candidate).resolve()


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    if not isinstance(expected, str) or not expected.strip():
        raise AnalysisInputError(f"manifest must include {label}")
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise AnalysisInputError(
            f"{label} does not match the run manifest: expected {expected}, found {actual}"
        )


def _output_paths(
    manifest_file: Path,
    *,
    summary_path: str | Path | None,
    degradation_path: str | Path | None,
    interaction_path: str | Path | None,
    effective_context_path: str | Path | None,
) -> dict[str, Path]:
    results_root = manifest_file.parent.parent
    values = {
        "summary": summary_path or results_root / "processed/summary.csv",
        "degradation": degradation_path
        or results_root / "processed/relative-degradation.csv",
        "interaction": interaction_path or results_root / "processed/interaction.json",
        "effective_context": effective_context_path
        or results_root / "processed/effective-context.json",
    }
    return {key: Path(value).resolve() for key, value in values.items()}


def _write_rows_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = []
    for row in rows:
        item = dict(row)
        for key, value in item.items():
            if isinstance(value, (dict, list)):
                item[key] = json.dumps(value, sort_keys=True)
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
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"summary_rows", "effective_context", "interaction_reports"}},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
