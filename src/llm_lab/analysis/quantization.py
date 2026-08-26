"""Notebook-facing joins and recommendation rules for exp_002."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping

from llm_lab.quantization import QuantizationManifest, QuantizationVariant


class QuantizationAnalysisError(ValueError):
    """Raised when a quantization comparison is incomplete or unmeasured."""


REQUIRED_TRADEOFF_METRICS = (
    "median_stream_ttft_s",
    "median_prompt_throughput_proxy_tok_s",
    "median_post_first_chunk_output_tok_s",
    "median_peak_memory_bytes",
)


def tradeoff_rows(
    summaries: Iterable[Mapping[str, Any]],
    manifest: QuantizationManifest | Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join aggregated trial metrics to resolved artifact provenance.

    Scored accuracy is recomputed from task-level counts so summary rows are
    weighted by their number of scored trials. Runtime and invalid-output
    failures remain in the attempted denominator and are reported separately.
    """

    variants = _manifest_variants(manifest)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        condition_id = row.get("condition_id")
        if condition_id is not None:
            grouped[str(condition_id)].append(row)

    rows: list[dict[str, Any]] = []
    for variant in variants:
        condition_rows = grouped.get(variant.condition_id)
        if not condition_rows:
            raise QuantizationAnalysisError(
                f"missing measured summaries for {variant.condition_id}"
            )
        attempted_n, scored_n, correct_n, failure_n = _weighted_outcomes(
            condition_rows,
            variant.condition_id,
        )
        if attempted_n == 0:
            raise QuantizationAnalysisError(
                f"{variant.condition_id} has no attempted measurements"
            )
        scored_accuracy = correct_n / scored_n if scored_n else None
        metrics = {
            key: _required_median(condition_rows, key, variant.condition_id)
            for key in REQUIRED_TRADEOFF_METRICS
        }
        rows.append(
            {
                "condition_id": variant.condition_id,
                "label": variant.label,
                "format": variant.format,
                "quantization_type": variant.quantization_type,
                "bits": variant.bits,
                "runtime_kernel": variant.runtime_kernel,
                "artifact_uri": variant.artifact.artifact_uri,
                "artifact_sha256": variant.artifact.artifact_sha256,
                "artifact_size_bytes": variant.artifact.artifact_size_bytes,
                "attempted_n": attempted_n,
                "scored_n": scored_n,
                "correct_n": correct_n,
                "failure_n": failure_n,
                "scored_accuracy": scored_accuracy,
                "end_to_end_success": correct_n / attempted_n,
                "failure_rate": failure_n / attempted_n,
                "accuracy": scored_accuracy,
                **metrics,
            }
        )
    return rows


def recommend_baseline(
    rows: Iterable[Mapping[str, Any]],
    *,
    accuracy_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Choose the smallest artifact near the best end-to-end success."""

    if not 0.0 <= accuracy_tolerance < 1.0:
        raise ValueError("accuracy_tolerance must be between 0 and 1")
    candidates = [dict(row) for row in rows]
    if not candidates:
        raise QuantizationAnalysisError("cannot recommend from empty measurements")
    for row in candidates:
        for field in (
            "end_to_end_success",
            "artifact_size_bytes",
            "median_peak_memory_bytes",
        ):
            if not _is_number(row.get(field)):
                raise QuantizationAnalysisError(
                    f"{row.get('condition_id', 'unknown')} is missing measured {field}"
                )
        if not isinstance(row.get("scored_n"), int) or row["scored_n"] < 1:
            raise QuantizationAnalysisError(
                f"{row.get('condition_id', 'unknown')} has no scored measurements"
            )
    best_end_to_end_success = max(
        float(row["end_to_end_success"]) for row in candidates
    )
    floor = best_end_to_end_success - accuracy_tolerance
    eligible = [
        row
        for row in candidates
        if float(row["end_to_end_success"]) >= floor
    ]
    if not eligible:
        raise QuantizationAnalysisError("no measured condition meets accuracy tolerance")
    selected = min(
        eligible,
        key=lambda row: (
            int(row["artifact_size_bytes"]),
            int(row["median_peak_memory_bytes"]),
            -float(row.get("median_post_first_chunk_output_tok_s", 0.0)),
        ),
    )
    selected["best_end_to_end_success"] = best_end_to_end_success
    selected["accuracy_tolerance"] = accuracy_tolerance
    selected["minimum_eligible_end_to_end_success"] = floor
    return selected


def _manifest_variants(
    manifest: QuantizationManifest | Mapping[str, Any],
) -> tuple[QuantizationVariant, ...]:
    if isinstance(manifest, QuantizationManifest):
        return manifest.variants
    return QuantizationManifest.from_record(manifest).variants


def _weighted_outcomes(
    rows: Iterable[Mapping[str, Any]], condition_id: str
) -> tuple[int, int, float, int]:
    attempted_n = 0
    scored_n = 0
    correct_n = 0.0
    failure_n = 0
    for row in rows:
        count = row.get("scored_n", 0)
        if not isinstance(count, int) or count < 0:
            raise QuantizationAnalysisError(
                f"{condition_id} has invalid scored_n: {count!r}"
            )
        attempted = row.get("attempted_n", row.get("n", count))
        if not isinstance(attempted, int) or attempted < count:
            raise QuantizationAnalysisError(
                f"{condition_id} has invalid attempted_n: {attempted!r}"
            )
        explicit_correct = row.get("correct_n")
        if explicit_correct is None:
            accuracy = row.get("accuracy", row.get("scored_accuracy"))
            if count and (
                not _is_number(accuracy) or not 0.0 <= float(accuracy) <= 1.0
            ):
                raise QuantizationAnalysisError(
                    f"{condition_id} requires accuracy when scored_n is positive"
                )
            successes = float(accuracy or 0.0) * count
        else:
            if (
                not _is_number(explicit_correct)
                or not 0.0 <= float(explicit_correct) <= count
            ):
                raise QuantizationAnalysisError(
                    f"{condition_id} has invalid correct_n: {explicit_correct!r}"
                )
            successes = float(explicit_correct)
        failures = row.get("failure_n", row.get("error_n", attempted - count))
        if not isinstance(failures, int) or not 0 <= failures <= attempted - count:
            raise QuantizationAnalysisError(
                f"{condition_id} has invalid failure_n: {failures!r}"
            )
        attempted_n += attempted
        scored_n += count
        correct_n += successes
        failure_n += failures
    return attempted_n, scored_n, correct_n, failure_n


def _required_median(
    rows: Iterable[Mapping[str, Any]], key: str, condition_id: str
) -> float:
    values = [float(row[key]) for row in rows if _is_number(row.get(key))]
    if not values:
        raise QuantizationAnalysisError(
            f"{condition_id} is missing measured {key}"
        )
    return float(median(values))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
