"""Notebook-facing joins and recommendation rules for exp_002."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping

from llm_lab.quantization import QuantizationManifest, QuantizationVariant


class QuantizationAnalysisError(ValueError):
    """Raised when a quantization comparison is incomplete or unmeasured."""


REQUIRED_TRADEOFF_METRICS = (
    "median_ttft_s",
    "median_prefill_tokens_per_second",
    "median_decode_tokens_per_second",
    "median_peak_memory_bytes",
)


def tradeoff_rows(
    summaries: Iterable[Mapping[str, Any]],
    manifest: QuantizationManifest | Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join aggregated trial metrics to resolved artifact provenance.

    Accuracy is recomputed from ``accuracy * scored_n`` so task-level summary
    rows are weighted by their number of scored trials. Runtime failures remain
    represented by the summary counts but do not become accuracy observations.
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
        scored_n, successes = _weighted_accuracy(condition_rows, variant.condition_id)
        if scored_n == 0:
            raise QuantizationAnalysisError(
                f"{variant.condition_id} has no scored accuracy measurements"
            )
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
                "scored_n": scored_n,
                "accuracy": successes / scored_n,
                **metrics,
            }
        )
    return rows


def recommend_baseline(
    rows: Iterable[Mapping[str, Any]],
    *,
    accuracy_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Choose the smallest measured artifact near the best measured accuracy."""

    if not 0.0 <= accuracy_tolerance < 1.0:
        raise ValueError("accuracy_tolerance must be between 0 and 1")
    candidates = [dict(row) for row in rows]
    if not candidates:
        raise QuantizationAnalysisError("cannot recommend from empty measurements")
    for row in candidates:
        for field in ("accuracy", "artifact_size_bytes", "median_peak_memory_bytes"):
            if not _is_number(row.get(field)):
                raise QuantizationAnalysisError(
                    f"{row.get('condition_id', 'unknown')} is missing measured {field}"
                )
    best_accuracy = max(float(row["accuracy"]) for row in candidates)
    floor = best_accuracy - accuracy_tolerance
    eligible = [row for row in candidates if float(row["accuracy"]) >= floor]
    if not eligible:
        raise QuantizationAnalysisError("no measured condition meets accuracy tolerance")
    selected = min(
        eligible,
        key=lambda row: (
            int(row["artifact_size_bytes"]),
            int(row["median_peak_memory_bytes"]),
            -float(row.get("median_decode_tokens_per_second", 0.0)),
        ),
    )
    selected["best_accuracy"] = best_accuracy
    selected["accuracy_tolerance"] = accuracy_tolerance
    selected["minimum_eligible_accuracy"] = floor
    return selected


def _manifest_variants(
    manifest: QuantizationManifest | Mapping[str, Any],
) -> tuple[QuantizationVariant, ...]:
    if isinstance(manifest, QuantizationManifest):
        return manifest.variants
    return QuantizationManifest.from_record(manifest).variants


def _weighted_accuracy(
    rows: Iterable[Mapping[str, Any]], condition_id: str
) -> tuple[int, float]:
    scored_n = 0
    successes = 0.0
    for row in rows:
        count = row.get("scored_n", 0)
        accuracy = row.get("accuracy")
        if not isinstance(count, int) or count < 0:
            raise QuantizationAnalysisError(
                f"{condition_id} has invalid scored_n: {count!r}"
            )
        if count == 0:
            continue
        if not _is_number(accuracy) or not 0.0 <= float(accuracy) <= 1.0:
            raise QuantizationAnalysisError(
                f"{condition_id} requires accuracy when scored_n is positive"
            )
        scored_n += count
        successes += float(accuracy) * count
    return scored_n, successes


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
