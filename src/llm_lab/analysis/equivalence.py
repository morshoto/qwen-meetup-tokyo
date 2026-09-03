"""Paired practical-equivalence analysis for matched quantization trials."""

from __future__ import annotations

import random
from collections import defaultdict
from math import isfinite
from typing import Any, Iterable, Mapping

from llm_lab.evaluation import TrialResult, TrialStatus


class EquivalenceAnalysisError(ValueError):
    """Raised when a paired equivalence comparison is not analysis-ready."""


DEFAULT_METRICS = {
    "end_to_end": "correct",
    "exact": "exact_correct",
    "answer_bearing": "answer_bearing_correct",
    "format_valid": "format_valid",
}


def paired_equivalence_report(
    trials: Iterable[TrialResult | Mapping[str, Any]],
    *,
    reference_variant: str,
    candidate_variant: str,
    margin: float = 0.10,
    confidence: float = 0.95,
    bootstrap_repeats: int = 20_000,
    seed: int = 42,
    metric_fields: Mapping[str, str] | None = None,
    pair_fields: Iterable[str] = (
        "task_id",
        "target_context_tokens",
        "requested_evidence_position",
    ),
) -> dict[str, Any]:
    """Compare two variants using matched task/context pairs.

    The returned decision is *practical* equivalence: the two-sided bootstrap
    confidence interval for candidate minus reference must lie completely
    inside ``[-margin, margin]``. This is intentionally distinct from exact
    equality and fails closed on missing, duplicate, or non-completed trials.
    """

    if not reference_variant.strip() or not candidate_variant.strip():
        raise EquivalenceAnalysisError("variant labels must be non-empty")
    if reference_variant == candidate_variant:
        raise EquivalenceAnalysisError("reference and candidate variants must differ")
    if not isfinite(float(margin)) or not 0.0 < float(margin) < 1.0:
        raise EquivalenceAnalysisError("margin must be finite and between zero and one")
    if not isfinite(float(confidence)) or not 0.0 < float(confidence) < 1.0:
        raise EquivalenceAnalysisError("confidence must be finite and between zero and one")
    if isinstance(bootstrap_repeats, bool) or bootstrap_repeats < 1:
        raise EquivalenceAnalysisError("bootstrap_repeats must be a positive integer")
    selected_pair_fields = tuple(str(field) for field in pair_fields)
    if not selected_pair_fields:
        raise EquivalenceAnalysisError("at least one pair field is required")
    metrics = dict(metric_fields or DEFAULT_METRICS)
    if not metrics:
        raise EquivalenceAnalysisError("at least one metric is required")

    parsed = [
        value if isinstance(value, TrialResult) else TrialResult.from_record(value)
        for value in trials
    ]
    if not parsed:
        raise EquivalenceAnalysisError("cannot compare an empty trial set")
    pairs = _matched_pairs(
        parsed,
        reference_variant=reference_variant,
        candidate_variant=candidate_variant,
        pair_fields=selected_pair_fields,
    )
    metric_reports = []
    for metric_name, score_field in metrics.items():
        differences = [
            _binary_value(candidate.score, score_field)
            - _binary_value(reference.score, score_field)
            for reference, candidate in pairs
        ]
        observed = sum(differences) / len(differences)
        ci_low, ci_high = _paired_bootstrap_interval(
            differences,
            confidence=float(confidence),
            repeats=int(bootstrap_repeats),
            seed=int(seed),
        )
        equivalent = ci_low >= -float(margin) and ci_high <= float(margin)
        if equivalent:
            decision = "equivalent"
        elif ci_high < -float(margin) or ci_low > float(margin):
            decision = "not_equivalent"
        else:
            decision = "inconclusive"
        metric_reports.append(
            {
                "metric": str(metric_name),
                "score_field": str(score_field),
                "reference_variant": reference_variant,
                "candidate_variant": candidate_variant,
                "pair_n": len(differences),
                "reference_success_n": sum(
                    _binary_value(reference.score, score_field) for reference, _ in pairs
                ),
                "candidate_success_n": sum(
                    _binary_value(candidate.score, score_field) for _, candidate in pairs
                ),
                "observed_difference": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "confidence": float(confidence),
                "equivalence_margin": float(margin),
                "bootstrap_repeats": int(bootstrap_repeats),
                "bootstrap_seed": int(seed),
                "decision": decision,
                "discordant_pair_n": sum(difference != 0 for difference in differences),
            }
        )
    return {
        "schema_version": 1,
        "analysis": "paired_practical_equivalence",
        "reference_variant": reference_variant,
        "candidate_variant": candidate_variant,
        "pair_fields": list(selected_pair_fields),
        "pair_n": len(pairs),
        "equivalence_margin": float(margin),
        "confidence": float(confidence),
        "bootstrap_repeats": int(bootstrap_repeats),
        "bootstrap_seed": int(seed),
        "metrics": metric_reports,
        "complete_quality_equivalent": all(
            report["decision"] == "equivalent" for report in metric_reports
        ),
        "interpretation": (
            "Practical equivalence is established only for metrics whose full "
            "confidence interval is inside the pre-registered margin; this does "
            "not prove exact equality or generalize beyond the matched catalog."
        ),
    }


def _matched_pairs(
    trials: Iterable[TrialResult],
    *,
    reference_variant: str,
    candidate_variant: str,
    pair_fields: tuple[str, ...],
) -> list[tuple[TrialResult, TrialResult]]:
    by_variant: dict[str, dict[tuple[Any, ...], TrialResult]] = defaultdict(dict)
    for trial in trials:
        variant = _variant_label(trial)
        if variant not in {reference_variant, candidate_variant}:
            continue
        if trial.status != TrialStatus.COMPLETED:
            raise EquivalenceAnalysisError(
                "quality equivalence requires completed trials; found "
                f"{trial.status.value} in {trial.trial_id}"
            )
        key = tuple(_pair_value(trial, field) for field in pair_fields)
        if any(value is None for value in key):
            raise EquivalenceAnalysisError(
                f"trial {trial.trial_id} is missing a matched pair field"
            )
        if key in by_variant[variant]:
            raise EquivalenceAnalysisError(
                f"duplicate matched trial for {variant}: {key}"
            )
        by_variant[variant][key] = trial
    reference = by_variant[reference_variant]
    candidate = by_variant[candidate_variant]
    if not reference or not candidate:
        raise EquivalenceAnalysisError("both variants need measured trials")
    missing_candidate = sorted(set(reference) - set(candidate), key=str)
    missing_reference = sorted(set(candidate) - set(reference), key=str)
    if missing_candidate or missing_reference:
        raise EquivalenceAnalysisError(
            "variants do not share the same matched pairs: "
            f"missing_candidate={missing_candidate[:3]}, "
            f"missing_reference={missing_reference[:3]}"
        )
    return [(reference[key], candidate[key]) for key in sorted(reference, key=str)]


def _variant_label(trial: TrialResult) -> str:
    value = trial.input.get("variant_label")
    if value is None:
        value = trial.input.get("quantization_type")
    if not isinstance(value, str) or not value.strip():
        raise EquivalenceAnalysisError(f"trial {trial.trial_id} is missing variant label")
    return value


def _pair_value(trial: TrialResult, field: str) -> Any:
    if field == "task_id":
        return trial.task_id
    return trial.input.get(field)


def _binary_value(score: Mapping[str, Any], field: str) -> int:
    value = score.get(field)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isfinite(float(value)) and float(value) in (0.0, 1.0):
            return int(value)
    raise EquivalenceAnalysisError(
        f"score field {field!r} must be a boolean or 0/1 value; found {value!r}"
    )


def _paired_bootstrap_interval(
    differences: list[int],
    *,
    confidence: float,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    if not differences:
        raise EquivalenceAnalysisError("cannot bootstrap an empty paired sample")
    rng = random.Random(seed)
    n = len(differences)
    samples = []
    for _ in range(repeats):
        samples.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    return _percentile(samples, alpha), _percentile(samples, 1.0 - alpha)


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise EquivalenceAnalysisError("cannot calculate a percentile of no values")
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction
