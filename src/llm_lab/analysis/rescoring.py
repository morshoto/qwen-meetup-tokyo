"""Diagnostic re-scoring for historical trial records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from llm_lab.evaluation import (
    CalibratedAnswerScorer,
    EvaluationTask,
    ScoreResult,
    Scorer,
    TrialResult,
    TrialStatus,
)
from llm_lab.generation import GenerationResponse


EXACT_MATCH = "exact_match"
MISMATCH = "mismatch"
FORMAT_FAILURE = "format_failure"
RUNTIME_FAILURE = "runtime_failure"


@dataclass(frozen=True)
class RescoredTrial:
    """A non-mutating comparison of one historical trial and new score."""

    trial: TrialResult
    score: ScoreResult | None
    category: str
    legacy_correct: bool | None


def rescore_trial(
    trial: TrialResult,
    task: EvaluationTask,
    *,
    scorer: Scorer | None = None,
) -> RescoredTrial:
    """Apply a calibrated scorer without changing the historical trial."""

    legacy_correct = trial.score.get("correct")
    if not isinstance(legacy_correct, bool):
        legacy_correct = None

    if trial.status != TrialStatus.COMPLETED:
        return RescoredTrial(
            trial=trial,
            score=None,
            category=RUNTIME_FAILURE,
            legacy_correct=legacy_correct,
        )

    output = trial.generation.get("output_text")
    response = GenerationResponse(output_text=output if isinstance(output, str) else "")
    calibrated_score = (scorer or CalibratedAnswerScorer()).score(task, response)
    return RescoredTrial(
        trial=trial,
        score=calibrated_score,
        category=_category_for(calibrated_score),
        legacy_correct=legacy_correct,
    )


def rescore_trials(
    trials: Iterable[TrialResult | Mapping[str, Any]],
    tasks: Mapping[str, EvaluationTask],
    *,
    scorer: Scorer | None = None,
) -> list[RescoredTrial]:
    """Re-score a batch using an explicitly supplied task catalog."""

    results: list[RescoredTrial] = []
    for value in trials:
        trial = value if isinstance(value, TrialResult) else TrialResult.from_record(value)
        task = tasks.get(trial.task_id)
        if task is None:
            raise ValueError(f"missing task definition for {trial.task_id!r}")
        results.append(rescore_trial(trial, task, scorer=scorer))
    return results


def comparison_rows(
    trials: Iterable[RescoredTrial],
) -> list[dict[str, Any]]:
    """Aggregate legacy/calibrated outcomes by task, variant, and context."""

    groups: dict[tuple[Any, ...], list[RescoredTrial]] = {}
    for result in trials:
        source = result.trial
        metadata = source.input
        key = (
            source.experiment_id,
            source.task_id,
            metadata.get("task_type", "unknown"),
            metadata.get("variant_condition_id"),
            metadata.get("target_context_tokens"),
            metadata.get("condition_id", "default"),
        )
        groups.setdefault(key, []).append(result)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=_group_sort_key):
        (
            experiment_id,
            task_id,
            task_type,
            variant_condition_id,
            target_context_tokens,
            condition_id,
        ) = key
        legacy_values = [
            result.legacy_correct
            for result in group
            if isinstance(result.legacy_correct, bool)
        ]
        exact_values = [
            result.score.exact_correct
            for result in group
            if result.score is not None and isinstance(result.score.exact_correct, bool)
        ]
        answer_bearing_values = [
            result.score.answer_bearing_correct
            for result in group
            if result.score is not None
            and isinstance(result.score.answer_bearing_correct, bool)
        ]
        format_values = [
            result.score.format_valid
            for result in group
            if result.score is not None and isinstance(result.score.format_valid, bool)
        ]
        rows.append(
            {
                "experiment_id": experiment_id,
                "task_id": task_id,
                "task_type": task_type,
                "condition_id": condition_id,
                "variant_condition_id": variant_condition_id,
                "variant_label": _common_input_value(group, "variant_label"),
                "quantization_type": _common_input_value(group, "quantization_type"),
                "target_context_tokens": target_context_tokens,
                "attempted_n": len(group),
                "completed_n": sum(
                    result.trial.status == TrialStatus.COMPLETED for result in group
                ),
                "old_scorer": _common_old_score_value(group, "scorer"),
                "old_scored_n": len(legacy_values),
                "old_correct_n": sum(legacy_values),
                "old_accuracy": _accuracy(legacy_values),
                "new_scorer": _common_new_score_value(group),
                "new_exact_scored_n": len(exact_values),
                "new_exact_correct_n": sum(exact_values),
                "new_exact_accuracy": _accuracy(exact_values),
                "new_answer_bearing_scored_n": len(answer_bearing_values),
                "new_answer_bearing_correct_n": sum(answer_bearing_values),
                "new_answer_bearing_accuracy": _accuracy(answer_bearing_values),
                "new_format_scored_n": len(format_values),
                "new_format_valid_n": sum(format_values),
                "new_format_validity": _accuracy(format_values),
                "score_changed_n": sum(
                    result.legacy_correct != result.score.exact_correct
                    for result in group
                    if isinstance(result.legacy_correct, bool)
                    and result.score is not None
                    and isinstance(result.score.exact_correct, bool)
                ),
                "exact_match_n": _category_count(group, EXACT_MATCH),
                "mismatch_n": _category_count(group, MISMATCH),
                "format_failure_n": _category_count(group, FORMAT_FAILURE),
                "runtime_failure_n": _category_count(group, RUNTIME_FAILURE),
            }
        )
    return rows


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for an existing input artifact."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    digest = hashlib.sha256()
    with source_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_report(
    *,
    raw_path: str | Path,
    task_catalog_path: str | Path,
    raw_trial_n: int,
    rows: Iterable[Mapping[str, Any]],
) -> str:
    """Render a deterministic provenance and interpretation-boundary report."""

    comparison = list(rows)
    variants: dict[str, list[Mapping[str, Any]]] = {}
    for row in comparison:
        variant = str(row.get("variant_condition_id") or "<unknown>")
        variants.setdefault(variant, []).append(row)

    lines = [
        "# exp_002 diagnostic re-scoring report",
        "",
        "**Status: Diagnostic re-scoring only.** This report reinterprets existing",
        "generated outputs; it is not a new model run or formal quantization",
        "measurement.",
        "",
        "## Provenance",
        "",
        f"- Raw input: `results/raw/{Path(raw_path).name}`",
        f"- Raw trial count: {raw_trial_n}",
        f"- Raw SHA-256: `{sha256_file(raw_path)}`",
        f"- Task catalog: `data/tasks/{Path(task_catalog_path).name}`",
        "- Legacy scorer: `expected.v1` (preserved in the raw records)",
        "- Calibrated scorer: `calibrated.v1`",
        "- Processing entry point: `rescore.py`",
        "",
        "## Old/new comparison by quantization variant",
        "",
        "The detailed comparison table is `rescored-summary.csv`, with one row",
        "per task, quantization variant, and context length.",
        "",
        "| Variant | Old correct/scored | New exact correct/scored | New answer-bearing correct/scored | New format-valid/scored | Mismatch | Format failure | Runtime failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, variant_rows in sorted(variants.items()):
        label = _common_row_value(variant_rows, "variant_label") or variant
        lines.append(
            "| "
            + " | ".join(
                (
                    str(label),
                    _count_cell(variant_rows, "old_correct_n", "old_scored_n"),
                    _count_cell(
                        variant_rows,
                        "new_exact_correct_n",
                        "new_exact_scored_n",
                    ),
                    _count_cell(
                        variant_rows,
                        "new_answer_bearing_correct_n",
                        "new_answer_bearing_scored_n",
                    ),
                    _count_cell(
                        variant_rows,
                        "new_format_valid_n",
                        "new_format_scored_n",
                    ),
                    str(_sum_rows(variant_rows, "mismatch_n")),
                    str(_sum_rows(variant_rows, "format_failure_n")),
                    str(_sum_rows(variant_rows, "runtime_failure_n")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failure classification",
            "",
            "- `mismatch`: completed output is not an exact calibrated answer but",
            "  remains format-valid.",
            "- `format_failure`: completed output is empty or violates the expected",
            "  answer shape; this category takes precedence over mismatch.",
            "- `runtime_failure`: the original trial did not complete; no output is",
            "  rescored and it remains in the attempted denominator.",
            "",
            "## Interpretation boundary",
            "",
            "The historical raw JSONL and its `expected.v1` scores are preserved.",
            "This diagnostic table must not be used for a final quantization claim",
            "without the required caveat and a formal re-measurement under the",
            "calibrated policy. The existing formal `summary.csv` is unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def _category_for(score: ScoreResult) -> str:
    if score.format_valid is False:
        return FORMAT_FAILURE
    if score.exact_correct is False:
        return MISMATCH
    if score.exact_correct is True:
        return EXACT_MATCH
    raise ValueError("calibrated score must declare exactness or format validity")


def _accuracy(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _category_count(group: Iterable[RescoredTrial], category: str) -> int:
    return sum(result.category == category for result in group)


def _common_input_value(
    group: Iterable[RescoredTrial],
    key: str,
) -> Any:
    values = [result.trial.input.get(key) for result in group]
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _common_old_score_value(
    group: Iterable[RescoredTrial],
    key: str,
) -> Any:
    values = [result.trial.score.get(key) for result in group]
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _common_new_score_value(group: Iterable[RescoredTrial]) -> str | None:
    values = [
        result.score.scorer
        for result in group
        if result.score is not None
    ]
    if not values or any(value != values[0] for value in values[1:]):
        return None
    return values[0]


def _group_sort_key(item: tuple[tuple[Any, ...], list[RescoredTrial]]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in item[0])


def _sum_rows(rows: Iterable[Mapping[str, Any]], key: str) -> int:
    return sum(
        int(row.get(key, 0))
        for row in rows
        if isinstance(row.get(key, 0), int)
    )


def _count_cell(
    rows: Iterable[Mapping[str, Any]],
    correct_key: str,
    scored_key: str,
) -> str:
    return f"{_sum_rows(rows, correct_key)}/{_sum_rows(rows, scored_key)}"


def _common_row_value(
    rows: Iterable[Mapping[str, Any]],
    key: str,
) -> Any:
    values = [row.get(key) for row in rows]
    if not values or any(value is None for value in values):
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None
