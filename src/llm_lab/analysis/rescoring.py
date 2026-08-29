"""Diagnostic re-scoring for historical trial records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


def _category_for(score: ScoreResult) -> str:
    if score.format_valid is False:
        return FORMAT_FAILURE
    if score.exact_correct is False:
        return MISMATCH
    if score.exact_correct is True:
        return EXACT_MATCH
    raise ValueError("calibrated score must declare exactness or format validity")
