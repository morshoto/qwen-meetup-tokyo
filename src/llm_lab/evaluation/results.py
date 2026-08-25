"""Stable per-trial result records and deterministic trial identities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class TrialStatus(str, Enum):
    COMPLETED = "completed"
    RUNTIME_ERROR = "runtime_error"
    SCORER_ERROR = "scorer_error"
    INVALID_INPUT = "invalid_input"
    INVALID_OUTPUT = "invalid_output"
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


def make_trial_id(
    experiment_id: str,
    task_id: str,
    *,
    condition_id: str = "default",
    repeat_index: int = 1,
) -> str:
    if not experiment_id.strip() or not task_id.strip() or not condition_id.strip():
        raise ValueError("experiment, task, and condition IDs must be non-empty")
    if repeat_index < 1:
        raise ValueError("repeat_index must be positive")
    return f"{experiment_id}:{task_id}:{condition_id}:run{repeat_index:02d}"


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    experiment_id: str
    task_id: str
    status: TrialStatus | str
    model: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    input: Mapping[str, Any] = field(default_factory=dict)
    generation: Mapping[str, Any] = field(default_factory=dict)
    score: Mapping[str, Any] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=dict)
    memory: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    error: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.trial_id.strip() or not self.experiment_id.strip() or not self.task_id.strip():
            raise ValueError("trial, experiment, and task IDs must be non-empty")
        try:
            normalized_status = TrialStatus(self.status)
        except ValueError as error:
            raise ValueError(f"unsupported trial status: {self.status!r}") from error
        object.__setattr__(self, "status", normalized_status)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trial_id": self.trial_id,
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "model": dict(self.model),
            "runtime": dict(self.runtime),
            "input": dict(self.input),
            "generation": dict(self.generation),
            "score": dict(self.score),
            "timing": dict(self.timing),
            "memory": dict(self.memory),
            "environment": dict(self.environment),
            "error": None if self.error is None else dict(self.error),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TrialResult":
        if record.get("schema_version") != 1:
            raise ValueError("unsupported trial result schema version")
        required = {"trial_id", "experiment_id", "task_id", "status"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"trial result is missing required fields: {sorted(missing)}")
        return cls(
            trial_id=str(record["trial_id"]),
            experiment_id=str(record["experiment_id"]),
            task_id=str(record["task_id"]),
            status=record["status"],
            model=_mapping_or_empty(record.get("model")),
            runtime=_mapping_or_empty(record.get("runtime")),
            input=_mapping_or_empty(record.get("input")),
            generation=_mapping_or_empty(record.get("generation")),
            score=_mapping_or_empty(record.get("score")),
            timing=_mapping_or_empty(record.get("timing")),
            memory=_mapping_or_empty(record.get("memory")),
            environment=_mapping_or_empty(record.get("environment")),
            error=None if record.get("error") is None else _mapping_or_empty(record["error"]),
        )


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("trial result sections must be JSON objects")
    return dict(value)
