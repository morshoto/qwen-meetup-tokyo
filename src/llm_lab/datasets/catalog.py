"""Loading and validating the small, versioned task catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class TaskDefinition:
    """A machine-checkable benchmark task definition."""

    task_id: str
    task_type: str
    version: int
    question: str
    expected: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]


class TaskCatalog:
    """Validated task definitions indexed in their source-file order."""

    def __init__(self, tasks: Iterable[TaskDefinition]) -> None:
        task_list = tuple(tasks)
        ids = [task.task_id for task in task_list]
        if len(ids) != len(set(ids)):
            raise ValueError("task IDs must be unique")
        self.tasks = task_list

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def get(self, task_id: str) -> TaskDefinition:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TaskCatalog":
        records: list[Mapping[str, Any]] = []
        with Path(path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON on line {line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"task on line {line_number} must be an object")
                records.append(record)
        return cls.from_records(records)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "TaskCatalog":
        return cls(_parse_task(record) for record in records)


def _parse_task(record: Mapping[str, Any]) -> TaskDefinition:
    required = {
        "schema_version",
        "id",
        "type",
        "version",
        "question",
        "expected",
        "evidence",
        "metadata",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"task is missing required fields: {sorted(missing)}")
    if record["schema_version"] != 1:
        raise ValueError("unsupported task schema version")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError("task id must be a non-empty string")
    if not isinstance(record["type"], str) or not record["type"]:
        raise ValueError("task type must be a non-empty string")
    if not isinstance(record["version"], int) or record["version"] < 1:
        raise ValueError("task version must be a positive integer")
    if not isinstance(record["question"], str) or not record["question"].strip():
        raise ValueError("task question must be non-empty")
    expected = record["expected"]
    if not isinstance(expected, dict) or not isinstance(expected.get("type"), str):
        raise ValueError("expected must include scorer metadata under 'type'")
    if "value" not in expected and "accepted" not in expected:
        raise ValueError("expected must include a value or accepted answers")
    evidence = record["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must be a non-empty list")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("id"), str)
        or not isinstance(item.get("text"), str)
        for item in evidence
    ):
        raise ValueError("each evidence item needs string id and text fields")
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    for key in ("seed", "source", "license"):
        if key not in metadata:
            raise ValueError(f"metadata must include {key!r}")

    return TaskDefinition(
        task_id=record["id"],
        task_type=record["type"],
        version=record["version"],
        question=record["question"],
        expected=expected,
        evidence=tuple(evidence),
        metadata=metadata,
    )
