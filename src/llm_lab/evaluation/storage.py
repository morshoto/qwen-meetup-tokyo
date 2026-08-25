"""Append-only JSONL storage for trial records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from .results import TrialResult


class JsonlResultWriter:
    """Append trial records while rejecting duplicate trial IDs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._seen_ids = {result.trial_id for result in load_trial_results(self.path)}

    def append(self, result: TrialResult) -> None:
        if result.trial_id in self._seen_ids:
            raise ValueError(f"duplicate trial_id: {result.trial_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result.to_record(), ensure_ascii=False, sort_keys=True))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        self._seen_ids.add(result.trial_id)


def load_trial_results(path: str | Path) -> list[TrialResult]:
    source_path = Path(path)
    if not source_path.exists():
        return []
    results: list[TrialResult] = []
    seen_ids: set[str] = set()
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid trial JSON on line {line_number}") from error
            try:
                result = TrialResult.from_record(record)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid trial record on line {line_number}") from error
            if result.trial_id in seen_ids:
                raise ValueError(f"duplicate trial_id on line {line_number}: {result.trial_id}")
            seen_ids.add(result.trial_id)
            results.append(result)
    return results
