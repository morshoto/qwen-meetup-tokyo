"""Portable, injectable trial telemetry collection."""

from __future__ import annotations

import platform
import resource
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from llm_lab.generation import GenerationResponse


MemoryReader = Callable[[], tuple[int | None, str | None]]
EnvironmentFactory = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class TelemetryHandle:
    started_at: float
    starting_peak_memory: int | None


@dataclass(frozen=True)
class TelemetryRecord:
    total_seconds: float | None = None
    ttft_seconds: float | None = None
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    peak_memory_bytes: int | None = None
    memory_measurement: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_s": self.total_seconds,
            "ttft_s": self.ttft_seconds,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_measurement": self.memory_measurement,
            "environment": dict(self.environment),
        }


class TelemetryCollector:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = perf_counter,
        memory_reader: MemoryReader | None = None,
        environment_factory: EnvironmentFactory | None = None,
    ) -> None:
        self._clock = clock
        self._memory_reader = memory_reader or read_peak_memory
        self._environment_factory = environment_factory or capture_environment

    def start(self) -> TelemetryHandle:
        peak_memory, _ = self._memory_reader()
        return TelemetryHandle(
            started_at=self._clock(),
            starting_peak_memory=peak_memory,
        )

    def finish(
        self,
        handle: TelemetryHandle,
        response: GenerationResponse | None,
    ) -> TelemetryRecord:
        elapsed = self._clock() - handle.started_at
        ending_peak_memory, measurement = self._memory_reader()
        peak_memory = _max_optional(handle.starting_peak_memory, ending_peak_memory)
        timing = response.timing if response is not None else None
        usage = response.usage if response is not None else None
        prefill_seconds = timing.prefill_seconds if timing else None
        decode_seconds = timing.decode_seconds if timing else None
        total_seconds = timing.total_seconds if timing else None
        return TelemetryRecord(
            total_seconds=total_seconds if total_seconds is not None else elapsed,
            ttft_seconds=timing.ttft_seconds if timing else None,
            prefill_tokens_per_second=_rate(
                usage.prompt_tokens if usage else None,
                prefill_seconds,
            ),
            decode_tokens_per_second=_rate(
                usage.completion_tokens if usage else None,
                decode_seconds,
            ),
            peak_memory_bytes=peak_memory,
            memory_measurement=measurement,
            environment=self._environment_factory(),
        )


def read_peak_memory() -> tuple[int | None, str | None]:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError):
        return None, None
    if sys.platform.startswith("linux"):
        return value * 1024, "resource.getrusage.ru_maxrss_kib"
    return value, "resource.getrusage.ru_maxrss_bytes"


def capture_environment(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else None
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os": platform.system(),
        "git_sha": _git_sha(root),
    }


def _git_sha(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _max_optional(first: int | None, second: int | None) -> int | None:
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)


def _rate(tokens: int | None, seconds: float | None) -> float | None:
    if tokens is None or seconds is None or seconds <= 0:
        return None
    return tokens / seconds
