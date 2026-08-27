"""Backend-neutral request, response, and measurement types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from llm_lab.models import ModelSpec


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be greater than 0 and at most 1")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive when provided")

    def as_backend_kwargs(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.temperature > 0,
        }
        if self.top_k is not None:
            values["top_k"] = self.top_k
        if self.seed is not None:
            values["seed"] = self.seed
        return values


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    model: ModelSpec
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class GenerationTiming:
    ttft_seconds: float | None = None
    prefill_seconds: float | None = None
    decode_seconds: float | None = None
    post_first_chunk_seconds: float | None = None
    total_seconds: float | None = None


@dataclass(frozen=True)
class RuntimeMetadata:
    runtime_name: str
    runtime_version: str | None
    model_id: str
    tokenizer_id: str | None
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResponse:
    output_text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    timing: GenerationTiming = field(default_factory=GenerationTiming)
    runtime: RuntimeMetadata | None = None
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
