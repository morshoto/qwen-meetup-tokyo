"""Model identities and declared capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelCapabilities:
    max_context_tokens: int | None = None
    supports_vision: bool = False
    supports_video: bool = False
    supports_tools: bool = False
    supports_thinking: bool = False

    def __post_init__(self) -> None:
        if self.max_context_tokens is not None and self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be positive")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")


QWEN38_MODEL_ID = "Qwen/Qwen3.8-27B"


def qwen38_model_spec(
    *,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
) -> ModelSpec:
    """Return the Qwen3.8-27B identity without selecting a runtime."""

    return ModelSpec(
        model_id=QWEN38_MODEL_ID,
        revision=revision,
        tokenizer_id=QWEN38_MODEL_ID,
        tokenizer_revision=tokenizer_revision,
        capabilities=ModelCapabilities(
            supports_vision=True,
            supports_video=True,
            supports_thinking=True,
        ),
        metadata={"family": "qwen3.8"},
    )
