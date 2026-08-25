"""Protocol and configuration shared by local inference adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from llm_lab.generation import GenerationRequest, GenerationResponse
from llm_lab.models import ModelSpec


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    version: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("runtime name must be non-empty")


@runtime_checkable
class Runtime(Protocol):
    name: str

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        """Load the model artifact using the explicit runtime configuration."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate one response for a backend-neutral request."""

    def close(self) -> None:
        """Release model/runtime resources."""
