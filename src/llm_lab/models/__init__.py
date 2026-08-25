"""Model metadata and model-specific adapters."""

from .qwen import QwenPromptAdapter
from .specs import (
    QWEN38_MODEL_ID,
    ModelCapabilities,
    ModelSpec,
    qwen38_model_spec,
)

__all__ = [
    "QWEN38_MODEL_ID",
    "ModelCapabilities",
    "ModelSpec",
    "QwenPromptAdapter",
    "qwen38_model_spec",
]
