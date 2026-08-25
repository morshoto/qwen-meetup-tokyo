"""Qwen-specific prompt formatting kept outside the common runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QwenPromptAdapter:
    """Format a text request as the message shape expected by Qwen processors."""

    def messages_for(self, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
