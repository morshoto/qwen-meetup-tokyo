"""Deterministic synthetic context construction for controlled experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Evidence:
    """A labelled text span that must remain intact in generated context."""

    id: str
    text: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.text.strip():
            raise ValueError("evidence id and text must be non-empty")


@dataclass(frozen=True)
class EvidenceSpan:
    id: str
    text: str
    token_start: int
    token_end: int
    requested_position: float
    actual_position: float


@dataclass(frozen=True)
class GeneratedContext:
    text: str
    token_count: int
    evidence: tuple[EvidenceSpan, ...]
    metadata: dict[str, Any]


class SyntheticContextGenerator:
    """Generate exact-length whitespace-token contexts from a seed.

    The generator intentionally does not pretend that whitespace tokens equal a
    model's BPE tokens. It provides a stable, dependency-free fixture convention;
    a future tokenizer-aware generator can use the same result contract.
    """

    _FILLER_WORDS = (
        "amber",
        "cobalt",
        "granite",
        "harbor",
        "linen",
        "meadow",
        "orbit",
        "parcel",
        "ribbon",
        "signal",
        "timber",
        "velvet",
    )

    def generate(
        self,
        evidence: Sequence[Evidence],
        *,
        target_tokens: int,
        evidence_position: float,
        seed: int,
    ) -> GeneratedContext:
        if target_tokens < 1:
            raise ValueError("target_tokens must be positive")
        if not 0.0 <= evidence_position <= 1.0:
            raise ValueError("evidence_position must be between 0 and 1")
        if not evidence:
            raise ValueError("at least one evidence span is required")

        evidence_tokens = [self._tokenize(item.text) for item in evidence]
        evidence_token_count = sum(len(tokens) for tokens in evidence_tokens)
        filler_count = target_tokens - evidence_token_count
        if filler_count < 0:
            raise ValueError("target_tokens must fit all evidence tokens")

        rng = random.Random(seed)
        filler = [
            f"{self._FILLER_WORDS[rng.randrange(len(self._FILLER_WORDS))]}-{index:04d}"
            for index in range(filler_count)
        ]
        insertion = round(filler_count * evidence_position)
        tokens = filler[:insertion]
        spans: list[EvidenceSpan] = []
        for item, item_tokens in zip(evidence, evidence_tokens):
            start = len(tokens)
            tokens.extend(item_tokens)
            end = len(tokens)
            spans.append(
                EvidenceSpan(
                    id=item.id,
                    text=item.text,
                    token_start=start,
                    token_end=end,
                    requested_position=evidence_position,
                    actual_position=start / max(1, filler_count),
                )
            )
        tokens.extend(filler[insertion:])

        return GeneratedContext(
            text=" ".join(tokens),
            token_count=len(tokens),
            evidence=tuple(spans),
            metadata={
                "generator": "synthetic.v001",
                "seed": seed,
                "tokenization": "whitespace-v1",
                "target_tokens": target_tokens,
            },
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = text.split()
        if not tokens:
            raise ValueError("evidence text must contain at least one token")
        return tokens
