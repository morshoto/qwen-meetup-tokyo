"""Deterministic synthetic context construction for controlled experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class ContextTokenizer(Protocol):
    """Minimal tokenizer surface needed for exact context construction."""

    name: str

    def encode(self, text: str) -> Sequence[int]:
        """Encode text without adding special tokens."""

    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token IDs without removing ordinary text."""


@dataclass(frozen=True)
class InferenceTokenizer:
    """Adapt a Transformers tokenizer to the context construction contract."""

    backend: Any
    name: str

    def encode(self, text: str) -> list[int]:
        return list(self.backend.encode(text, add_special_tokens=False))

    def decode(self, tokens: Sequence[int]) -> str:
        return str(self.backend.decode(list(tokens), skip_special_tokens=False))


@dataclass(frozen=True)
class LlamaCppTokenizer:
    """Adapt a loaded llama.cpp client to the context tokenizer contract."""

    backend: Any
    name: str

    def encode(self, text: str) -> list[int]:
        return list(self.backend.tokenize(text.encode("utf-8"), add_bos=False))

    def decode(self, tokens: Sequence[int]) -> str:
        value = self.backend.detokenize(list(tokens))
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)


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
    """Generate exact-length contexts from a deterministic seed.

    Without a tokenizer, the generator retains the dependency-free
    ``whitespace-v1`` fixture convention. A supplied tokenizer constructs and
    validates the context in that tokenizer's token IDs, which is required for
    model-backed context measurements.
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

    def __init__(self, tokenizer: ContextTokenizer | None = None) -> None:
        if tokenizer is not None and not isinstance(tokenizer, ContextTokenizer):
            raise TypeError("tokenizer must implement encode, decode, and name")
        self._tokenizer = tokenizer

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

        if self._tokenizer is not None:
            return self._generate_with_tokenizer(
                evidence,
                target_tokens=target_tokens,
                evidence_position=evidence_position,
                seed=seed,
            )

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

    def _generate_with_tokenizer(
        self,
        evidence: Sequence[Evidence],
        *,
        target_tokens: int,
        evidence_position: float,
        seed: int,
    ) -> GeneratedContext:
        tokenizer = self._tokenizer
        assert tokenizer is not None
        evidence_tokens = [list(tokenizer.encode(item.text)) for item in evidence]
        if any(not tokens for tokens in evidence_tokens):
            raise ValueError("tokenizer produced no tokens for evidence text")
        evidence_token_count = sum(len(tokens) for tokens in evidence_tokens)
        filler_count = target_tokens - evidence_token_count
        if filler_count < 0:
            raise ValueError("target_tokens must fit all evidence tokens")

        rng = random.Random(seed)
        filler_tokens: list[int] = []
        filler_index = 0
        while len(filler_tokens) < filler_count:
            word = self._FILLER_WORDS[rng.randrange(len(self._FILLER_WORDS))]
            fragment = f" {word}-{filler_index:04d}"
            fragment_tokens = list(tokenizer.encode(fragment))
            if not fragment_tokens:
                raise ValueError("tokenizer produced no tokens for filler text")
            remaining = filler_count - len(filler_tokens)
            filler_tokens.extend(fragment_tokens[:remaining])
            filler_index += 1

        insertion = round(filler_count * evidence_position)
        token_ids = filler_tokens[:insertion]
        for item_tokens in evidence_tokens:
            token_ids.extend(item_tokens)
        token_ids.extend(filler_tokens[insertion:])
        text = tokenizer.decode(token_ids)
        text, round_trip_tokens = _stabilize_token_count(
            tokenizer,
            text,
            target_tokens=target_tokens,
        )
        spans = _locate_evidence_spans(
            tokenizer,
            text,
            evidence,
            requested_position=evidence_position,
        )

        return GeneratedContext(
            text=text,
            token_count=len(round_trip_tokens),
            evidence=tuple(spans),
            metadata={
                "generator": "synthetic.v001",
                "seed": seed,
                "tokenization": tokenizer.name,
                "tokenization_mode": "tokenizer",
                "target_tokens": target_tokens,
            },
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = text.split()
        if not tokens:
            raise ValueError("evidence text must contain at least one token")
        return tokens


def _stabilize_token_count(
    tokenizer: ContextTokenizer,
    text: str,
    *,
    target_tokens: int,
) -> tuple[str, list[int]]:
    """Retokenize decoded IDs and repair small boundary losses explicitly.

    Tokenizing fragments independently can create a non-canonical token ID
    sequence at fragment boundaries.  In particular, llama.cpp can decode a
    target-length sequence whose text retokenizes one or two tokens shorter.
    We only repair that case by appending a tokenizer-verified one-token
    filler; any other drift remains a hard failure.
    """

    round_trip_tokens = list(tokenizer.encode(text))
    if len(round_trip_tokens) > target_tokens:
        raise ValueError(
            "tokenizer decode/encode round trip exceeded the requested "
            f"token count: expected {target_tokens}, got {len(round_trip_tokens)}"
        )

    deficit = target_tokens - len(round_trip_tokens)
    if deficit == 0:
        return text, round_trip_tokens

    for fragment in (" a", " e", " i", " o", " u", " x", " 0", "\n"):
        fragment_tokens = list(tokenizer.encode(fragment))
        if len(fragment_tokens) != 1:
            continue
        candidate_text = text
        candidate_tokens = round_trip_tokens
        for _ in range(deficit):
            candidate_text += fragment
            candidate_tokens = list(tokenizer.encode(candidate_text))
            if len(candidate_tokens) != len(round_trip_tokens) + _ + 1:
                break
        else:
            if len(candidate_tokens) == target_tokens:
                return candidate_text, candidate_tokens

    raise ValueError(
        "tokenizer decode/encode round trip could not be repaired to the "
        f"requested token count: expected {target_tokens}, got "
        f"{len(round_trip_tokens)}"
    )


def _locate_evidence_spans(
    tokenizer: ContextTokenizer,
    text: str,
    evidence: Sequence[Evidence],
    *,
    requested_position: float,
) -> list[EvidenceSpan]:
    """Locate evidence in final text and report final-token offsets."""

    spans: list[EvidenceSpan] = []
    search_start = 0
    for item in evidence:
        start_char = text.find(item.text, search_start)
        if start_char < 0:
            raise ValueError(f"evidence text was lost during tokenization: {item.id}")
        end_char = start_char + len(item.text)
        token_start = len(tokenizer.encode(text[:start_char]))
        token_end = len(tokenizer.encode(text[:end_char]))
        spans.append(
            EvidenceSpan(
                id=item.id,
                text=item.text,
                token_start=token_start,
                token_end=token_end,
                requested_position=requested_position,
                actual_position=token_start / max(1, len(tokenizer.encode(text))),
            )
        )
        search_start = end_char
    return spans
