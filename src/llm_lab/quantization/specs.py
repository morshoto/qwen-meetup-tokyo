"""Validated provenance and control metadata for quantized model artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


SUPPORTED_GGUF_QUANTIZATIONS = frozenset(
    {"F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"}
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_CONDITION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_BITS_BY_QUANTIZATION = {
    "F16": 16,
    "Q8_0": 8,
    "Q6_K": 6,
    "Q5_K_M": 5,
    "Q4_K_M": 4,
}


@dataclass(frozen=True)
class ArtifactProvenance:
    """Source and immutable identity of one materialized model artifact."""

    source_uri: str
    source_revision: str
    conversion_command: str
    converter_revision: str
    artifact_uri: str
    artifact_sha256: str
    artifact_size_bytes: int

    def __post_init__(self) -> None:
        for field_name in (
            "source_uri",
            "source_revision",
            "conversion_command",
            "converter_revision",
            "artifact_uri",
        ):
            _require_text(field_name, getattr(self, field_name))
        if not _SHA256_RE.fullmatch(self.artifact_sha256):
            raise ValueError("artifact_sha256 must be a 64 hexadecimal character string")
        if self.artifact_size_bytes < 1:
            raise ValueError("artifact_size_bytes must be positive")

    def to_record(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "source_revision": self.source_revision,
            "conversion_command": self.conversion_command,
            "converter_revision": self.converter_revision,
            "artifact_uri": self.artifact_uri,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ArtifactProvenance":
        return cls(
            source_uri=str(record["source_uri"]),
            source_revision=str(record["source_revision"]),
            conversion_command=str(record["conversion_command"]),
            converter_revision=str(record["converter_revision"]),
            artifact_uri=str(record["artifact_uri"]),
            artifact_sha256=str(record["artifact_sha256"]),
            artifact_size_bytes=int(record["artifact_size_bytes"]),
        )


@dataclass(frozen=True)
class QuantizationVariant:
    """One fixed-format condition in a quantization comparison."""

    condition_id: str
    label: str
    format: str
    quantization_type: str
    bits: int
    artifact: ArtifactProvenance
    runtime_kernel: str

    def __post_init__(self) -> None:
        if not _CONDITION_ID_RE.fullmatch(self.condition_id):
            raise ValueError(
                "condition_id must contain lowercase letters, digits, '.', '_', or '-'")
        _require_text("label", self.label)
        if self.format != "GGUF":
            raise ValueError("format must be GGUF")
        if self.quantization_type not in SUPPORTED_GGUF_QUANTIZATIONS:
            raise ValueError(
                f"unsupported GGUF quantization type: {self.quantization_type!r}")
        if self.bits != _BITS_BY_QUANTIZATION[self.quantization_type]:
            raise ValueError(
                f"bits must be {_BITS_BY_QUANTIZATION[self.quantization_type]} "
                f"for {self.quantization_type}")
        _require_text("runtime_kernel", self.runtime_kernel)

    def to_record(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "label": self.label,
            "format": self.format,
            "quantization_type": self.quantization_type,
            "bits": self.bits,
            "artifact": self.artifact.to_record(),
            "runtime_kernel": self.runtime_kernel,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "QuantizationVariant":
        return cls(
            condition_id=str(record["condition_id"]),
            label=str(record["label"]),
            format=str(record["format"]),
            quantization_type=str(record["quantization_type"]),
            bits=int(record["bits"]),
            artifact=ArtifactProvenance.from_record(record["artifact"]),
            runtime_kernel=str(record["runtime_kernel"]),
        )


@dataclass(frozen=True)
class QuantizationManifest:
    """The controls and variants needed to interpret one sweep."""

    experiment_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    runtime_name: str
    runtime_version: str
    prompt_id: str
    task_ids: tuple[str, ...]
    context_lengths: tuple[int, ...]
    sampling: Mapping[str, Any]
    variants: tuple[QuantizationVariant, ...]
    repeats: int = 1
    context_length_semantics: str = "input_tokens"
    context_overhead_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "runtime_name",
            "runtime_version",
            "prompt_id",
        ):
            _require_text(field_name, getattr(self, field_name))
        if not self.task_ids or any(not task_id.strip() for task_id in self.task_ids):
            raise ValueError("task_ids must contain at least one non-empty ID")
        if not self.context_lengths or any(length < 1 for length in self.context_lengths):
            raise ValueError("context_lengths must contain positive lengths")
        if tuple(sorted(set(self.context_lengths))) != self.context_lengths:
            raise ValueError("context_lengths must be strictly increasing")
        if not self.sampling:
            raise ValueError("sampling controls must not be empty")
        if not self.variants:
            raise ValueError("variants must contain at least one condition")
        condition_ids = [variant.condition_id for variant in self.variants]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("duplicate condition_id in variants")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if self.context_length_semantics != "input_tokens":
            raise ValueError("context_length_semantics must be input_tokens")
        if self.context_overhead_tokens < 0:
            raise ValueError("context_overhead_tokens cannot be negative")

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(variant.condition_id for variant in self.variants)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
                "tokenizer_id": self.tokenizer_id,
                "tokenizer_revision": self.tokenizer_revision,
            },
            "runtime": {
                "name": self.runtime_name,
                "version": self.runtime_version,
            },
            "controls": {
                "prompt_id": self.prompt_id,
                "task_ids": list(self.task_ids),
                "context_lengths": list(self.context_lengths),
                "sampling": dict(self.sampling),
                "repeats": self.repeats,
                "context_length_semantics": self.context_length_semantics,
                "context_overhead_tokens": self.context_overhead_tokens,
            },
            "variants": [variant.to_record() for variant in self.variants],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "QuantizationManifest":
        if record.get("schema_version") != 1:
            raise ValueError("unsupported quantization manifest schema version")
        model = record["model"]
        runtime = record["runtime"]
        controls = record["controls"]
        return cls(
            experiment_id=str(record["experiment_id"]),
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            tokenizer_id=str(model["tokenizer_id"]),
            tokenizer_revision=str(model["tokenizer_revision"]),
            runtime_name=str(runtime["name"]),
            runtime_version=str(runtime["version"]),
            prompt_id=str(controls["prompt_id"]),
            task_ids=tuple(str(value) for value in controls["task_ids"]),
            context_lengths=tuple(int(value) for value in controls["context_lengths"]),
            sampling=dict(controls["sampling"]),
            variants=tuple(
                QuantizationVariant.from_record(value) for value in record["variants"]
            ),
            repeats=int(controls.get("repeats", 1)),
            context_length_semantics=str(
                controls.get("context_length_semantics", "input_tokens")
            ),
            context_overhead_tokens=int(controls.get("context_overhead_tokens", 0)),
        )


def _require_text(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
