"""Optional llama.cpp/GGUF runtime adapter with explicit stream timings."""

from __future__ import annotations

import importlib
import importlib.metadata
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

from llm_lab.generation import (
    GenerationRequest,
    GenerationResponse,
    GenerationTiming,
    RuntimeMetadata,
    TokenUsage,
)
from llm_lab.models import ModelSpec

from .base import RuntimeConfig


ClientLoader = Callable[[ModelSpec, RuntimeConfig], Any]


class LlamaCppRuntime:
    """Run GGUF artifacts through the optional llama-cpp-python binding.

    The binding's streaming API does not expose portable backend timing
    counters. The response therefore records stream TTFT and elapsed time
    after the first streamed chunk explicitly; it does not populate the shared
    prefill/decode fields with measurements that could be mistaken for native
    kernel counters.
    """

    name = "llama.cpp"

    def __init__(self, *, loader: ClientLoader | None = None) -> None:
        self._loader = loader or _load_llama_cpp_client
        self._client: Any | None = None
        self._model_spec: ModelSpec | None = None
        self._runtime_config: RuntimeConfig | None = None

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        model_path = config.options.get("model_path")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("llama.cpp RuntimeConfig requires a non-empty model_path")
        self._client = self._loader(model, config)
        self._model_spec = model
        self._runtime_config = config

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._client is None or self._runtime_config is None:
            raise RuntimeError("runtime must be loaded before generation")

        started = perf_counter()
        stream = self._client.create_completion(
            prompt=request.prompt,
            stream=True,
            **_sampling_kwargs(request),
        )
        output_parts: list[str] = []
        first_chunk_at: float | None = None
        finish_reason: str | None = None
        usage: Mapping[str, Any] = {}
        for chunk in _iter_chunks(stream):
            if first_chunk_at is None:
                first_chunk_at = perf_counter()
            output_parts.append(_chunk_text(chunk))
            finish_reason = _chunk_finish_reason(chunk) or finish_reason
            usage = _chunk_usage(chunk) or usage

        finished = perf_counter()
        total_seconds = finished - started
        ttft_seconds = (
            first_chunk_at - started if first_chunk_at is not None else None
        )
        decode_seconds = (
            max(0.0, total_seconds - ttft_seconds)
            if ttft_seconds is not None
            else None
        )
        output_text = "".join(output_parts)
        prompt_tokens = _usage_int(usage, "prompt_tokens")
        completion_tokens = _usage_int(usage, "completion_tokens")
        if prompt_tokens is None:
            prompt_tokens = _token_count(self._client, request.prompt, add_bos=True)
        if completion_tokens is None:
            completion_tokens = _token_count(self._client, output_text, add_bos=False)
        model_spec = self._model_spec or request.model
        runtime_options = dict(self._runtime_config.options)
        runtime_options["timing_source"] = "first_stream_chunk"
        runtime_options["timing_semantics"] = (
            "stream_ttft_and_post_first_chunk_elapsed"
        )
        return GenerationResponse(
            output_text=output_text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            timing=GenerationTiming(
                ttft_seconds=ttft_seconds,
                post_first_chunk_seconds=decode_seconds,
                total_seconds=total_seconds,
            ),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version=(
                    self._runtime_config.version or _package_version()
                ),
                model_id=model_spec.model_id,
                tokenizer_id=model_spec.tokenizer_id,
                config=runtime_options,
            ),
            finish_reason=finish_reason,
        )

    def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        self._client = None
        self._model_spec = None
        self._runtime_config = None


def _load_llama_cpp_client(model: ModelSpec, config: RuntimeConfig) -> Any:
    try:
        llama_cpp = importlib.import_module("llama_cpp")
    except ImportError as error:
        raise RuntimeError(
            "LlamaCppRuntime requires the optional 'llama-cpp-python' dependency; "
            "install llm-lab[llama-cpp]"
        ) from error

    options = dict(config.options)
    model_path = options.pop("model_path")
    options.setdefault("verbose", False)
    return llama_cpp.Llama(model_path=model_path, **options)


def _sampling_kwargs(request: GenerationRequest) -> dict[str, Any]:
    sampling = request.sampling
    values: dict[str, Any] = {
        "max_tokens": sampling.max_new_tokens,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
    }
    if sampling.top_k is not None:
        values["top_k"] = sampling.top_k
    if sampling.seed is not None:
        values["seed"] = sampling.seed
    return values


def _iter_chunks(stream: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(stream, Mapping):
        return (stream,)
    return (chunk for chunk in stream if isinstance(chunk, Mapping))


def _chunk_text(chunk: Mapping[str, Any]) -> str:
    choices = chunk.get("choices", ())
    if not choices or not isinstance(choices[0], Mapping):
        return ""
    return str(choices[0].get("text", ""))


def _chunk_finish_reason(chunk: Mapping[str, Any]) -> str | None:
    choices = chunk.get("choices", ())
    if not choices or not isinstance(choices[0], Mapping):
        return None
    value = choices[0].get("finish_reason")
    return None if value is None else str(value)


def _chunk_usage(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = chunk.get("usage")
    return value if isinstance(value, Mapping) else None


def _usage_int(usage: Mapping[str, Any], key: str) -> int | None:
    value = usage.get(key)
    return int(value) if isinstance(value, (int, float)) else None


def _token_count(client: Any, text: str, *, add_bos: bool) -> int | None:
    tokenize = getattr(client, "tokenize", None)
    if not callable(tokenize):
        return None
    try:
        tokens = tokenize(text.encode("utf-8"), add_bos=add_bos)
    except (OSError, TypeError, ValueError):
        return None
    try:
        return len(tokens)
    except TypeError:
        return None


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        return None
