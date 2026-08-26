"""Optional Hugging Face Transformers adapter for local Qwen generation."""

from __future__ import annotations

import importlib
import importlib.metadata
from time import perf_counter
from typing import Any, Callable

from llm_lab.generation import (
    GenerationRequest,
    GenerationResponse,
    GenerationTiming,
    RuntimeMetadata,
    TokenUsage,
)
from llm_lab.models import ModelSpec
from llm_lab.models.qwen import QwenPromptAdapter

from .base import RuntimeConfig


ComponentLoader = Callable[[ModelSpec, RuntimeConfig], tuple[Any, Any]]


class TransformersRuntime:
    """Run a model through Transformers without importing it at package import time.

    ``loader`` is injectable for unit tests and for applications that manage model
    loading themselves. The default loader imports Transformers only when ``load``
    is called, keeping the base package usable without heavyweight dependencies.
    """

    name = "transformers"

    def __init__(
        self,
        *,
        loader: ComponentLoader | None = None,
        prompt_adapter: Any | None = None,
    ) -> None:
        self._loader = loader or _load_transformers_components
        self._prompt_adapter = prompt_adapter or _PlainTextPromptAdapter()
        self._processor: Any | None = None
        self._model: Any | None = None
        self._model_spec: ModelSpec | None = None
        self._runtime_config: RuntimeConfig | None = None

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        self._processor, self._model = self._loader(model, config)
        self._model_spec = model
        self._runtime_config = config

    def get_tokenizer(self) -> Any:
        """Return the tokenizer used by the loaded processor for input encoding."""

        if self._processor is None:
            raise RuntimeError("runtime must be loaded before accessing its tokenizer")
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        if not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise RuntimeError("loaded processor does not expose a compatible tokenizer")
        return tokenizer

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._processor is None or self._model is None or self._runtime_config is None:
            raise RuntimeError("runtime must be loaded before generation")

        started = perf_counter()
        messages = self._prompt_adapter.messages_for(request.prompt)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(getattr(self._model, "device", "cpu"))
        prompt_tokens = _token_count(inputs["input_ids"])
        generated = self._model.generate(
            **inputs,
            **request.sampling.as_backend_kwargs(),
        )
        sequence = _first_sequence(generated)
        output_tokens = max(0, _token_count(sequence) - prompt_tokens)
        output_text = self._processor.decode(
            sequence[prompt_tokens:],
            skip_special_tokens=True,
        )
        elapsed = perf_counter() - started
        model_spec = self._model_spec or request.model
        config = dict(self._runtime_config.options)
        config.setdefault("model_revision", model_spec.revision)
        config.setdefault("tokenizer_revision", model_spec.tokenizer_revision)
        return GenerationResponse(
            output_text=output_text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=output_tokens,
            ),
            timing=GenerationTiming(total_seconds=elapsed),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version=self._runtime_config.version or _package_version(),
                model_id=model_spec.model_id,
                tokenizer_id=model_spec.tokenizer_id,
                config=config,
            ),
        )

    def close(self) -> None:
        self._processor = None
        self._model = None
        self._model_spec = None
        self._runtime_config = None


class QwenTransformersRuntime(TransformersRuntime):
    """Transformers adapter with Qwen's model-specific message formatting."""

    def __init__(self, *, loader: ComponentLoader | None = None) -> None:
        super().__init__(loader=loader, prompt_adapter=QwenPromptAdapter())


class _PlainTextPromptAdapter:
    def messages_for(self, prompt: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": prompt}]


def _load_transformers_components(model: ModelSpec, config: RuntimeConfig) -> tuple[Any, Any]:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise RuntimeError(
            "TransformersRuntime requires the optional 'transformers' dependency; "
            "install llm-lab[transformers]"
        ) from error

    options = dict(config.options)
    tokenizer_id = model.tokenizer_id or model.model_id
    tokenizer_revision = model.tokenizer_revision or options.get("tokenizer_revision")
    revision = model.revision or options.get("revision")
    processor_kwargs = _revision_kwargs(tokenizer_revision)
    processor_kwargs["trust_remote_code"] = options.get("trust_remote_code", True)
    processor = transformers.AutoProcessor.from_pretrained(tokenizer_id, **processor_kwargs)

    model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:
        model_class = transformers.AutoModelForCausalLM
    model_kwargs = _revision_kwargs(revision)
    model_kwargs["device_map"] = options.get("device_map", "auto")
    model_kwargs["trust_remote_code"] = options.get("trust_remote_code", True)
    if "torch_dtype" in options:
        model_kwargs["torch_dtype"] = _resolve_torch_dtype(options["torch_dtype"])
    return processor, model_class.from_pretrained(model.model_id, **model_kwargs)


def _revision_kwargs(revision: Any) -> dict[str, Any]:
    return {} if revision is None else {"revision": revision}


def _resolve_torch_dtype(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        torch = importlib.import_module("torch")
        return getattr(torch, value)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"unknown torch_dtype {value!r}") from error


def _first_sequence(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return value[0]
    return value[0] if hasattr(value, "__getitem__") and getattr(value, "ndim", 1) > 1 else value


def _token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape:
        return int(shape[-1])
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return len(value[0])
        return len(value)
    raise TypeError("cannot determine token count from backend output")


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return None
