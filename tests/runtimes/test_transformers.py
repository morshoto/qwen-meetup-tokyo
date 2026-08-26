import unittest
from typing import Any

from llm_lab.generation import GenerationRequest, SamplingConfig
from llm_lab.models import qwen38_model_spec
from llm_lab.models.qwen import QwenPromptAdapter
from llm_lab.runtimes import Runtime, RuntimeConfig
from llm_lab.runtimes.transformers import QwenTransformersRuntime


class FakeBatch(dict[str, Any]):
    def to(self, device: str) -> "FakeBatch":
        self["device_seen"] = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> FakeBatch:
        self.messages = messages
        self.template_kwargs = kwargs
        return FakeBatch({"input_ids": [[101, 102, 103]]})

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        self.decoded_tokens = tokens
        self.skip_special_tokens = skip_special_tokens
        return "fixture answer"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return list(text.encode("utf-8"))


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.generation_kwargs = kwargs
        return [[101, 102, 103, 201, 202]]


class TransformersRuntimeTests(unittest.TestCase):
    def test_qwen_prompt_adapter_owns_model_specific_message_shape(self) -> None:
        messages = QwenPromptAdapter().messages_for("What is the answer?")

        self.assertEqual(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the answer?"}],
                }
            ],
            messages,
        )

    def test_qwen_transformers_runtime_generates_without_downloading_a_model(self) -> None:
        processor = FakeProcessor()
        model = FakeModel()

        def loader(model_spec: Any, config: RuntimeConfig) -> tuple[FakeProcessor, FakeModel]:
            self.assertEqual("Qwen/Qwen3.8-27B", model_spec.model_id)
            self.assertEqual("cpu", config.options["device"])
            return processor, model

        runtime = QwenTransformersRuntime(loader=loader)
        model_spec = qwen38_model_spec(
            revision="model-sha",
            tokenizer_revision="tokenizer-sha",
        )
        config = RuntimeConfig(
            name="transformers",
            version="fixture",
            options={"device": "cpu", "torch_dtype": "float16"},
        )

        self.assertIsInstance(runtime, Runtime)
        runtime.load(model_spec, config)
        self.assertIs(runtime.get_tokenizer(), processor)
        response = runtime.generate(
            GenerationRequest(
                prompt="What is the answer?",
                model=model_spec,
                sampling=SamplingConfig(max_new_tokens=8),
            )
        )

        self.assertEqual("fixture answer", response.output_text)
        self.assertEqual(3, response.usage.prompt_tokens)
        self.assertEqual(2, response.usage.completion_tokens)
        self.assertEqual("transformers", response.runtime.runtime_name)
        self.assertEqual("Qwen/Qwen3.8-27B", response.runtime.model_id)
        self.assertEqual("Qwen/Qwen3.8-27B", response.runtime.tokenizer_id)
        self.assertEqual("float16", response.runtime.config["torch_dtype"])
        self.assertEqual(
            [{"role": "user", "content": [{"type": "text", "text": "What is the answer?"}]}],
            processor.messages,
        )
        self.assertEqual(8, model.generation_kwargs["max_new_tokens"])
        runtime.close()


if __name__ == "__main__":
    unittest.main()
