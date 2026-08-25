import unittest
from typing import Any

from llm_lab.generation import GenerationRequest, SamplingConfig
from llm_lab.models import ModelSpec
from llm_lab.runtimes import Runtime, RuntimeConfig
from llm_lab.runtimes.llama_cpp import LlamaCppRuntime


class FakeLlamaClient:
    def __init__(self) -> None:
        self.completion_kwargs: dict[str, Any] | None = None
        self.closed = False

    def create_completion(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.completion_kwargs = kwargs
        return [
            {"choices": [{"text": "fixture"}]},
            {
                "choices": [{"text": " answer", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        ]

    def tokenize(self, value: bytes, *, add_bos: bool) -> list[int]:
        return list(range(3 if add_bos else 2))

    def close(self) -> None:
        self.closed = True


class LlamaCppRuntimeTests(unittest.TestCase):
    def test_streamed_generation_records_measurement_fields(self) -> None:
        client = FakeLlamaClient()

        def loader(model: ModelSpec, config: RuntimeConfig) -> FakeLlamaClient:
            self.assertEqual("Qwen/Qwen3.8-27B", model.model_id)
            self.assertEqual("/models/q8_0.gguf", config.options["model_path"])
            return client

        runtime = LlamaCppRuntime(loader=loader)
        model = ModelSpec(model_id="Qwen/Qwen3.8-27B", tokenizer_id="Qwen/Qwen3.8-27B")
        runtime.load(
            model,
            RuntimeConfig(
                name="llama.cpp",
                version="fixture",
                options={
                    "model_path": "/models/q8_0.gguf",
                    "n_ctx": 32768,
                    "n_gpu_layers": -1,
                },
            ),
        )

        response = runtime.generate(
            GenerationRequest(
                prompt="What is the answer?",
                model=model,
                sampling=SamplingConfig(
                    max_new_tokens=8,
                    temperature=0.0,
                    top_p=0.9,
                    top_k=20,
                    seed=7,
                ),
            )
        )

        self.assertIsInstance(runtime, Runtime)
        self.assertEqual("fixture answer", response.output_text)
        self.assertEqual(3, response.usage.prompt_tokens)
        self.assertEqual(2, response.usage.completion_tokens)
        self.assertEqual("llama.cpp", response.runtime.runtime_name)
        self.assertEqual("/models/q8_0.gguf", response.runtime.config["model_path"])
        self.assertEqual("first_stream_chunk", response.runtime.config["timing_source"])
        self.assertIsNotNone(response.timing.ttft_seconds)
        self.assertIsNotNone(response.timing.prefill_seconds)
        self.assertIsNotNone(response.timing.decode_seconds)
        self.assertGreaterEqual(response.timing.total_seconds or 0.0, 0.0)
        self.assertEqual(
            {
                "prompt": "What is the answer?",
                "stream": True,
                "max_tokens": 8,
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": 20,
                "seed": 7,
            },
            client.completion_kwargs,
        )

        runtime.close()
        self.assertTrue(client.closed)

    def test_load_requires_a_model_path(self) -> None:
        runtime = LlamaCppRuntime(loader=lambda model, config: object())

        with self.assertRaisesRegex(ValueError, "model_path"):
            runtime.load(
                ModelSpec(model_id="fixture/model"),
                RuntimeConfig(name="llama.cpp", options={}),
            )


if __name__ == "__main__":
    unittest.main()
