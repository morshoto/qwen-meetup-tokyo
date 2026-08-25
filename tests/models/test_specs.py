import unittest

from llm_lab.generation import GenerationRequest, GenerationResponse, SamplingConfig
from llm_lab.models import ModelCapabilities, ModelSpec, qwen38_model_spec
from llm_lab.generation.types import GenerationTiming, RuntimeMetadata, TokenUsage


class ModelAndGenerationTypeTests(unittest.TestCase):
    def test_qwen_spec_keeps_model_and_tokenizer_metadata_separate(self) -> None:
        spec = qwen38_model_spec(
            revision="model-sha",
            tokenizer_revision="tokenizer-sha",
        )

        self.assertEqual("Qwen/Qwen3.8-27B", spec.model_id)
        self.assertEqual("Qwen/Qwen3.8-27B", spec.tokenizer_id)
        self.assertEqual("model-sha", spec.revision)
        self.assertEqual("tokenizer-sha", spec.tokenizer_revision)
        self.assertTrue(spec.capabilities.supports_vision)
        self.assertTrue(spec.capabilities.supports_thinking)

    def test_request_and_response_expose_backend_neutral_metadata(self) -> None:
        spec = ModelSpec(
            model_id="fixture/model",
            tokenizer_id="fixture/tokenizer",
            capabilities=ModelCapabilities(max_context_tokens=128),
        )
        request = GenerationRequest(
            prompt="Answer with one word.",
            model=spec,
            sampling=SamplingConfig(max_new_tokens=8, seed=7),
        )
        response = GenerationResponse(
            output_text="okay",
            usage=TokenUsage(prompt_tokens=4, completion_tokens=1),
            timing=GenerationTiming(total_seconds=0.25),
            runtime=RuntimeMetadata(
                runtime_name="fixture",
                runtime_version="1.0",
                model_id=spec.model_id,
                tokenizer_id=spec.tokenizer_id,
                config={"device": "cpu"},
            ),
        )

        self.assertEqual("Answer with one word.", request.prompt)
        self.assertEqual(5, response.usage.total_tokens)
        self.assertEqual(0.25, response.timing.total_seconds)
        self.assertEqual("fixture/tokenizer", response.runtime.tokenizer_id)
        self.assertEqual({"device": "cpu"}, response.runtime.config)

    def test_sampling_config_rejects_invalid_generation_limits(self) -> None:
        with self.assertRaises(ValueError):
            SamplingConfig(max_new_tokens=0)
        with self.assertRaises(ValueError):
            SamplingConfig(temperature=-0.1)
        with self.assertRaises(ValueError):
            SamplingConfig(top_p=1.1)


if __name__ == "__main__":
    unittest.main()
