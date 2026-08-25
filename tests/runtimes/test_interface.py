import unittest

from llm_lab.generation import (
    GenerationRequest,
    GenerationResponse,
    RuntimeMetadata,
    TokenUsage,
)
from llm_lab.models import ModelSpec
from llm_lab.runtimes import Runtime, RuntimeConfig


class FixtureRuntime:
    name = "fixture"

    def __init__(self) -> None:
        self.loaded: tuple[ModelSpec, RuntimeConfig] | None = None
        self.closed = False

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        self.loaded = (model, config)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.loaded is None:
            raise RuntimeError("runtime must be loaded before generation")
        return GenerationResponse(
            output_text="fixture answer",
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version=self.loaded[1].version,
                model_id=request.model.model_id,
                tokenizer_id=request.model.tokenizer_id,
                config=self.loaded[1].options,
            ),
        )

    def close(self) -> None:
        self.closed = True


class RuntimeInterfaceTests(unittest.TestCase):
    def test_fixture_prompt_runs_through_the_common_runtime_contract(self) -> None:
        runtime = FixtureRuntime()
        model = ModelSpec(model_id="fixture/model", tokenizer_id="fixture/tokenizer")
        config = RuntimeConfig(
            name="fixture",
            version="0.1",
            options={"device": "cpu", "seed": 7},
        )

        self.assertIsInstance(runtime, Runtime)
        runtime.load(model, config)
        response = runtime.generate(GenerationRequest(prompt="Say hello.", model=model))
        runtime.close()

        self.assertEqual("fixture answer", response.output_text)
        self.assertEqual("fixture/tokenizer", response.runtime.tokenizer_id)
        self.assertEqual({"device": "cpu", "seed": 7}, response.runtime.config)
        self.assertTrue(runtime.closed)

    def test_runtime_config_rejects_an_empty_name(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeConfig(name="")


if __name__ == "__main__":
    unittest.main()
