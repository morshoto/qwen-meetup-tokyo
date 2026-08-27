import re
import unittest
from pathlib import Path


EXPERIMENT = Path("experiments/exp_002-quantization_llama_cpp_gguf")


class QuantizationExperimentContractTests(unittest.TestCase):
    def test_experiment_config_names_llama_cpp_gguf_and_fixed_conditions(self) -> None:
        config = (EXPERIMENT / "config.yaml").read_text(encoding="utf-8")
        readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("approach_a", config)
        self.assertIn("name: quantization_llama_cpp_gguf", config)
        self.assertIn("approach: llama.cpp", config)
        self.assertIn("format: GGUF", config)
        self.assertIn("binding: llama-cpp-python", config)
        self.assertIn("prompt.qa.v001", config)
        self.assertIn("data/tasks/core.v001.jsonl", config)
        self.assertIn("context_lengths: [8192, 32768]", config)
        self.assertIn("context_length_semantics: input_tokens", config)
        self.assertIn("context_overhead_tokens: 256", config)
        n_ctx = int(re.search(r"n_ctx:\s*(\d+)", config).group(1))
        max_context = max(
            int(value)
            for value in re.search(
                r"context_lengths:\s*\[([^]]+)\]", config
            ).group(1).split(",")
        )
        max_new_tokens = int(re.search(r"max_new_tokens:\s*(\d+)", config).group(1))
        margin = int(re.search(r"context_overhead_tokens:\s*(\d+)", config).group(1))
        self.assertGreaterEqual(n_ctx, max_context + max_new_tokens + margin)
        for quantization_type in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"):
            self.assertIn(quantization_type, config)

        self.assertIn("llama.cpp", readme)
        self.assertIn("GGUF", readme)
        self.assertIn("kernel", readme.lower())
        self.assertIn("sha-256", readme.lower())

    def test_resolved_manifest_template_declares_provenance_for_each_variant(self) -> None:
        template = (EXPERIMENT / "manifest.template.json").read_text(encoding="utf-8")

        self.assertIn('"source_revision"', template)
        self.assertIn('"conversion_command"', template)
        self.assertIn('"artifact_sha256"', template)
        self.assertIn('"artifact_size_bytes"', template)
        self.assertIn('"context_length_semantics": "input_tokens"', template)
        self.assertIn('"context_overhead_tokens": 256', template)
        for condition_id in ("q8_0", "q6_k", "q5_k_m", "q4_k_m"):
            self.assertIn(f'"condition_id": "{condition_id}"', template)


if __name__ == "__main__":
    unittest.main()
