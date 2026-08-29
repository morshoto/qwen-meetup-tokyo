import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/exp_003-context_x_quantization"


class Exp003ExperimentContractTests(unittest.TestCase):
    def test_config_declares_matched_context_quantization_matrix(self) -> None:
        config = (EXPERIMENT / "config.yaml").read_text(encoding="utf-8")
        readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8")
        results_readme = (EXPERIMENT / "results/README.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("id: exp_003", config)
        self.assertIn("name: context_x_quantization", config)
        self.assertIn(
            "source_manifest: ../exp_002-quantization_llama_cpp_gguf/results/manifest.json",
            config,
        )
        self.assertIn("task_catalog: data/tasks/core.v002.jsonl", config)
        self.assertIn("prompt_id: prompt.qa.v001", config)
        self.assertIn(
            "lengths: [8192, 32768, 65536, 131072, 262144]",
            config,
        )
        self.assertIn(
            "evidence_positions: [0.05, 0.25, 0.50, 0.75, 0.95]",
            config,
        )
        self.assertIn(
            "task_types: [literal_retrieval, semantic_retrieval, multi_hop]",
            config,
        )
        self.assertIn(
            "variants: [q8_0, q6_k, q5_k_m, q4_k_m]",
            config,
        )
        self.assertIn("context_length_semantics: input_tokens", config)
        self.assertIn("n_ctx: 262464", config)
        self.assertIn("approx_constant_gap_tolerance: 0.10", config)

        for text in (readme, results_readme):
            self.assertIn("exp_003", text)
            self.assertIn("llama.cpp", text)
            self.assertIn("fixture", text.lower())
            self.assertIn("not a Qwen measurement", text)

    def test_notebook_contains_required_interaction_sections(self) -> None:
        notebook = (EXPERIMENT / "analysis.ipynb").read_text(encoding="utf-8")

        for required in (
            "PHASE = 'smoke'",
            "Change PHASE to 'pilot' or 'main'",
            "RAW_PATH",
            "SUMMARY_PATH",
            "MANIFEST_PATH",
            "FileNotFoundError",
            "aggregate_jsonl",
            "matched_cell_rows",
            "relative_degradation_rows",
            "interaction_report",
            "effective_context_by_variant_and_task",
            "context × quantization",
            "position × context",
            "quantization_gap",
            "accuracy_degradation",
            "docs/findings.md",
            "not yet measured",
        ):
            self.assertIn(required, notebook)


if __name__ == "__main__":
    unittest.main()
