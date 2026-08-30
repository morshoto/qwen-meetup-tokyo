import json
import unittest
from pathlib import Path

import yaml


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
            "lengths: [8192, 32768, 65536, 131072]",
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
        self.assertIn("n_ctx: 131392", config)
        self.assertIn("approx_constant_gap_tolerance: 0.10", config)
        self.assertIn("scorer_version: calibrated.v1", config)

        for text in (readme, results_readme):
            self.assertIn("exp_003", text)
            self.assertIn("llama.cpp", text)
            self.assertIn("fixture", text.lower())
            self.assertIn("not a Qwen measurement", text)
        self.assertIn("execution source of truth", readme)
        self.assertIn("CalibratedAnswerScorer", readme)
        self.assertIn("`calibrated.v1`", readme)
        self.assertIn("selected task IDs", readme)
        self.assertNotIn("explicit legacy exception", readme)

    def test_main_declares_issue_21_matrix(self) -> None:
        protocol = yaml.safe_load(
            (EXPERIMENT / "config.yaml").read_text(encoding="utf-8")
        )
        main = protocol["phases"]["main"]

        self.assertEqual([8192, 32768, 65536, 131072], main["lengths"])
        self.assertEqual(
            [0.05, 0.25, 0.50, 0.75, 0.95],
            main["evidence_positions"],
        )
        self.assertGreater(main["repeats"], 1)
        self.assertTrue({"q8_0", "q4_k_m"}.issubset(protocol["quantization"]["variants"]))
        self.assertEqual("data/tasks/core.v002.jsonl", protocol["experiment"]["task_catalog"])
        self.assertEqual("calibrated.v1", protocol["analysis"]["scorer_version"])

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
            "analyze.py",
            "regenerate",
            "relative-degradation.csv",
            "interaction.json",
            "scorer_version",
            "insufficient_data",
        ):
            self.assertIn(required, notebook)

    def test_notebook_is_portable_measured_artifact(self) -> None:
        notebook_path = EXPERIMENT / "analysis.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        notebook_text = notebook_path.read_text(encoding="utf-8")
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]

        self.assertIn("PHASE = 'main'", notebook_text)
        self.assertIn("RESULTS_DIR = ROOT /", notebook_text)
        self.assertIn("runpy.run_path", notebook_text)
        self.assertIn("allow_fixture=False", notebook_text)
        self.assertNotIn("allow_fixture=True", notebook_text)
        self.assertNotIn("/var/folders/", notebook_text)
        self.assertNotIn("exec(compile", notebook_text)
        for cell in code_cells:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual([], cell["outputs"])


if __name__ == "__main__":
    unittest.main()
