import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/exp_001-context_measurement"


class Exp001ExperimentContractTests(unittest.TestCase):
    def test_config_declares_issue_23_real_model_matrix_and_outputs(self) -> None:
        config = (EXPERIMENT / "config.yaml").read_text(encoding="utf-8")

        self.assertIn("id: exp_001", config)
        self.assertIn("task_catalog: data/tasks/core.v002.jsonl", config)
        self.assertIn(
            "lengths: [8192, 32768, 65536, 131072]",
            config,
        )
        self.assertIn(
            "evidence_positions: [0.05, 0.25, 0.50, 0.75, 0.95]",
            config,
        )
        self.assertIn("baseline_accuracy_gate: 0.80", config)
        self.assertIn("position_gap", config)
        self.assertIn("effective_context", config)
        self.assertIn("reference_backend: llama.cpp", config)
        self.assertIn("artifact_sha256:", config)

    def test_results_contract_names_regenerable_baseline_outputs(self) -> None:
        results_readme = (EXPERIMENT / "results/README.md").read_text(
            encoding="utf-8"
        )

        for required in (
            "raw/",
            "processed/",
            "manifests/",
            "position-gap",
            "effective-context",
            "analyze.py",
            "not a Qwen measurement",
        ):
            self.assertIn(required, results_readme)

    def test_notebook_is_measured_only_and_saves_required_figures(self) -> None:
        notebook = (EXPERIMENT / "analysis.ipynb").read_text(encoding="utf-8")
        readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8")

        for required in (
            "regenerate",
            "allow_fixture",
            "summary_rows = regeneration['summary_rows']",
            "available_summary_rows",
            "baseline_context_tokens=baseline_length",
            "minimum_baseline_accuracy=baseline_gate",
            "alpha=alpha",
            "position_gap_rows",
            "position-gap-vs-context.png",
            "effective-context-vs-context.png",
            "savefig",
            "baseline-limited",
            "--resume",
            "real-model",
            "llama.cpp",
            "capability_repeats",
        ):
            self.assertIn(required, notebook + readme)


if __name__ == "__main__":
    unittest.main()
