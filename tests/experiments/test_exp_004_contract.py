import json
import unittest
from pathlib import Path


EXPERIMENT = Path("experiments/exp_004-agent_context_growth")


class Exp004ContractTests(unittest.TestCase):
    def test_config_declares_agent_growth_matrix_and_provenance_controls(self) -> None:
        config_text = (EXPERIMENT / "config.yaml").read_text(encoding="utf-8")
        readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8")

        config = json.loads(config_text)
        self.assertEqual("exp_004", config["experiment"]["id"])
        self.assertEqual("agent_context_growth", config["experiment"]["name"])
        self.assertEqual(
            "experiments/exp_003-context_x_quantization/results/manifest.json",
            config["experiment"]["source_manifest"],
        )
        self.assertEqual(
            "data/tasks/agent.v002.jsonl",
            config["experiment"]["task_catalog"],
        )
        self.assertEqual(["q8_0", "q4_k_m"], config["quantization"]["variants"])
        self.assertEqual([4, 8, 16, 32], config["trajectory"]["lengths"])
        self.assertEqual(
            [0.05, 0.25, 0.5, 0.75, 0.95],
            config["trajectory"]["critical_positions"],
        )
        self.assertIn("deterministic", readme)
        self.assertIn("fixture", readme.lower())
        self.assertIn("provenance", readme.lower())
        self.assertIn("exp_003", readme)

    def test_results_contract_and_notebook_are_measured_data_only(self) -> None:
        results_readme = (EXPERIMENT / "results" / "README.md").read_text(
            encoding="utf-8"
        )
        notebook = json.loads(
            (EXPERIMENT / "analysis.ipynb").read_text(encoding="utf-8")
        )
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        for directory in ("raw", "processed", "manifests", "figures"):
            self.assertIn(directory, results_readme)
        for required in (
            "require_measured_trials",
            "fixture_only",
            "validate_complete_matrix",
            "aggregate_agent_trials",
            "plot_reliability_by_length",
            "plot_reliability_by_position",
            "final_task_success",
            "failure_category_counts",
            "trajectory_context_tokens",
        ):
            self.assertIn(required, source)
        self.assertIn("source_manifest", source)
        self.assertIn("processed", source)


if __name__ == "__main__":
    unittest.main()
