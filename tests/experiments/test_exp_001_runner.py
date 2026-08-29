import csv
import importlib.util
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

from llm_lab.datasets import TaskCatalog
from llm_lab.evaluation import TrialResult, TrialStatus, load_trial_results


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "experiments/exp_001-context_measurement/runner.py"
SPEC = importlib.util.spec_from_file_location("exp_001_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Exp001RunnerTests(unittest.TestCase):
    def test_checked_in_smoke_artifact_covers_expanded_catalog(self) -> None:
        manifest = json.loads(
            (
                ROOT
                / "experiments/exp_001-context_measurement/results/manifests/smoke.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual("data/tasks/core.v002.jsonl", manifest["task_catalog"])
        self.assertEqual(30, len(manifest["task_ids"]))
        self.assertEqual(180, manifest["planned_trial_n"])
        self.assertEqual(180, manifest["actual_trial_n"])
        self.assertEqual(
            180,
            len(
                load_trial_results(
                    ROOT
                    / "experiments/exp_001-context_measurement/results/raw/smoke-trials.jsonl"
                )
            ),
        )
        self.assertTrue(
            all(row["independent_task_n"] == 10 for row in manifest["coverage"])
        )
        self.assertTrue(all(row["status"] == "valid" for row in manifest["coverage"]))
        with (
            ROOT
            / "experiments/exp_001-context_measurement/results/processed/summary.csv"
        ).open(encoding="utf-8", newline="") as summary_file:
            summary_rows = list(csv.DictReader(summary_file))
        self.assertEqual(18, len(summary_rows))
        self.assertTrue(all(row["n"] == "10" for row in summary_rows))

    def test_planned_conditions_match_smoke_and_main_matrix(self) -> None:
        smoke = runner.planned_conditions("smoke")
        main = runner.planned_conditions("main")

        self.assertEqual(6, len(smoke))
        self.assertEqual(20, len(main))
        self.assertEqual("baseline:ctx008192:p005", smoke[0].condition_id)
        self.assertEqual("baseline:ctx131072:p095", main[-1].condition_id)

    def test_build_tasks_records_context_provenance_and_evidence_offsets(self) -> None:
        catalog = TaskCatalog.from_jsonl(ROOT / "data/tasks/core.v002.jsonl")
        condition = runner.planned_conditions("smoke")[0]

        tasks = runner.build_tasks(
            catalog,
            condition,
            fixture_seed=42,
        )

        self.assertEqual(30, len(tasks))
        self.assertEqual("literal_retrieval", tasks[0].task_type)
        request = tasks[0].build_request(
            runner.qwen38_model_spec(),
            runner.SamplingConfig(max_new_tokens=8),
        )
        self.assertEqual(8192, request.metadata["target_context_tokens"])
        self.assertEqual(0.05, request.metadata["requested_evidence_position"])
        self.assertEqual("whitespace-v1", request.metadata["context_tokenization"])
        self.assertEqual(42, request.metadata["fixture_seed"])
        self.assertEqual(
            "aurora-access",
            request.metadata["evidence_spans"][0]["id"],
        )
        self.assertEqual(
            "task.literal.000001:baseline:ctx008192:p005:seed1043",
            request.metadata["context_instance_id"],
        )
        self.assertEqual(64, len(request.metadata["context_sha256"]))
        self.assertLess(
            request.metadata["evidence_spans"][0]["token_start"],
            request.metadata["evidence_spans"][0]["token_end"],
        )

    def test_build_tasks_accepts_the_inference_tokenizer_for_model_runs(self) -> None:
        class FakeTokenizer:
            name = "fixture-inference-tokenizer"

            def encode(self, text: str) -> list[int]:
                return list(text.encode("utf-8"))

            def decode(self, tokens: list[int]) -> str:
                return bytes(tokens).decode("utf-8")

        catalog = TaskCatalog.from_jsonl(ROOT / "data/tasks/core.v002.jsonl")
        condition = runner.planned_conditions("smoke")[0]

        tasks = runner.build_tasks(
            catalog,
            condition,
            fixture_seed=42,
            tokenizer=FakeTokenizer(),
        )

        request = tasks[0].build_request(
            runner.qwen38_model_spec(),
            runner.SamplingConfig(max_new_tokens=8),
        )
        self.assertEqual(
            "fixture-inference-tokenizer",
            request.metadata["context_tokenization"],
        )
        self.assertEqual("tokenizer", request.metadata["context_tokenization_mode"])
        self.assertEqual(8192, request.metadata["actual_context_tokens"])

    def test_fixture_smoke_can_be_regenerated_without_duplicate_trial_ids(self) -> None:
        results_root = ROOT / "experiments/exp_001-context_measurement/results"
        with TemporaryDirectory(dir=results_root) as temporary_directory:
            directory = Path(temporary_directory)
            output_path = directory / "smoke-trials.jsonl"
            manifest_path = directory / "smoke.json"

            first = runner.run_experiment(
                phase="smoke",
                backend="fixture",
                output_path=output_path,
                manifest_path=manifest_path,
                overwrite_smoke=True,
            )
            with self.assertRaises(FileExistsError):
                runner.run_experiment(
                    phase="smoke",
                    backend="fixture",
                    output_path=output_path,
                    manifest_path=manifest_path,
                )
            second = runner.run_experiment(
                phase="smoke",
                backend="fixture",
                output_path=output_path,
                manifest_path=manifest_path,
                overwrite_smoke=True,
            )

            self.assertEqual(180, first["actual_trial_n"])
            self.assertEqual(180, second["actual_trial_n"])
            persisted = load_trial_results(output_path)
            self.assertEqual(180, len(persisted))
            self.assertEqual(180, len({result.trial_id for result in persisted}))
            self.assertTrue(all(result.score["scorer"] == "calibrated.v1" for result in persisted))
            self.assertTrue(all("answer_bearing_correct" in result.score for result in persisted))

    def test_analysis_notebook_requires_calibrated_scorer(self) -> None:
        notebook = (ROOT / "experiments/exp_001-context_measurement/analysis.ipynb").read_text(
            encoding="utf-8"
        )

        self.assertIn("expected_scorer='calibrated.v1'", notebook)

    def test_manifest_records_dimensions_and_exclusion_reason(self) -> None:
        catalog = TaskCatalog.from_jsonl(ROOT / "data/tasks/core.v002.jsonl")
        condition = runner.Condition(8192, 0.05)
        result = TrialResult(
            trial_id="exp_001:task.literal.000001:test:run01",
            experiment_id="exp_001",
            task_id="task.literal.000001",
            status=TrialStatus.OUT_OF_MEMORY,
            input={
                "task_type": "literal_retrieval",
                "condition_id": condition.condition_id,
                "target_context_tokens": condition.target_context_tokens,
                "requested_evidence_position": condition.evidence_position,
            },
            score={"scorer": "calibrated.v1"},
            error={"type": "MemoryError", "message": "simulated"},
        )

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "raw.jsonl"
            output_path.write_text("{}\n", encoding="utf-8")
            manifest = runner._manifest(
                phase="main",
                backend="transformers",
                output_path=output_path,
                manifest_path=Path(directory) / "main.json",
                conditions=(condition,),
                catalog=catalog,
                repeats=1,
                results=(result,),
                fixture_seed=42,
            )

        self.assertEqual([8192], manifest["context_lengths"])
        self.assertEqual([0.05], manifest["evidence_positions"])
        self.assertEqual(["literal_retrieval", "semantic_retrieval", "multi_hop"], manifest["task_types"])
        literal = next(
            row for row in manifest["coverage"] if row["task_type"] == "literal_retrieval"
        )
        self.assertEqual("excluded", literal["status"])
        self.assertIn("scored", literal["exclusion_reason"])

    def test_runner_uses_phase_dimensions_and_repeats_from_config(self) -> None:
        config_text = """
experiment:
  id: exp_001
  fixture_seed: 42
  task_catalog: data/tasks/core.v002.jsonl
model:
  model: Qwen/Qwen3.8-27B
phases:
  smoke:
    lengths: [8192]
    evidence_positions: [0.05]
    repeats: 1
sampling:
  temperature: 0.0
  max_new_tokens: 32
effective_context:
  baseline_length: 8192
  baseline_accuracy_gate: 0.80
  alpha: 0.90
"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(config_text, encoding="utf-8")
            manifest = runner.run_experiment(
                phase="smoke",
                backend="fixture",
                output_path=root / "raw" / "trials.jsonl",
                manifest_path=root / "manifests" / "smoke.json",
                config_path=config_path,
            )

        self.assertEqual(1, manifest["planned_condition_n"])
        self.assertEqual(30, manifest["planned_trial_n"])
        self.assertEqual([8192], manifest["context_lengths"])
        self.assertEqual([0.05], manifest["evidence_positions"])

if __name__ == "__main__":
    unittest.main()
