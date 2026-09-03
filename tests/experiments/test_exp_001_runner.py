import csv
import hashlib
import importlib.util
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from unittest.mock import patch

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
    def test_llama_cpp_options_verifies_and_records_reference_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "q8_0.gguf"
            artifact.write_bytes(b"q8 fixture artifact")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            size = artifact.stat().st_size
            options = runner._llama_cpp_options(
                {
                    "runtime": {
                        "llama_cpp": {
                            "model_path": str(artifact),
                            "artifact_sha256": digest,
                            "artifact_size_bytes": size,
                            "version": "llama-test",
                            "n_ctx": 8192,
                            "n_batch": 128,
                            "n_gpu_layers": 0,
                            "flash_attn": False,
                            "verbose": True,
                        }
                    }
                }
            )

        self.assertEqual(str(artifact), options["model_path"])
        self.assertEqual(str(artifact), options["_artifact_uri"])
        self.assertEqual("llama-test", options["version"])
        self.assertEqual(digest, options["_artifact_sha256"])
        self.assertEqual(size, options["_artifact_size_bytes"])
        self.assertEqual(8192, options["n_ctx"])
        self.assertFalse(options["flash_attn"])

    def test_llama_cpp_options_rejects_reference_artifact_hash_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "q8_0.gguf"
            artifact.write_bytes(b"q8 fixture artifact")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                runner._llama_cpp_options(
                    {
                        "runtime": {
                            "llama_cpp": {
                                "model_path": str(artifact),
                                "artifact_sha256": "0" * 64,
                            }
                        }
                    }
                )

    def test_llama_cpp_options_scales_context_to_selected_phase(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "q8_0.gguf"
            artifact.write_bytes(b"q8 fixture artifact")
            options = runner._llama_cpp_options(
                {
                    "context": {"lengths": [8192, 131072]},
                    "runtime": {
                        "llama_cpp": {
                            "model_path": str(artifact),
                            "n_ctx": 131392,
                        }
                    },
                },
                max_context_tokens=32768,
            )

        self.assertEqual(33088, options["n_ctx"])

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
        feasibility = runner.planned_conditions("feasibility")
        self.assertEqual(3, len(feasibility))
        self.assertEqual("feasibility:ctx262144:p050", feasibility[-1].condition_id)

    def test_feasibility_phase_declares_lengths_timeout_and_shared_tasks(self) -> None:
        feasibility = runner.load_config()["phases"]["feasibility"]

        self.assertEqual([65536, 131072, 262144], feasibility["lengths"])
        self.assertEqual([0.50], feasibility["evidence_positions"])
        self.assertEqual(1, feasibility["repeats"])
        self.assertEqual(900, feasibility["timeout_seconds"])
        self.assertEqual(
            [
                "task.literal.000001",
                "task.semantic.000001",
                "task.multihop.000001",
            ],
            feasibility["task_ids"],
        )

    def test_feasibility_task_selection_is_explicit_and_bounded(self) -> None:
        catalog = TaskCatalog.from_jsonl(ROOT / "data/tasks/core.v002.jsonl")
        selected = runner.select_probe_tasks(
            catalog,
            [
                "task.literal.000001",
                "task.semantic.000001",
                "task.multihop.000001",
            ],
        )

        self.assertEqual(
            [
                "task.literal.000001",
                "task.semantic.000001",
                "task.multihop.000001",
            ],
            [task.task_id for task in selected.tasks],
        )
        with self.assertRaisesRegex(ValueError, "between one and three"):
            runner.select_probe_tasks(catalog, [])
        with self.assertRaisesRegex(ValueError, "between one and three"):
            runner.select_probe_tasks(catalog, list(catalog.ids[:4]))

    def test_feasibility_trial_preserves_timeout_metrics_and_record_hash(self) -> None:
        catalog = TaskCatalog.from_jsonl(ROOT / "data/tasks/core.v002.jsonl")
        definition = catalog.get("task.literal.000001")
        condition = runner.planned_conditions("feasibility")[0]
        result = runner._trial_from_probe(
            runner.ProbeOutcome(
                value=None,
                timed_out=True,
                exit_code=-15,
                peak_memory_bytes=123456,
                memory_measurement="psutil.child_rss_sampled",
                termination_reason="timeout",
                error={"type": "TimeoutError", "message": "simulated"},
            ),
            definition=definition,
            condition=condition,
            model=runner.qwen38_model_spec(),
            runtime_version="llama-test",
            runtime_options={"n_ctx": 262404},
            sampling=runner.SamplingConfig(max_new_tokens=64),
            timeout_seconds=900,
            fixture_seed=42,
        )

        self.assertEqual(TrialStatus.TIMEOUT, result.status)
        self.assertEqual(900, result.input["feasibility_probe"]["timeout_seconds"])
        self.assertEqual(-15, result.runtime["config"]["probe_exit_code"])
        self.assertEqual(123456, result.memory["rss_peak_bytes"])
        self.assertIsNone(result.timing["ttft_s"])
        self.assertEqual("calibrated.v1", result.score["scorer"])
        self.assertEqual(
            64,
            len(result.input["provenance"]["raw_record_sha256"]),
        )

    def test_feasibility_run_writes_manifest_and_all_probe_trials(self) -> None:
        def fake_probe(worker, payload, *, timeout_seconds):
            del worker, timeout_seconds
            task = payload["task_definition"]
            condition = payload["condition"]
            length = int(condition["target_context_tokens"])
            position = float(condition["evidence_position"])
            task_id = str(task["id"])
            task_type = str(task["type"])
            trial_result = TrialResult(
                trial_id=(
                    f"exp_001:{task_id}:feasibility:ctx{length:06d}:"
                    "p050:run01"
                ),
                experiment_id="exp_001",
                task_id=task_id,
                status=TrialStatus.COMPLETED,
                input={
                    "task_type": task_type,
                    "condition_id": f"feasibility:ctx{length:06d}:p050",
                    "target_context_tokens": length,
                    "requested_evidence_position": position,
                    "sampling": payload["sampling"],
                },
                score={
                    "correct": True,
                    "answer_bearing_correct": True,
                    "exact_correct": True,
                    "format_valid": True,
                    "scorer": "calibrated.v1",
                },
                timing={"ttft_s": 0.1},
            )
            return runner.ProbeOutcome(
                value=trial_result.to_record(),
                timed_out=False,
                exit_code=0,
                peak_memory_bytes=100,
                memory_measurement="test",
                termination_reason=None,
            )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            options = {
                "model_path": str(root / "q8_0.gguf"),
                "version": "llama-test",
                "n_ctx": 262404,
                "n_batch": 128,
                "n_gpu_layers": 0,
                "flash_attn": False,
                "verbose": False,
                "_artifact_uri": "artifacts/q8_0.gguf",
                "_artifact_sha256": "a" * 64,
                "_artifact_size_bytes": 1,
            }
            with patch.object(runner, "_llama_cpp_options", return_value=options):
                with patch.object(runner, "run_isolated_probe", side_effect=fake_probe):
                    manifest = runner.run_experiment(
                        phase="feasibility",
                        backend="llama.cpp",
                        output_path=root / "raw" / "feasibility.jsonl",
                        manifest_path=root / "manifests" / "feasibility.json",
                        config_path=ROOT / "experiments/exp_001-context_measurement/config.yaml",
                    )

            self.assertEqual(9, manifest["planned_trial_n"])
            self.assertEqual(9, manifest["actual_trial_n"])
            self.assertEqual(900, manifest["probe"]["timeout_seconds"])
            self.assertEqual([65536, 131072, 262144], manifest["context_lengths"])
            self.assertEqual(
                ["accepted_and_useful"] * 3,
                [row["classification"] for row in manifest["feasibility"]["classifications"]],
            )
            persisted = load_trial_results(root / "raw" / "feasibility.jsonl")
            self.assertEqual(9, len(persisted))
            self.assertEqual(
                manifest["context_provenance"]["config_sha256"],
                persisted[0].input["provenance"]["config_sha256"],
            )
            self.assertTrue(manifest["raw_results_sha256"])

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

    def test_fixture_backend_is_rejected_outside_smoke_phase(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "fixture backend is harness-only"):
                runner.run_experiment(
                    phase="main",
                    backend="fixture",
                    output_path=root / "raw" / "trials.jsonl",
                    manifest_path=root / "manifests" / "main.json",
                    config_path=ROOT / "experiments/exp_001-context_measurement/config.yaml",
                )

    def test_llama_cpp_model_run_writes_resume_checkpoint_before_first_trial(self) -> None:
        class ByteTokenizer:
            name = "checkpoint-test-tokenizer"

            def encode(self, text: str) -> list[int]:
                return list(text.encode("utf-8"))

            def decode(self, tokens: list[int]) -> str:
                return bytes(tokens).decode("utf-8")

        class FakeLlamaRuntime:
            name = "llama.cpp"

            def load(self, model, config) -> None:
                del model, config

            def get_tokenizer(self) -> ByteTokenizer:
                return ByteTokenizer()

            def close(self) -> None:
                return None

        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "raw" / "smoke-trials.jsonl"
            manifest_path = root / "manifests" / "smoke.json"
            llama_options = {
                "model_path": str(root / "q8_0.gguf"),
                "version": "llama-test",
                "n_ctx": 131392,
                "n_batch": 128,
                "n_gpu_layers": -1,
                "flash_attn": True,
                "verbose": False,
                "_artifact_uri": "artifacts/q8_0.gguf",
                "_artifact_sha256": "a" * 64,
                "_artifact_size_bytes": 1,
            }

            with (
                patch.object(runner, "LlamaCppRuntime", FakeLlamaRuntime),
                patch.object(runner, "_llama_cpp_options", return_value=llama_options),
                patch.object(
                    runner,
                    "EvaluationRunner",
                    side_effect=RuntimeError("checkpoint sentinel"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "checkpoint sentinel"):
                    runner.run_experiment(
                        phase="smoke",
                        backend="llama.cpp",
                        output_path=output_path,
                        manifest_path=manifest_path,
                        config_path=ROOT / "experiments/exp_001-context_measurement/config.yaml",
                    )

            checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("in_progress", checkpoint["status"])
            self.assertEqual("llama.cpp", checkpoint["backend"])
            self.assertEqual(str(output_path), checkpoint["raw_results"])
            self.assertEqual(
                "configured-seed",
                checkpoint["sampling"]["generation_seed_policy"],
            )
            self.assertEqual(42, checkpoint["sampling"]["seed"])

    def test_analysis_notebook_requires_calibrated_scorer(self) -> None:
        notebook = (ROOT / "experiments/exp_001-context_measurement/analysis.ipynb").read_text(
            encoding="utf-8"
        )

        self.assertIn("expected_scorer='calibrated.v1'", notebook)
        self.assertIn("EXP001_PHASE=feasibility", notebook)
        self.assertIn("feasibility_rows", notebook)

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
  generation_seed: record-at-run-time
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
            persisted = load_trial_results(root / "raw" / "trials.jsonl")

        self.assertEqual(1, manifest["planned_condition_n"])
        self.assertEqual(30, manifest["planned_trial_n"])
        self.assertEqual([8192], manifest["context_lengths"])
        self.assertEqual([0.05], manifest["evidence_positions"])
        self.assertEqual("Qwen/Qwen3.8-27B", manifest["model"]["id"])
        self.assertIsNone(manifest["model"]["revision"])
        self.assertIsNone(manifest["sampling"]["seed"])
        self.assertEqual(42, manifest["context_provenance"]["fixture_seed"])
        self.assertTrue(manifest["context_provenance"]["source_revision"])
        self.assertEqual(
            [
                {
                    "condition_id": "baseline:ctx008192:p005",
                    "target_context_tokens": 8192,
                    "evidence_position": 0.05,
                }
            ],
            manifest["context_provenance"]["conditions"],
        )
        self.assertEqual(
            "greedy-decoding-no-seed",
            manifest["sampling"]["generation_seed_policy"],
        )
        self.assertEqual(
            "greedy-decoding-no-seed",
            persisted[0].input["sampling"]["generation_seed_policy"],
        )

    def test_sampling_resume_reuses_and_validates_resolved_seed(self) -> None:
        config = {
            "temperature": 0.7,
            "max_new_tokens": 32,
            "generation_seed": "record-at-run-time",
        }
        first, first_policy = runner._sampling_from_config(config)
        manifest = {
            "sampling": {
                **first.to_record(),
                "generation_seed_policy": first_policy,
            }
        }

        resumed, resumed_policy = runner._sampling_from_config(
            config,
            resume_manifest=manifest,
        )

        self.assertEqual(first.seed, resumed.seed)
        self.assertEqual("run-resolved-seed", resumed_policy)
        mismatched_config = {**config, "max_new_tokens": 64}
        with self.assertRaisesRegex(ValueError, "sampling provenance mismatch"):
            runner._sampling_from_config(mismatched_config, resume_manifest=manifest)

    def test_context_provenance_requires_source_revision(self) -> None:
        config_path = ROOT / "experiments/exp_001-context_measurement/config.yaml"
        catalog_path = ROOT / "data/tasks/core.v002.jsonl"
        catalog = TaskCatalog.from_jsonl(catalog_path)

        with patch.object(runner, "capture_environment", return_value={"git_sha": None}):
            with self.assertRaisesRegex(RuntimeError, "repository revision"):
                runner._context_provenance(
                    config_path=config_path,
                    catalog_path=catalog_path,
                    catalog=catalog,
                    conditions=runner.planned_conditions("smoke"),
                    repeats=1,
                    fixture_seed=42,
                )

    def test_resume_checkpoint_persists_resolved_sampling(self) -> None:
        with TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "main.json"
            output_path = Path(directory) / "main-trials.jsonl"
            sampling = runner.SamplingConfig(
                max_new_tokens=32,
                temperature=0.7,
                seed=1234,
            )

            runner._write_resume_checkpoint(
                manifest_path=manifest_path,
                phase="main",
                backend="transformers",
                model=runner.qwen38_model_spec(
                    revision="model-commit",
                    tokenizer_revision="tokenizer-commit",
                ),
                output_path=output_path,
                sampling=sampling,
                generation_seed_policy="run-resolved-seed",
                context_provenance={"fixture_seed": 42, "conditions": []},
            )

            checkpoint = runner._load_resume_manifest(manifest_path)

        self.assertEqual("in_progress", checkpoint["status"])
        self.assertEqual(1234, checkpoint["sampling"]["seed"])
        self.assertEqual(
            "run-resolved-seed",
            checkpoint["sampling"]["generation_seed_policy"],
        )
        self.assertEqual(42, checkpoint["context_provenance"]["fixture_seed"])

    def test_resume_checkpoint_requires_matching_in_progress_run_identity(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "main-trials.jsonl"
            model = runner.qwen38_model_spec(
                revision="model-commit",
                tokenizer_revision="tokenizer-commit",
            )
            checkpoint = {
                "schema_version": 1,
                "experiment_id": "exp_001",
                "phase": "main",
                "backend": "transformers",
                "status": "in_progress",
                "model": runner._model_record(model),
                "raw_results": str(output_path),
                "context_provenance": {
                    "fixture_seed": 42,
                    "source_revision": "source-commit",
                    "config_path": "config.yaml",
                    "config_sha256": "config-hash",
                    "task_catalog": "tasks.jsonl",
                    "task_catalog_sha256": "catalog-hash",
                    "task_ids": ["task.literal.000001"],
                    "task_types": ["literal_retrieval"],
                    "conditions": [
                        {
                            "condition_id": "baseline:ctx008192:p005",
                            "target_context_tokens": 8192,
                            "evidence_position": 0.05,
                        }
                    ],
                    "repeats": 1,
                },
            }

            runner._validate_resume_checkpoint(
                checkpoint,
                phase="main",
                backend="transformers",
                output_path=output_path,
                model=model,
                context_provenance=checkpoint["context_provenance"],
            )

            invalid_checkpoints = (
                ("status", "completed"),
                ("phase", "smoke"),
                ("backend", "fixture"),
                ("raw_results", str(root / "other-trials.jsonl")),
                (
                    "model",
                    {
                        **runner._model_record(model),
                        "tokenizer_revision": "other-tokenizer-commit",
                    },
                ),
                (
                    "context_provenance",
                    {
                        **checkpoint["context_provenance"],
                        "fixture_seed": 43,
                    },
                ),
                (
                    "context_provenance.source_revision",
                    {
                        **checkpoint["context_provenance"],
                        "source_revision": "other-source-commit",
                    },
                ),
            )
            for field, value in invalid_checkpoints:
                with self.subTest(field=field):
                    invalid = dict(checkpoint)
                    if field.startswith("context_provenance."):
                        invalid["context_provenance"] = value
                    else:
                        invalid[field] = value
                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        runner._validate_resume_checkpoint(
                            invalid,
                            phase="main",
                            backend="transformers",
                            output_path=output_path,
                            model=model,
                            context_provenance=checkpoint["context_provenance"],
                        )

if __name__ == "__main__":
    unittest.main()
