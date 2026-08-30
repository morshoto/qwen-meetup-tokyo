import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from llm_lab.evaluation import TrialResult, TrialStatus, make_trial_id
from llm_lab.evaluation.storage import JsonlResultWriter


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_PATH = ROOT / "experiments/exp_003-context_x_quantization/analyze.py"
SPEC = importlib.util.spec_from_file_location("exp_003_analysis", ANALYSIS_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


TASK_IDS = ("task.literal.000001", "task.literal.000002")
VARIANTS = ("q8_0", "q4_k_m")
CONTEXTS = (8192, 32768)
POSITIONS = (0.50,)
CATALOG_PATH = ROOT / "data/tasks/core.v002.jsonl"
CATALOG_SHA256 = hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()


def trial(
    task_id: str,
    variant: str,
    context_tokens: int,
    correct: bool,
) -> TrialResult:
    condition_id = f"{variant}:ctx{context_tokens}:p050"
    instance_id = f"{task_id}:ctx{context_tokens}:p050"
    return TrialResult(
        trial_id=make_trial_id(
            "exp_003",
            task_id,
            condition_id=condition_id,
        ),
        experiment_id="exp_003",
        task_id=task_id,
        status=TrialStatus.COMPLETED,
        input={
            "task_type": "literal_retrieval",
            "condition_id": condition_id,
            "variant_condition_id": variant,
            "variant_label": variant,
            "quantization_type": variant.upper(),
            "target_context_tokens": context_tokens,
            "requested_evidence_position": 0.50,
            "actual_evidence_position": 0.50,
            "context_instance_id": instance_id,
            "context_sha256": hashlib.sha256(instance_id.encode()).hexdigest(),
            "task_catalog": "data/tasks/core.v002.jsonl",
            "task_catalog_sha256": CATALOG_SHA256,
            "scorer_version": "calibrated.v1",
        },
        score={
            "correct": correct,
            "value": 1.0 if correct else 0.0,
            "scorer": "calibrated.v1",
            "exact_correct": correct,
            "answer_bearing_correct": correct,
            "format_valid": True,
        },
    )


def write_manifest(root: Path, raw_path: Path, *, backend: str = "llama.cpp") -> Path:
    source_manifest = ROOT / "experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json"
    coverage = [
        {
            "variant_condition_id": variant,
            "task_type": "literal_retrieval",
            "condition_id": f"{variant}:ctx{context_tokens}:p050",
            "target_context_tokens": context_tokens,
            "requested_evidence_position": 0.50,
            "trial_n": 2,
            "scored_n": 2,
            "independent_task_n": 2,
            "expected_trial_n": 2,
            "status": "valid",
            "exclusion_reason": None,
        }
        for variant in VARIANTS
        for context_tokens in CONTEXTS
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": "exp_003",
        "phase": "main",
        "backend": backend,
        "scorer_version": "calibrated.v1",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "task_catalog": "data/tasks/core.v002.jsonl",
        "task_catalog_sha256": CATALOG_SHA256,
        "task_ids": list(TASK_IDS),
        "task_types": ["literal_retrieval"],
        "quantization_variants": [
            {"condition_id": variant} for variant in VARIANTS
        ],
        "context_lengths": list(CONTEXTS),
        "evidence_positions": list(POSITIONS),
        "repeats": 1,
        "planned_cell_n": len(coverage),
        "planned_trial_n": 8,
        "actual_trial_n": 8,
        "raw_results": str(raw_path),
        "raw_results_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "coverage": coverage,
        "effective_context": {
            "baseline_length": 8192,
            "baseline_accuracy_gate": 0.80,
            "alpha": 0.90,
        },
        "analysis": {
            "primary_gap_reference": "q8_0",
            "approx_constant_gap_tolerance": 0.10,
        },
    }
    path = root / "manifests" / "main.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class Exp003AnalysisTests(unittest.TestCase):
    def _write_raw(self, root: Path) -> Path:
        raw_path = root / "raw" / "main-trials.jsonl"
        writer = JsonlResultWriter(raw_path)
        for task_id in TASK_IDS:
            for variant in VARIANTS:
                for context_tokens in CONTEXTS:
                    writer.append(
                        trial(
                            task_id,
                            variant,
                            context_tokens,
                            correct=variant == "q8_0" or context_tokens == 8192,
                        )
                    )
        return raw_path

    def test_regeneration_writes_task_level_interaction_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = self._write_raw(root)
            manifest_path = write_manifest(root, raw_path)

            result = analysis.regenerate(manifest_path)

            self.assertEqual(8, result["trial_n"])
            self.assertEqual(8, result["summary_row_n"])
            self.assertEqual(1, len(result["interaction_reports"]))
            self.assertEqual(
                "context_dependent",
                result["interaction_reports"][0]["classification"],
            )
            self.assertTrue((root / "processed/summary.csv").is_file())
            self.assertTrue((root / "processed/relative-degradation.csv").is_file())
            self.assertTrue((root / "processed/interaction.json").is_file())
            self.assertTrue((root / "processed/effective-context.json").is_file())
            summary = (root / "processed/summary.csv").read_text(encoding="utf-8")
            self.assertIn("task_id", summary)
            self.assertIn("scorer_version", summary)

    def test_regeneration_rejects_fixture_without_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = self._write_raw(root)
            manifest_path = write_manifest(root, raw_path, backend="fixture")

            with self.assertRaisesRegex(ValueError, "fixture results are harness-only"):
                analysis.regenerate(manifest_path)

    def test_regeneration_keeps_outputs_in_manifest_results_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = self._write_raw(root)
            nested_manifest = write_manifest(root, raw_path)
            manifest_path = root / "main.json"
            nested_manifest.replace(manifest_path)

            result = analysis.regenerate(manifest_path)

            self.assertEqual(root / "processed/summary.csv", Path(result["outputs"]["summary"]))
            self.assertTrue((root / "processed/summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
