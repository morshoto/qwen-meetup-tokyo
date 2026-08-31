import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_lab.evaluation import TrialResult, TrialStatus
from llm_lab.evaluation.storage import JsonlResultWriter


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_PATH = ROOT / "experiments/exp_001-context_measurement/analyze.py"
SPEC = importlib.util.spec_from_file_location("exp_001_analyze", ANALYSIS_PATH)
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def trial(
    task_type: str,
    context_tokens: int,
    position: float,
    repeat_index: int,
    *,
    correct: bool | None,
) -> TrialResult:
    score = {"scorer": "calibrated.v1"}
    if correct is not None:
        score.update(
            {
                "correct": correct,
                "value": float(correct),
                "exact_correct": correct,
                "answer_bearing_correct": correct,
                "format_valid": True,
            }
        )
    return TrialResult(
        trial_id=(
            f"exp_001:task.literal.000001:"
            f"ctx{context_tokens}:p{int(position * 100):03d}:run{repeat_index:02d}"
        ),
        experiment_id="exp_001",
        task_id="task.literal.000001",
        status=TrialStatus.COMPLETED if correct is not None else TrialStatus.RUNTIME_ERROR,
        input={
            "task_type": task_type,
            "condition_id": f"ctx{context_tokens}:p{int(position * 100):03d}",
            "target_context_tokens": context_tokens,
            "requested_evidence_position": position,
            "actual_evidence_position": position,
        },
        score=score,
        error=None if correct is not None else {"type": "MemoryError", "message": "simulated"},
    )


class Exp001AnalysisTests(unittest.TestCase):
    def _manifest(self, root: Path, raw_path: Path) -> Path:
        manifest_path = root / "manifests" / "main.json"
        manifest_path.parent.mkdir(parents=True)
        coverage = [
            {
                "task_type": "literal_retrieval",
                "target_context_tokens": context_tokens,
                "requested_evidence_position": position,
                # A complete attempted cell remains valid even when one
                # trial fails at runtime; the failure stays in the primary
                # end-to-end denominator.
                "status": "valid",
                "reason": None,
                "expected_trial_n": 2,
                "trial_n": 2,
                "scored_n": 1 if context_tokens == 32768 and position == 0.50 else 2,
            }
            for context_tokens in (8192, 32768)
            for position in (0.05, 0.50, 0.95)
        ]
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": "exp_001",
                    "phase": "main",
                    "backend": "transformers",
                    "scorer_version": "calibrated.v1",
                    "raw_results": str(raw_path),
                    "raw_results_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    "context_lengths": [8192, 32768],
                    "evidence_positions": [0.05, 0.50, 0.95],
                    "task_types": ["literal_retrieval"],
                    "effective_context": {
                        "baseline_length": 8192,
                        "baseline_accuracy_gate": 0.80,
                        "alpha": 0.90,
                    },
                    "coverage": coverage,
                }
            ),
            encoding="utf-8",
        )
        return manifest_path

    def test_regeneration_writes_provenance_checked_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw" / "main-trials.jsonl"
            writer = JsonlResultWriter(raw_path)
            for context_tokens in (8192, 32768):
                for position in (0.05, 0.50, 0.95):
                    writer.append(
                        trial(
                            "literal_retrieval",
                            context_tokens,
                            position,
                            1,
                            correct=not (context_tokens == 32768 and position == 0.50),
                        )
                    )
                    writer.append(
                        trial(
                            "literal_retrieval",
                            context_tokens,
                            position,
                            2,
                            correct=None if context_tokens == 32768 and position == 0.50 else True,
                        )
                    )
            manifest_path = self._manifest(root, raw_path)

            result = analysis.regenerate(manifest_path)

            self.assertEqual(6, result["summary_row_n"])
            self.assertEqual(2, result["position_gap_row_n"])
            self.assertTrue((root / "processed/summary.csv").is_file())
            self.assertTrue((root / "processed/position-gap.csv").is_file())
            self.assertTrue((root / "processed/effective-context.json").is_file())
            self.assertTrue(
                (root / "processed/effective-context-by-position.json").is_file()
            )
            summary = (root / "processed/summary.csv").read_text(encoding="utf-8")
            self.assertIn("attempted_n", summary)
            self.assertIn("runtime_error_n", summary)
            self.assertIn("analysis_status", summary)
            failed_cell = next(
                row
                for row in result["summary_rows"]
                if row["target_context_tokens"] == 32768
                and row["requested_evidence_position"] == 0.50
            )
            self.assertEqual("available", failed_cell["analysis_status"])
            self.assertEqual(2, failed_cell["attempted_n"])
            self.assertEqual(1, failed_cell["scored_n"])
            self.assertEqual(0.0, failed_cell["end_to_end_success"])
            gaps = json.loads(
                (root / "processed/effective-context.json").read_text(encoding="utf-8")
            )
            self.assertEqual("provisional", gaps[0]["status"])

    def test_regeneration_rejects_raw_hash_mismatch_before_writing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw" / "main-trials.jsonl"
            writer = JsonlResultWriter(raw_path)
            writer.append(
                trial("literal_retrieval", 8192, 0.05, 1, correct=True)
            )
            manifest_path = self._manifest(root, raw_path)
            raw_path.write_text(raw_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                analysis.regenerate(manifest_path)

            self.assertFalse((root / "processed/summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
