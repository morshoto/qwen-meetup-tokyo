import json
import math
import unittest
from pathlib import Path

from llm_lab.analysis.quantization import (
    QuantizationAnalysisError,
    recommend_baseline,
    tradeoff_rows,
    validate_complete_quantization_matrix,
)
from llm_lab.quantization import (
    ArtifactProvenance,
    QuantizationManifest,
    QuantizationVariant,
)


def artifact(size: int) -> ArtifactProvenance:
    return ArtifactProvenance(
        source_uri="hf://Qwen/Qwen3.8-27B",
        source_revision="model-sha",
        conversion_command="convert_hf_to_gguf.py",
        converter_revision="llama.cpp-sha",
        artifact_uri=f"file:///models/{size}.gguf",
        artifact_sha256="a" * 64,
        artifact_size_bytes=size,
    )


def variant(condition_id: str, label: str, quantization_type: str, bits: int, size: int) -> QuantizationVariant:
    return QuantizationVariant(
        condition_id=condition_id,
        label=label,
        format="GGUF",
        quantization_type=quantization_type,
        bits=bits,
        artifact=artifact(size),
        runtime_kernel="ggml",
    )


def manifest() -> QuantizationManifest:
    return QuantizationManifest(
        experiment_id="exp_002",
        model_id="Qwen/Qwen3.8-27B",
        model_revision="model-sha",
        tokenizer_id="Qwen/Qwen3.8-27B",
        tokenizer_revision="tokenizer-sha",
        runtime_name="llama.cpp",
        runtime_version="llama.cpp-sha",
        prompt_id="prompt.qa.v001",
        task_ids=("task.literal.000001",),
        context_lengths=(8192, 32768),
        sampling={"temperature": 0.0},
        variants=(
            variant("q8_0", "Q8_0", "Q8_0", 8, 100),
            variant("q6_k", "Q6_K", "Q6_K", 6, 70),
            variant("q5_k_m", "Q5_K_M", "Q5_K_M", 5, 50),
            variant("q4_k_m", "Q4_K_M", "Q4_K_M", 4, 35),
        ),
        repeats=5,
    )


def summary(
    condition_id: str,
    accuracy: float,
    scored_n: int = 10,
    *,
    attempted_n: int | None = None,
    correct_n: int | None = None,
    error_n: int = 0,
) -> dict[str, object]:
    attempted_n = scored_n + error_n if attempted_n is None else attempted_n
    return {
        "experiment_id": "exp_002",
        "task_type": "literal_retrieval",
        "condition_id": condition_id,
        "n": scored_n,
        "completed_n": scored_n,
        "error_n": error_n,
        "scored_n": scored_n,
        "attempted_n": attempted_n,
        "correct_n": correct_n,
        "accuracy": accuracy,
        "median_stream_ttft_s": 1.0,
        "median_prompt_throughput_proxy_tok_s": 100.0,
        "median_post_first_chunk_output_tok_s": 20.0,
        "median_peak_memory_bytes": 200,
    }


def complete_matrix_summaries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition_id in ("q8_0", "q6_k", "q5_k_m", "q4_k_m"):
        for context_length in (8192, 32768):
            row = summary(f"{condition_id}:ctx{context_length}", 0.8, scored_n=5)
            row.update(
                {
                    "task_id": "task.literal.000001",
                    "target_context_tokens": context_length,
                    "variant_condition_id": condition_id,
                    "context_instance_id": f"task.literal.000001:ctx{context_length}",
                    "context_sha256": "a" * 64,
                }
            )
            rows.append(row)
    return rows


class QuantizationAnalysisTests(unittest.TestCase):
    def test_tradeoff_rows_join_required_metrics_and_weighted_accuracy(self) -> None:
        rows = tradeoff_rows(
            [
                summary("q8_0", 0.8, 10),
                summary("q8_0", 1.0, 20),
                summary("q6_k", 0.9),
                summary("q5_k_m", 0.9),
                summary("q4_k_m", 0.9),
            ],
            manifest(),
        )

        q8 = rows[0]
        self.assertEqual("q8_0", q8["condition_id"])
        self.assertEqual(100, q8["artifact_size_bytes"])
        self.assertEqual(0.9333333333333333, q8["accuracy"])
        self.assertEqual(30, q8["scored_n"])
        self.assertEqual(30, q8["exact_scored_n"])
        self.assertEqual(0.9333333333333333, q8["exact_accuracy"])
        self.assertEqual(200, q8["median_peak_memory_bytes"])

    def test_tradeoff_rows_groups_context_scoped_execution_conditions_by_variant(self) -> None:
        rows = tradeoff_rows(
            [
                {
                    **summary("q8_0:ctx8192", 0.8, 10),
                    "variant_condition_id": "q8_0",
                },
                {
                    **summary("q8_0:ctx32768", 1.0, 20),
                    "variant_condition_id": "q8_0",
                },
                summary("q6_k", 0.9),
                summary("q5_k_m", 0.9),
                summary("q4_k_m", 0.9),
            ],
            manifest(),
        )

        self.assertEqual(30, rows[0]["scored_n"])
        self.assertEqual(0.9333333333333333, rows[0]["accuracy"])

    def test_tradeoff_rows_fail_when_a_declared_condition_has_no_measurements(self) -> None:
        with self.assertRaisesRegex(QuantizationAnalysisError, "q4_k_m"):
            tradeoff_rows(
                [summary(condition_id, 1.0) for condition_id in ("q8_0", "q6_k", "q5_k_m")],
                manifest(),
            )

    def test_recommendation_prefers_smallest_measured_artifact_within_accuracy_tolerance(self) -> None:
        rows = tradeoff_rows(
            [
                summary("q8_0", 0.86),
                summary("q6_k", 0.85),
                summary("q5_k_m", 0.84),
                summary("q4_k_m", 0.70),
            ],
            manifest(),
        )

        recommendation = recommend_baseline(rows, accuracy_tolerance=0.03)

        self.assertEqual("q5_k_m", recommendation["condition_id"])
        self.assertEqual(0.86, recommendation["best_end_to_end_success"])
        self.assertEqual(0.03, recommendation["accuracy_tolerance"])

    def test_tradeoff_rows_reports_failures_and_recommendation_uses_end_to_end_success(self) -> None:
        rows = tradeoff_rows(
            [
                summary(
                    "q8_0",
                    1.0,
                    scored_n=8,
                    attempted_n=10,
                    correct_n=8,
                    error_n=2,
                ),
                summary(
                    "q6_k",
                    0.875,
                    scored_n=8,
                    attempted_n=8,
                    correct_n=7,
                ),
                summary("q5_k_m", 0.75, correct_n=6),
                summary("q4_k_m", 0.5, correct_n=5),
            ],
            manifest(),
        )

        q8 = rows[0]
        self.assertEqual(10, q8["attempted_n"])
        self.assertEqual(8, q8["correct_n"])
        self.assertEqual(1.0, q8["scored_accuracy"])
        self.assertEqual(0.8, q8["end_to_end_success"])
        self.assertEqual(0.2, q8["failure_rate"])

        recommendation = recommend_baseline(rows, accuracy_tolerance=0.03)
        self.assertEqual("q6_k", recommendation["condition_id"])
        self.assertEqual(0.875, recommendation["best_end_to_end_success"])

    def test_tradeoff_rows_rejects_non_finite_metrics(self) -> None:
        summaries = [
            summary(condition_id, 1.0)
            for condition_id in ("q8_0", "q6_k", "q5_k_m", "q4_k_m")
        ]
        summaries[0]["median_peak_memory_bytes"] = math.nan

        with self.assertRaisesRegex(
            QuantizationAnalysisError,
            "q8_0 is missing measured median_peak_memory_bytes",
        ):
            tradeoff_rows(summaries, manifest())

    def test_tradeoff_rows_rejects_partial_matrix_before_recommendation(self) -> None:
        summaries = complete_matrix_summaries()
        summaries.pop()

        with self.assertRaisesRegex(
            QuantizationAnalysisError, "complete quantization matrix"
        ):
            tradeoff_rows(summaries, manifest(), require_complete=True)

    def test_complete_quantization_matrix_requires_manifest_repeats(self) -> None:
        summaries = complete_matrix_summaries()
        summaries[0]["attempted_n"] = 4

        with self.assertRaisesRegex(
            QuantizationAnalysisError, "expected manifest repeats=5"
        ):
            validate_complete_quantization_matrix(summaries, manifest())

    def test_complete_quantization_matrix_requires_matched_context_identity(self) -> None:
        summaries = complete_matrix_summaries()
        summaries[2]["context_sha256"] = "b" * 64

        with self.assertRaisesRegex(QuantizationAnalysisError, "do not share one context"):
            validate_complete_quantization_matrix(summaries, manifest())

    def test_notebook_contains_required_analysis_sections(self) -> None:
        notebook_path = Path(
            "experiments/exp_002-quantization_llama_cpp_gguf/analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        for required in (
            "SUMMARY_PATH = RESULTS_DIR / 'processed/summary.csv'",
            "'manifest.full.json' if PHASE == 'full' else 'manifest.json'",
            "tradeoff_rows",
            "accuracy_vs_memory",
            "speed_vs_memory",
            "end_to_end_success",
            "exact_accuracy",
            "prompt_throughput_proxy_tok_s",
            "post_first_chunk_output_tok_s",
            "artifact-size-by-variant.png",
            "rss-by-variant.png",
            "success-vs-artifact-size.png",
            "accuracy-vs-rss.png",
            "ttft-vs-context.png",
            "throughput-vs-context.png",
            "recommend_baseline",
        ):
            self.assertIn(required, source)

        self.assertIn("ROOT = next(", source)
        self.assertIn(
            "ROOT / 'experiments/exp_002-quantization_llama_cpp_gguf/results'",
            source,
        )
        self.assertNotIn("SUMMARY_PATH = Path('results/processed/summary.csv')", source)

    def test_notebook_separates_capability_and_systems_metrics(self) -> None:
        notebook_path = Path(
            "experiments/exp_002-quantization_llama_cpp_gguf/analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        for required in (
            "capability_frame",
            "systems_cost_frame",
            "median_peak_memory_bytes",
            "stream-derived proxies",
            "native prefill/decode counters are unavailable",
        ):
            self.assertIn(required, source)

    def test_notebook_has_no_cached_analysis_outputs(self) -> None:
        notebook_path = Path(
            "experiments/exp_002-quantization_llama_cpp_gguf/analysis.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(code_cells)
        self.assertTrue(all(cell.get("execution_count") is None for cell in code_cells))
        self.assertTrue(all(cell.get("outputs") == [] for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
