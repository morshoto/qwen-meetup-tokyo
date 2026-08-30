"""Run the manifest-driven exp_002 GGUF quantization matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_lab.analysis import aggregate_jsonl, write_summary_csv  # noqa: E402
from llm_lab.context import ContextTokenizer, Evidence, SyntheticContextGenerator  # noqa: E402
from llm_lab.datasets import TaskCatalog  # noqa: E402
from llm_lab.evaluation import (  # noqa: E402
    EvaluationRunner,
    EvaluationTask,
    CalibratedAnswerScorer,
    TrialResult,
    load_trial_results,
    make_trial_id,
)
from llm_lab.generation import SamplingConfig  # noqa: E402
from llm_lab.models import ModelSpec  # noqa: E402
from llm_lab.quantization import QuantizationManifest, QuantizationVariant  # noqa: E402
from llm_lab.runtimes import LlamaCppRuntime, RuntimeConfig  # noqa: E402


TASK_CATALOG = ROOT / "data/tasks/core.v002.jsonl"
SCORER_VERSION = CalibratedAnswerScorer.name
PLACEHOLDER_MARKERS = ("REPLACE_WITH", "SET_TO_")
SOURCE_REVISION_PATHS = {
    "runner.py": Path(__file__),
    "context/synthetic.py": ROOT / "src/llm_lab/context/synthetic.py",
    "datasets/catalog.py": ROOT / "src/llm_lab/datasets/catalog.py",
    "evaluation/contracts.py": ROOT / "src/llm_lab/evaluation/contracts.py",
    "evaluation/runner.py": ROOT / "src/llm_lab/evaluation/runner.py",
    "generation/types.py": ROOT / "src/llm_lab/generation/types.py",
    "runtimes/base.py": ROOT / "src/llm_lab/runtimes/base.py",
    "runtimes/llama_cpp.py": ROOT / "src/llm_lab/runtimes/llama_cpp.py",
}
RuntimeFactory = Callable[[], LlamaCppRuntime]


def load_manifest(path: Path) -> QuantizationManifest:
    """Load one resolved manifest and reject templates before parsing it."""

    path = _rooted(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("template") is True:
        raise ValueError("a resolved manifest is required; template manifest supplied")
    if _contains_placeholder(record):
        raise ValueError("resolved manifest contains placeholder values")
    manifest = QuantizationManifest.from_record(record)
    if manifest.experiment_id != "exp_002":
        raise ValueError(f"unsupported experiment in manifest: {manifest.experiment_id!r}")
    if manifest.runtime_name != "llama.cpp":
        raise ValueError(f"unsupported runtime in manifest: {manifest.runtime_name!r}")
    return manifest


def expected_trial_count(manifest: QuantizationManifest) -> int:
    """Return the full manifest matrix size before any CLI selection."""

    return (
        len(manifest.variants)
        * len(manifest.context_lengths)
        * len(manifest.task_ids)
        * (manifest.capability_repeats or manifest.repeats)
    )


def run_experiment(
    *,
    manifest_path: Path,
    output_path: Path,
    processed_path: Path,
    timing_output_path: Path | None = None,
    timing_processed_path: Path | None = None,
    condition_ids: Iterable[str] | None = None,
    context_lengths: Iterable[int] | None = None,
    repeats: int | None = None,
    timing_repeats: int | None = None,
    runtime_factory: RuntimeFactory = LlamaCppRuntime,
) -> dict[str, Any]:
    """Run a selected manifest matrix, safely resuming an existing JSONL file."""

    manifest_path = _rooted(manifest_path)
    output_path = _rooted(output_path)
    processed_path = _rooted(processed_path)
    if timing_output_path is not None:
        timing_output_path = _rooted(timing_output_path)
        if timing_output_path == output_path:
            raise ValueError("timing output must be separate from capability output")
        timing_processed_path = _rooted(
            timing_processed_path
            or processed_path.with_name("timing-summary.csv")
        )
    elif timing_processed_path is not None:
        raise ValueError("timing_processed_path requires timing_output_path")
    manifest = load_manifest(manifest_path)
    variants = _select_variants(manifest, condition_ids)
    lengths = _select_lengths(manifest, context_lengths)
    repeat_ceiling = manifest.capability_repeats or manifest.repeats
    run_repeats = repeat_ceiling if repeats is None else repeats
    if run_repeats < 1 or run_repeats > repeat_ceiling:
        raise ValueError(
            "repeats must be between 1 and the manifest capability repeat count"
        )
    timing_run_repeats: int | None = None
    if timing_output_path is not None:
        timing_ceiling = manifest.timing_repeats
        if timing_ceiling is None:
            raise ValueError(
                "a resolved manifest with explicit timing_repeats is required "
                "for separate timing probes"
            )
        timing_run_repeats = (
            timing_ceiling if timing_repeats is None else int(timing_repeats)
        )
        if timing_run_repeats < 3 or timing_run_repeats > 5:
            raise ValueError("timing_repeats must be between 3 and 5")
        if timing_run_repeats > timing_ceiling:
            raise ValueError(
                "timing_repeats cannot exceed the manifest timing repeat count"
            )
    source_revisions = _source_revisions()
    fingerprint = _run_fingerprint(
        manifest, source_revisions=source_revisions
    )
    artifact_paths = _verify_artifacts(manifest_path, variants)
    catalog_path = _task_catalog_path(manifest)
    if manifest.task_catalog_sha256 is not None:
        actual_catalog_sha256 = _sha256(catalog_path)
        if actual_catalog_sha256.lower() != manifest.task_catalog_sha256.lower():
            raise ValueError("task catalog SHA-256 mismatch")
    catalog = TaskCatalog.from_jsonl(catalog_path)
    if manifest.scorer_version != SCORER_VERSION:
        raise ValueError(
            f"unsupported scorer version {manifest.scorer_version!r}; "
            f"expected {SCORER_VERSION!r}"
        )
    unknown_task_ids = set(manifest.task_ids) - set(catalog.ids)
    if unknown_task_ids:
        raise ValueError(
            "manifest task_ids must exist in the shared task catalog: "
            f"{sorted(unknown_task_ids)}"
        )
    model = ModelSpec(
        model_id=manifest.model_id,
        revision=manifest.model_revision,
        tokenizer_id=manifest.tokenizer_id,
        tokenizer_revision=manifest.tokenizer_revision,
    )
    sampling = _sampling_config(manifest.sampling)
    existing = load_trial_results(output_path)
    expected_ids = {
        make_trial_id(
            manifest.experiment_id,
            task_id,
            condition_id=_execution_condition_id(variant, length),
            repeat_index=repeat_index,
        )
        for variant in variants
        for length in lengths
        for task_id in manifest.task_ids
        for repeat_index in range(1, run_repeats + 1)
    }
    expected_timing_ids = (
        {
            make_trial_id(
                manifest.experiment_id,
                task_id,
                condition_id=_execution_condition_id(variant, length),
                repeat_index=repeat_index,
            )
            for variant in variants
            for length in lengths
            for task_id in manifest.task_ids
            for repeat_index in range(1, timing_run_repeats + 1)
        }
        if timing_run_repeats is not None
        else set()
    )
    _validate_existing(
        existing,
        expected_ids,
        fingerprint,
        expected_scorer=SCORER_VERSION,
        source_revisions=source_revisions,
    )
    timing_existing = (
        load_trial_results(timing_output_path)
        if timing_output_path is not None
        else []
    )
    if timing_output_path is not None:
        _validate_existing(
            timing_existing,
            expected_timing_ids,
            fingerprint,
            expected_scorer=SCORER_VERSION,
            source_revisions=source_revisions,
        )
    existing_ids = {result.trial_id for result in existing}
    existing_timing_ids = {result.trial_id for result in timing_existing}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if timing_output_path is not None:
        timing_output_path.parent.mkdir(parents=True, exist_ok=True)
    base_tasks_by_length: dict[int, tuple[EvaluationTask, ...]] = {}
    total_results = len(existing)
    total_timing_results = len(timing_existing)
    tokenizer: ContextTokenizer | None = None

    for variant_index, variant in enumerate(variants):
        runtime = runtime_factory()
        runtime.load(
            model,
            _runtime_config(manifest, artifact_paths[variant.condition_id]),
        )
        try:
            if tokenizer is None:
                tokenizer = runtime.get_tokenizer()
                base_tasks_by_length = {
                    length: tuple(
                        _build_tasks(
                            catalog,
                            manifest.task_ids,
                            length,
                            tokenizer=tokenizer,
                            prompt_id=manifest.prompt_id,
                            seed=sampling.seed or 0,
                            task_catalog=manifest.task_catalog,
                            task_catalog_sha256=manifest.task_catalog_sha256,
                            scorer_version=manifest.scorer_version,
                            source_revisions=source_revisions,
                        )
                    )
                    for length in lengths
                }
            evaluator = EvaluationRunner(
                runtime=runtime,
                model=model,
                scorer=CalibratedAnswerScorer(),
                experiment_id=manifest.experiment_id,
                output_path=output_path,
            )
            timing_evaluator = (
                EvaluationRunner(
                    runtime=runtime,
                    model=model,
                    scorer=CalibratedAnswerScorer(),
                    experiment_id=manifest.experiment_id,
                    output_path=timing_output_path,
                )
                if timing_output_path is not None
                else None
            )
            for length in lengths:
                variant_tasks = tuple(
                    replace(
                        task,
                        metadata={
                            **dict(task.metadata),
                            "variant_condition_id": variant.condition_id,
                            "variant_label": variant.label,
                            "quantization_type": variant.quantization_type,
                            "artifact_uri": variant.artifact.artifact_uri,
                            "artifact_sha256": variant.artifact.artifact_sha256,
                            "artifact_size_bytes": variant.artifact.artifact_size_bytes,
                            "run_fingerprint": fingerprint,
                        },
                    )
                    for task in base_tasks_by_length[length]
                )
                execution_condition_id = _execution_condition_id(variant, length)
                for repeat_index in range(1, run_repeats + 1):
                    missing_tasks = [
                        task
                        for task in variant_tasks
                        if make_trial_id(
                            manifest.experiment_id,
                            task.task_id,
                            condition_id=execution_condition_id,
                            repeat_index=repeat_index,
                        )
                        not in existing_ids
                    ]
                    if not missing_tasks:
                        continue
                    new_results = evaluator.run(
                        missing_tasks,
                        repeats=1,
                        repeat_indices=(repeat_index,),
                        condition_id=execution_condition_id,
                        sampling=sampling,
                    )
                    total_results += len(new_results)
                    existing_ids.update(result.trial_id for result in new_results)
                if timing_evaluator is not None and timing_run_repeats is not None:
                    timing_tasks = tuple(
                        replace(
                            task,
                            metadata={
                                **dict(task.metadata),
                                "sample_role": "timing",
                            },
                        )
                        for task in variant_tasks
                    )
                    for repeat_index in range(1, timing_run_repeats + 1):
                        missing_tasks = [
                            task
                            for task in timing_tasks
                            if make_trial_id(
                                manifest.experiment_id,
                                task.task_id,
                                condition_id=execution_condition_id,
                                repeat_index=repeat_index,
                            )
                            not in existing_timing_ids
                        ]
                        if not missing_tasks:
                            continue
                        new_timing_results = timing_evaluator.run(
                            missing_tasks,
                            repeats=1,
                            repeat_indices=(repeat_index,),
                            condition_id=execution_condition_id,
                            sampling=sampling,
                        )
                        total_timing_results += len(new_timing_results)
                        existing_timing_ids.update(
                            result.trial_id for result in new_timing_results
                        )
        finally:
            runtime.close()

    summaries = aggregate_jsonl(
        output_path,
        expected_scorer=SCORER_VERSION,
        group_by_task=True,
    )
    raw_results_sha256 = _sha256(output_path)
    _attach_raw_provenance(
        summaries,
        path=output_path,
        sha256=raw_results_sha256,
        path_key="raw_results",
        hash_key="raw_results_sha256",
    )
    write_summary_csv(processed_path, summaries)
    timing_summary_row_n: int | None = None
    timing_raw_results_sha256: str | None = None
    if timing_output_path is not None and timing_processed_path is not None:
        timing_summaries = aggregate_jsonl(
            timing_output_path,
            expected_scorer=SCORER_VERSION,
            group_by_task=True,
        )
        timing_raw_results_sha256 = _sha256(timing_output_path)
        _attach_raw_provenance(
            timing_summaries,
            path=timing_output_path,
            sha256=timing_raw_results_sha256,
            path_key="timing_raw_results",
            hash_key="timing_raw_results_sha256",
        )
        write_summary_csv(timing_processed_path, timing_summaries)
        timing_summary_row_n = len(timing_summaries)
    return {
        "experiment_id": manifest.experiment_id,
        "expected_trial_n": len(expected_ids),
        "actual_trial_n": total_results,
        "expected_timing_trial_n": len(expected_timing_ids),
        "actual_timing_trial_n": total_timing_results,
        "skipped_trial_n": len(existing),
        "skipped_timing_trial_n": len(timing_existing),
        "summary_row_n": len(summaries),
        "timing_summary_row_n": timing_summary_row_n,
        "scorer_version": SCORER_VERSION,
        "raw_results_sha256": raw_results_sha256,
        "timing_raw_results_sha256": timing_raw_results_sha256,
        "output_path": str(output_path),
        "processed_path": str(processed_path),
        "timing_output_path": (
            str(timing_output_path) if timing_output_path is not None else None
        ),
        "timing_processed_path": (
            str(timing_processed_path) if timing_processed_path is not None else None
        ),
        "run_fingerprint": fingerprint,
    }


def _attach_raw_provenance(
    summaries: list[dict[str, Any]],
    *,
    path: Path,
    sha256: str,
    path_key: str,
    hash_key: str,
) -> None:
    """Attach immutable raw-output identity to every processed summary row."""

    for row in summaries:
        row[path_key] = str(path)
        row[hash_key] = sha256


def _build_tasks(
    catalog: TaskCatalog,
    task_ids: Iterable[str],
    context_tokens: int,
    *,
    tokenizer: ContextTokenizer,
    prompt_id: str,
    seed: int,
    task_catalog: str | None,
    task_catalog_sha256: str | None,
    scorer_version: str,
    source_revisions: Mapping[str, str],
) -> list[EvaluationTask]:
    generator = SyntheticContextGenerator(tokenizer=tokenizer)
    tasks: list[EvaluationTask] = []
    for task_id in task_ids:
        definition = catalog.get(task_id)
        task_seed = seed + int(definition.metadata["seed"])
        generated = generator.generate(
            [
                Evidence(id=str(item["id"]), text=str(item["text"]))
                for item in definition.evidence
            ],
            target_tokens=context_tokens,
            evidence_position=0.5,
            seed=task_seed,
        )
        spans = [
            {
                "id": span.id,
                "text": span.text,
                "token_start": span.token_start,
                "token_end": span.token_end,
                "requested_position": span.requested_position,
                "actual_position": span.actual_position,
            }
            for span in generated.evidence
        ]
        tasks.append(
            EvaluationTask.from_definition(
                definition,
                context=generated.text,
                prompt_id=prompt_id,
                metadata={
                    "target_context_tokens": context_tokens,
                    "actual_context_tokens": generated.token_count,
                    "requested_evidence_position": 0.5,
                    "actual_evidence_position": sum(
                        span["actual_position"] for span in spans
                    )
                    / len(spans),
                    "evidence_spans": spans,
                    "context_sha256": hashlib.sha256(
                        generated.text.encode("utf-8")
                    ).hexdigest(),
                    "context_generator": generated.metadata["generator"],
                    "context_tokenization": generated.metadata["tokenization"],
                    "context_tokenization_mode": generated.metadata.get(
                        "tokenization_mode", "tokenizer"
                    ),
                    "target_unit": "input-tokenizer-tokens",
                    "task_catalog": task_catalog,
                    "task_catalog_sha256": task_catalog_sha256,
                    "scorer_version": scorer_version,
                    "source_revisions": dict(source_revisions),
                },
            )
        )
    return tasks


def _runtime_config(
    manifest: QuantizationManifest,
    artifact_path: Path,
) -> RuntimeConfig:
    options = dict(manifest.runtime_options)
    options["model_path"] = str(artifact_path)
    return RuntimeConfig(
        name=manifest.runtime_name,
        version=manifest.runtime_version,
        options=options,
    )


def _sampling_config(values: Mapping[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        max_new_tokens=int(values["max_new_tokens"]),
        temperature=float(values["temperature"]),
        top_p=float(values["top_p"]),
        top_k=None if values.get("top_k") is None else int(values["top_k"]),
        seed=None if values.get("seed") is None else int(values["seed"]),
    )


def _select_variants(
    manifest: QuantizationManifest,
    condition_ids: Iterable[str] | None,
) -> tuple[QuantizationVariant, ...]:
    selected = manifest.condition_ids if condition_ids is None else tuple(condition_ids)
    if not selected:
        raise ValueError("at least one variant must be selected")
    by_id = {variant.condition_id: variant for variant in manifest.variants}
    unknown = set(selected) - set(by_id)
    if unknown:
        raise ValueError(f"unknown variant condition IDs: {sorted(unknown)}")
    return tuple(by_id[condition_id] for condition_id in selected)


def _select_lengths(
    manifest: QuantizationManifest,
    context_lengths: Iterable[int] | None,
) -> tuple[int, ...]:
    selected = (
        manifest.context_lengths
        if context_lengths is None
        else tuple(int(length) for length in context_lengths)
    )
    if not selected:
        raise ValueError("at least one context length must be selected")
    unknown = set(selected) - set(manifest.context_lengths)
    if unknown:
        raise ValueError(f"unknown context lengths: {sorted(unknown)}")
    return selected


def _execution_condition_id(variant: QuantizationVariant, length: int) -> str:
    return f"{variant.condition_id}:ctx{length}"


def _run_fingerprint(
    manifest: QuantizationManifest,
    *,
    source_revisions: Mapping[str, str] | None = None,
) -> str:
    """Fingerprint immutable controls so a pilot can safely grow to full."""

    payload = {
        "manifest": manifest.to_record(),
        "source_revisions": dict(
            _source_revisions() if source_revisions is None else source_revisions
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_revisions() -> dict[str, str]:
    return {
        name: _sha256(path)
        for name, path in SOURCE_REVISION_PATHS.items()
    }


def _task_catalog_path(manifest: QuantizationManifest) -> Path:
    if manifest.task_catalog is None:
        return TASK_CATALOG
    parsed = urlparse(manifest.task_catalog)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"unsupported task catalog URI host: {parsed.netloc}")
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(
            f"task catalog must be a local path or file URI: {manifest.task_catalog}"
        )
    path = Path(unquote(manifest.task_catalog))
    return path if path.is_absolute() else _rooted(path)


def _validate_existing(
    existing: Iterable[TrialResult],
    expected_ids: set[str],
    fingerprint: str,
    *,
    expected_scorer: str,
    source_revisions: Mapping[str, str],
) -> None:
    for result in existing:
        if result.trial_id not in expected_ids:
            raise ValueError(f"existing trial is outside selected run: {result.trial_id}")
        if result.input.get("run_fingerprint") != fingerprint:
            raise ValueError("existing raw results do not match the selected manifest run")
        if result.input.get("source_revisions") != dict(source_revisions):
            raise ValueError("existing raw results do not match the selected source revisions")
        if result.score.get("scorer") != expected_scorer:
            actual = result.score.get("scorer", "<missing>")
            raise ValueError(
                f"existing raw results use scorer version {actual!r}; "
                f"expected scorer version {expected_scorer!r}"
            )


def _verify_artifacts(
    manifest_path: Path,
    variants: Iterable[QuantizationVariant],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for variant in variants:
        path = _artifact_path(manifest_path, variant.artifact.artifact_uri)
        if not path.is_file():
            raise FileNotFoundError(f"artifact is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != variant.artifact.artifact_size_bytes:
            raise ValueError(
                f"artifact size mismatch for {variant.condition_id}: "
                f"manifest={variant.artifact.artifact_size_bytes}, actual={actual_size}"
            )
        digest = _sha256(path)
        if digest.lower() != variant.artifact.artifact_sha256.lower():
            raise ValueError(f"artifact SHA-256 mismatch for {variant.condition_id}")
        paths[variant.condition_id] = path
    return paths


def _artifact_path(manifest_path: Path, artifact_uri: str) -> Path:
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError(f"unsupported file URI host: {parsed.netloc}")
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"artifact URI must be a local path or file URI: {artifact_uri}")
    path = Path(unquote(artifact_uri))
    return path if path.is_absolute() else manifest_path.parent / path


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return any(marker in value for marker in PLACEHOLDER_MARKERS)
    if isinstance(value, Mapping):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(item) for item in value)
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processed", type=Path)
    parser.add_argument(
        "--timing-output",
        type=Path,
        help="separate JSONL output for repeated timing probes",
    )
    parser.add_argument(
        "--timing-processed",
        type=Path,
        help="CSV output for the separate timing probes",
    )
    parser.add_argument("--condition-id", action="append")
    parser.add_argument("--context-length", action="append", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--timing-repeats", type=int)
    args = parser.parse_args(argv)
    manifest_path = _rooted(args.manifest)
    default_results = manifest_path.parent / "raw"
    output_path = args.output or default_results / "trials.jsonl"
    processed_path = args.processed or manifest_path.parent / "processed" / "summary.csv"
    result = run_experiment(
        manifest_path=manifest_path,
        output_path=output_path,
        processed_path=processed_path,
        timing_output_path=args.timing_output,
        timing_processed_path=args.timing_processed,
        condition_ids=args.condition_id,
        context_lengths=args.context_length,
        repeats=args.repeats,
        timing_repeats=args.timing_repeats,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
