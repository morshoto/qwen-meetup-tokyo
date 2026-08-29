"""Run the exp_001 context-length and evidence-position matrix.

The fixture backend validates the experiment plumbing without model weights.
Use ``--backend transformers`` for a real local Qwen run after installing the
optional backend and recording the runtime/resource configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_lab.context import (  # noqa: E402
    ContextTokenizer,
    Evidence,
    InferenceTokenizer,
    SyntheticContextGenerator,
)
from llm_lab.datasets import TaskCatalog  # noqa: E402
from llm_lab.evaluation import (  # noqa: E402
    EvaluationRunner,
    EvaluationTask,
    CalibratedAnswerScorer,
    TrialResult,
    TrialStatus,
)
from llm_lab.generation import (  # noqa: E402
    GenerationRequest,
    GenerationResponse,
    GenerationTiming,
    RuntimeMetadata,
    SamplingConfig,
    TokenUsage,
)
from llm_lab.models import ModelSpec, qwen38_model_spec  # noqa: E402
from llm_lab.runtimes import RuntimeConfig  # noqa: E402
from llm_lab.runtimes.transformers import QwenTransformersRuntime  # noqa: E402
from llm_lab.telemetry import capture_environment  # noqa: E402


TASK_CATALOG = ROOT / "data/tasks/core.v002.jsonl"
EXPERIMENT_ID = "exp_001"
SCORER_VERSION = CalibratedAnswerScorer.name
TASK_TYPES = ("literal_retrieval", "semantic_retrieval", "multi_hop")
SMOKE_CONTEXT_LENGTHS = (8192, 32768)
PILOT_CONTEXT_LENGTHS = (8192, 32768, 65536)
MAIN_CONTEXT_LENGTHS = (8192, 32768, 65536, 131072)
SMOKE_POSITIONS = (0.05, 0.50, 0.95)
FULL_POSITIONS = (0.05, 0.25, 0.50, 0.75, 0.95)
REPEATS = {"smoke": 1, "pilot": 5, "main": 20}


class FixtureTokenizer:
    """Deterministic tokenizer used by tokenizer-contract tests only."""

    name = "tokenizer-v1"

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        del skip_special_tokens
        return " ".join(self._id_to_token[token_id] for token_id in token_ids)


@dataclass(frozen=True)
class Condition:
    target_context_tokens: int
    evidence_position: float

    @property
    def condition_id(self) -> str:
        return (
            f"baseline:ctx{self.target_context_tokens:06d}:"
            f"p{int(self.evidence_position * 100):03d}"
        )


def planned_conditions(phase: str) -> list[Condition]:
    """Return the deterministic condition grid for a named run phase."""

    if phase == "smoke":
        lengths, positions = SMOKE_CONTEXT_LENGTHS, SMOKE_POSITIONS
    elif phase == "pilot":
        lengths, positions = PILOT_CONTEXT_LENGTHS, FULL_POSITIONS
    elif phase == "main":
        lengths, positions = MAIN_CONTEXT_LENGTHS, FULL_POSITIONS
    else:
        raise ValueError(f"unsupported phase: {phase!r}")
    return [
        Condition(length, position)
        for length in lengths
        for position in positions
    ]


def build_tasks(
    catalog: TaskCatalog,
    condition: Condition,
    *,
    fixture_seed: int,
    tokenizer: ContextTokenizer | None = None,
) -> list[EvaluationTask]:
    """Build reproducible evaluation tasks for one context/position condition."""

    generator = SyntheticContextGenerator(tokenizer=tokenizer)
    tasks: list[EvaluationTask] = []
    for definition in catalog.tasks:
        task_seed = fixture_seed + int(definition.metadata["seed"])
        generated = generator.generate(
            [
                Evidence(
                    id=str(item["id"]),
                    text=str(item["text"]),
                )
                for item in definition.evidence
            ],
            target_tokens=condition.target_context_tokens,
            evidence_position=condition.evidence_position,
            seed=task_seed,
        )
        spans = [asdict(span) for span in generated.evidence]
        metadata = {
            "corpus_id": "corpus.synthetic.000001",
            "fixture_seed": fixture_seed,
            "task_seed": task_seed,
            "target_context_tokens": condition.target_context_tokens,
            "actual_context_tokens": generated.token_count,
            "requested_evidence_position": condition.evidence_position,
            "actual_evidence_position": sum(
                span["actual_position"] for span in spans
            )
            / len(spans),
            "evidence_spans": spans,
            "context_generator": generated.metadata["generator"],
            "context_tokenization": generated.metadata["tokenization"],
            "context_tokenization_mode": generated.metadata.get(
                "tokenization_mode", "whitespace-fixture"
            ),
            "target_unit": (
                "inference-tokenizer-tokens"
                if tokenizer is not None
                else "whitespace-fixture-tokens"
            ),
        }
        tasks.append(
            EvaluationTask.from_definition(
                definition,
                context=generated.text,
                metadata=metadata,
            )
        )
    return tasks


class FixtureRuntime:
    """Deterministic backend for smoke validation; it is not model evidence."""

    name = "fixture"

    def __init__(self, answers: Mapping[str, str] | None = None) -> None:
        self._answers = dict(answers or {})

    def load(self, model: ModelSpec, config: RuntimeConfig) -> None:
        del model, config

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        task_id = str(request.metadata["task_id"])
        output = self._answers[task_id]
        prompt_tokens = len(request.prompt.split())
        return GenerationResponse(
            output_text=output,
            usage=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=1),
            timing=GenerationTiming(
                ttft_seconds=0.001,
                prefill_seconds=0.002,
                decode_seconds=0.001,
                total_seconds=0.004,
            ),
            runtime=RuntimeMetadata(
                runtime_name=self.name,
                runtime_version="1.0",
                model_id=request.model.model_id,
                tokenizer_id=request.model.tokenizer_id,
                config={"purpose": "harness-smoke-only"},
            ),
        )

    def close(self) -> None:
        return None


def _fixture_answers(catalog: TaskCatalog) -> dict[str, str]:
    """Return canonical answers for the deterministic smoke backend."""

    answers: dict[str, str] = {}
    for task in catalog.tasks:
        value = task.expected.get("value")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"fixture smoke requires a string expected value: {task.task_id}"
            )
        answers[task.task_id] = value
    return answers


def run_experiment(
    *,
    phase: str,
    backend: str,
    output_path: Path,
    manifest_path: Path,
    fixture_seed: int = 42,
    overwrite_smoke: bool = False,
) -> dict[str, Any]:
    """Execute one phase and write raw JSONL plus a coverage manifest."""

    output_path = _rooted(output_path)
    manifest_path = _rooted(manifest_path)
    if overwrite_smoke and (phase != "smoke" or backend != "fixture"):
        raise ValueError("overwrite_smoke is only supported for the fixture smoke phase")
    if output_path == manifest_path:
        raise ValueError("output and manifest paths must be different")
    if not overwrite_smoke and (output_path.exists() or manifest_path.exists()):
        raise FileExistsError(
            "output or manifest already exists; choose new paths or pass "
            "--overwrite-smoke for an explicit fixture regeneration"
        )

    temporary_output = _temporary_sibling(output_path) if overwrite_smoke else None
    temporary_manifest = _temporary_sibling(manifest_path) if overwrite_smoke else None
    working_output_path = temporary_output or output_path
    catalog = TaskCatalog.from_jsonl(TASK_CATALOG)
    model = qwen38_model_spec()
    context_tokenizer: ContextTokenizer | None = None
    if backend == "fixture":
        runtime: Any = FixtureRuntime(_fixture_answers(catalog))
    elif backend == "transformers":
        runtime = QwenTransformersRuntime()
        runtime.load(
            model,
            RuntimeConfig(
                name="transformers",
                options={
                    "device_map": "auto",
                    "trust_remote_code": True,
                },
            ),
        )
        tokenizer_id = model.tokenizer_id or model.model_id
        revision = model.tokenizer_revision or "unresolved"
        context_tokenizer = InferenceTokenizer(
            backend=runtime.get_tokenizer(),
            name=f"transformers:{tokenizer_id}@{revision}",
        )
    else:
        raise ValueError(f"unsupported backend: {backend!r}")

    working_output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[TrialResult] = []
    conditions = planned_conditions(phase)
    repeats = REPEATS[phase]
    try:
        try:
            runner = EvaluationRunner(
                runtime=runtime,
                model=model,
                scorer=CalibratedAnswerScorer(),
                experiment_id=EXPERIMENT_ID,
                output_path=working_output_path,
            )
            for condition in conditions:
                results.extend(
                    runner.run(
                        build_tasks(
                            catalog,
                            condition,
                            fixture_seed=fixture_seed,
                            tokenizer=context_tokenizer,
                        ),
                        repeats=repeats,
                        condition_id=condition.condition_id,
                        sampling=SamplingConfig(max_new_tokens=32, temperature=0.0),
                    )
                )
        finally:
            runtime.close()
    except BaseException:
        _remove_temporary(temporary_output)
        _remove_temporary(temporary_manifest)
        raise

    if temporary_output is not None:
        os.replace(temporary_output, output_path)
    manifest = _manifest(
        phase=phase,
        backend=backend,
        output_path=output_path,
        manifest_path=manifest_path,
        conditions=conditions,
        catalog=catalog,
        repeats=repeats,
        results=results,
        fixture_seed=fixture_seed,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        (temporary_manifest or manifest_path).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if temporary_manifest is not None:
            os.replace(temporary_manifest, manifest_path)
    except BaseException:
        _remove_temporary(temporary_manifest)
        raise
    return manifest


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    return temporary_path


def _remove_temporary(path: Path | None) -> None:
    if path is not None:
        path.unlink(missing_ok=True)


def _manifest(
    *,
    phase: str,
    backend: str,
    output_path: Path,
    manifest_path: Path,
    conditions: Iterable[Condition],
    catalog: TaskCatalog,
    repeats: int,
    results: Iterable[TrialResult],
    fixture_seed: int,
) -> dict[str, Any]:
    result_list = list(results)
    condition_list = list(conditions)
    by_cell: dict[tuple[str, str], list[TrialResult]] = {}
    for result in result_list:
        by_cell.setdefault(
            (result.input["task_type"], result.input["condition_id"]), []
        ).append(result)

    coverage: list[dict[str, Any]] = []
    for condition in condition_list:
        for task_type in TASK_TYPES:
            cell_results = by_cell.get((task_type, condition.condition_id), [])
            statuses = Counter(result.status.value for result in cell_results)
            scored_n = sum(
                result.score.get("correct") is not None for result in cell_results
            )
            independent_task_n = sum(
                task.task_type == task_type for task in catalog.tasks
            )
            expected_trial_n = independent_task_n * repeats
            coverage.append(
                {
                    "task_type": task_type,
                    "condition_id": condition.condition_id,
                    "target_context_tokens": condition.target_context_tokens,
                    "requested_evidence_position": condition.evidence_position,
                    "independent_task_n": independent_task_n,
                    "expected_trial_n": expected_trial_n,
                    "trial_n": len(cell_results),
                    "scored_n": scored_n,
                    "statuses": dict(sorted(statuses.items())),
                    "status": (
                        "valid"
                        if len(cell_results) == expected_trial_n
                        and scored_n == expected_trial_n
                        else "excluded"
                    ),
                }
            )

    raw_sha256 = _sha256(output_path)
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scorer_version": SCORER_VERSION,
        "phase": phase,
        "backend": backend,
        "fixture_seed": fixture_seed,
        "task_catalog": str(TASK_CATALOG.relative_to(ROOT)),
        "task_ids": list(catalog.ids),
        "repeats": repeats,
        "planned_condition_n": len(condition_list),
        "planned_cell_n": len(coverage),
        "planned_trial_n": len(condition_list) * len(catalog.tasks) * repeats,
        "actual_trial_n": len(result_list),
        "raw_results": str(output_path.relative_to(ROOT)),
        "raw_results_sha256": raw_sha256,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "environment": capture_environment(ROOT),
        "coverage": coverage,
        "excluded_cells": [
            row for row in coverage if row["status"] == "excluded"
        ],
        "interpretation": (
            "Fixture backend validates task construction, scoring, storage, and "
            "coverage only; it is not a Qwen measurement."
            if backend == "fixture"
            else "Results are model/runtime observations under the recorded environment."
        ),
    }


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
    parser.add_argument("--phase", choices=tuple(REPEATS), default="smoke")
    parser.add_argument("--backend", choices=("fixture", "transformers"), default="fixture")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fixture-seed", type=int, default=42)
    parser.add_argument(
        "--overwrite-smoke",
        action="store_true",
        help="atomically regenerate existing fixture smoke artifacts",
    )
    args = parser.parse_args(argv)
    output_path = args.output or ROOT / "experiments/exp_001-context_measurement/results/raw/{phase}-trials.jsonl".format(phase=args.phase)
    manifest_path = args.manifest or ROOT / "experiments/exp_001-context_measurement/results/manifests/{phase}.json".format(phase=args.phase)
    manifest = run_experiment(
        phase=args.phase,
        backend=args.backend,
        output_path=output_path,
        manifest_path=manifest_path,
        fixture_seed=args.fixture_seed,
        overwrite_smoke=args.overwrite_smoke,
    )
    print(json.dumps({key: manifest[key] for key in (
        "phase", "backend", "actual_trial_n", "raw_results", "manifest_path"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
