"""Run a paired practical-equivalence check on the exp_002 capability matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_lab.analysis import paired_equivalence_report  # noqa: E402
from llm_lab.evaluation import load_trial_results  # noqa: E402


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = EXPERIMENT_ROOT / "results/raw/full-capability.jsonl"
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "results/manifest.full.json"
DEFAULT_JSON = EXPERIMENT_ROOT / "results/processed/q4-q8-equivalence.json"
DEFAULT_CSV = EXPERIMENT_ROOT / "results/processed/q4-q8-equivalence.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, metrics: list[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "metric",
        "score_field",
        "pair_n",
        "reference_success_n",
        "candidate_success_n",
        "observed_difference",
        "ci_low",
        "ci_high",
        "equivalence_margin",
        "confidence",
        "bootstrap_repeats",
        "bootstrap_seed",
        "discordant_pair_n",
        "decision",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in metrics)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-repeats", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    raw = args.raw.resolve()
    manifest = args.manifest.resolve()
    if not raw.is_file():
        raise FileNotFoundError(f"raw input does not exist: {raw}")
    if not manifest.is_file():
        raise FileNotFoundError(f"control manifest does not exist: {manifest}")
    trials = load_trial_results(raw)
    report = paired_equivalence_report(
        trials,
        reference_variant="Q8_0",
        candidate_variant="Q4_K_M",
        margin=args.margin,
        confidence=args.confidence,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    control_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    report.update(
        {
            "source": {
                "raw_results": str(raw),
                "raw_results_sha256": sha256(raw),
                "control_manifest": str(manifest),
                "control_manifest_sha256": sha256(manifest),
                "scorer_version": control_manifest["controls"]["scorer_version"],
                "task_catalog": control_manifest["controls"]["task_catalog"],
                "task_catalog_sha256": control_manifest["controls"]["task_catalog_sha256"],
                "context_lengths": control_manifest["controls"]["context_lengths"],
                "evidence_positions": [0.50],
                "capability_repeats": control_manifest["controls"]["capability_repeats"],
            },
            "scope": (
                "Matched Q8_0/Q4_K_M capability trials in the exp_002 synthetic "
                "catalog; practical equivalence is metric-specific and does not "
                "prove exact equality or generalize to other tasks."
            ),
        }
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.csv, report["metrics"])
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.json} and {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

