"""Resolve exp_002's manifest from materialized GGUF artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Mapping


def resolve_manifest(
    *,
    template_path: Path,
    output_path: Path,
    model_revision: str,
    tokenizer_revision: str,
    runtime_version: str,
    converter_revision: str,
    artifact_paths: Mapping[str, Path],
    commands: Mapping[str, str],
    task_catalog_path: Path | None = None,
) -> dict[str, object]:
    """Hash every artifact and write a complete manifest atomically."""

    template_path = template_path.resolve()
    output_path = output_path.resolve()
    if template_path == output_path:
        raise ValueError("resolved manifest must not overwrite its template")
    record = json.loads(template_path.read_text(encoding="utf-8"))
    if record.get("template") is not True:
        raise ValueError("input manifest must be marked as a template")
    for name, value in (
        ("model_revision", model_revision),
        ("tokenizer_revision", tokenizer_revision),
        ("runtime_version", runtime_version),
        ("converter_revision", converter_revision),
    ):
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")

    variants = record.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("template must declare variants")
    condition_ids = tuple(str(item["condition_id"]) for item in variants)
    if set(artifact_paths) != set(condition_ids):
        raise ValueError(
            "artifact paths must be provided exactly once for every variant: "
            f"expected {sorted(condition_ids)}, got {sorted(artifact_paths)}"
        )
    if set(commands) != set(condition_ids):
        raise ValueError(
            "commands must be provided exactly once for every variant: "
            f"expected {sorted(condition_ids)}, got {sorted(commands)}"
        )

    catalog_reference, resolved_catalog_path = _resolve_task_catalog(
        template_path=template_path,
        record=record,
        task_catalog_path=task_catalog_path,
    )

    record["template"] = False
    record["model"]["revision"] = model_revision
    record["model"]["tokenizer_revision"] = tokenizer_revision
    record["runtime"]["version"] = runtime_version
    if resolved_catalog_path is not None:
        record["controls"]["task_catalog"] = catalog_reference
        record["controls"]["task_catalog_sha256"] = _sha256(resolved_catalog_path)
    record["controls"]["scorer_version"] = "calibrated.v1"
    for variant in variants:
        condition_id = str(variant["condition_id"])
        artifact_path = artifact_paths[condition_id].resolve()
        command = commands[condition_id].strip()
        if not command:
            raise ValueError(f"conversion command must be non-empty for {condition_id}")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"artifact is missing for {condition_id}: {artifact_path}")
        size = artifact_path.stat().st_size
        if size < 1:
            raise ValueError(f"artifact is empty for {condition_id}: {artifact_path}")
        artifact = variant["artifact"]
        artifact["source_revision"] = model_revision
        artifact["conversion_command"] = command
        artifact["converter_revision"] = converter_revision
        artifact["artifact_uri"] = _artifact_uri(artifact_path, output_path.parent)
        artifact["artifact_sha256"] = _sha256(artifact_path)
        artifact["artifact_size_bytes"] = size

    if _contains_placeholder(record):
        raise ValueError("resolved manifest still contains placeholder values")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, output_path)
    return record


def _artifact_uri(path: Path, manifest_directory: Path) -> str:
    try:
        return path.relative_to(manifest_directory).as_posix()
    except ValueError:
        return path.as_uri()


def _resolve_task_catalog(
    *,
    template_path: Path,
    record: Mapping[str, object],
    task_catalog_path: Path | None,
) -> tuple[str | None, Path | None]:
    controls = record.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("manifest template controls must be an object")
    template_reference = controls.get("task_catalog")
    if task_catalog_path is None and template_reference is None:
        return None, None
    if task_catalog_path is not None:
        resolved_path = task_catalog_path.resolve()
        reference = str(task_catalog_path)
    else:
        if not isinstance(template_reference, str) or not template_reference.strip():
            raise ValueError("task_catalog must be a non-empty path")
        reference = template_reference
        candidate = Path(template_reference)
        roots = (
            template_path.parents[2],
            template_path.parent,
            Path.cwd(),
        )
        resolved_path = next(
            (root / candidate for root in roots if (root / candidate).is_file()),
            candidate,
        ).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"task catalog is missing: {resolved_path}")
    if resolved_path.stat().st_size < 1:
        raise ValueError(f"task catalog is empty: {resolved_path}")
    return reference, resolved_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_placeholder(value: object) -> bool:
    if isinstance(value, str):
        return "REPLACE_WITH" in value or "SET_TO_" in value
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def _mapping(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        condition_id, separator, payload = value.partition("=")
        if not separator or not condition_id or not payload.strip():
            raise ValueError(f"{label} must use CONDITION=VALUE: {value!r}")
        if condition_id in result:
            raise ValueError(f"duplicate {label} for {condition_id}")
        result[condition_id] = payload
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--runtime-version", required=True)
    parser.add_argument("--converter-revision", required=True)
    parser.add_argument("--task-catalog", type=Path)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="CONDITION=PATH",
    )
    parser.add_argument(
        "--command",
        action="append",
        required=True,
        metavar="CONDITION=COMMAND",
    )
    args = parser.parse_args(argv)
    artifact_values = _mapping(args.artifact, "artifact")
    command_values = _mapping(args.command, "command")
    record = resolve_manifest(
        template_path=args.template,
        output_path=args.output,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        runtime_version=args.runtime_version,
        converter_revision=args.converter_revision,
        artifact_paths={key: Path(value) for key, value in artifact_values.items()},
        commands=command_values,
        task_catalog_path=args.task_catalog,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
