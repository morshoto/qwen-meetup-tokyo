import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESOLVER_PATH = ROOT / "experiments/exp_002-quantization_llama_cpp_gguf/resolve_manifest.py"
spec = importlib.util.spec_from_file_location("exp002_resolve_manifest", RESOLVER_PATH)
assert spec is not None and spec.loader is not None
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


class Exp002ManifestResolverTests(unittest.TestCase):
    def test_resolver_records_real_artifact_identity_and_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_directory = root / "artifacts"
            artifact_directory.mkdir()
            artifacts = {}
            for condition_id in ("q8_0", "q6_k", "q5_k_m", "q4_k_m"):
                path = artifact_directory / f"{condition_id}.gguf"
                path.write_bytes(f"fixture-{condition_id}".encode())
                artifacts[condition_id] = path
            output_path = root / "manifest.json"

            record = resolver.resolve_manifest(
                template_path=ROOT / "experiments/exp_002-quantization_llama_cpp_gguf/manifest.template.json",
                output_path=output_path,
                model_revision="model-commit",
                tokenizer_revision="tokenizer-commit",
                runtime_version="llama-cpp-python==0.3.16",
                converter_revision="converter-commit",
                artifact_paths=artifacts,
                commands={
                    condition_id: f"convert-and-quantize {condition_id}"
                    for condition_id in artifacts
                },
            )

            self.assertFalse(record["template"])
            self.assertEqual("model-commit", record["model"]["revision"])
            self.assertEqual("tokenizer-commit", record["model"]["tokenizer_revision"])
            self.assertEqual(
                "llama-cpp-python==0.3.16", record["runtime"]["version"]
            )
            written = json.loads(output_path.read_text(encoding="utf-8"))
            for variant in written["variants"]:
                artifact = variant["artifact"]
                path = artifacts[variant["condition_id"]]
                self.assertEqual("model-commit", artifact["source_revision"])
                self.assertEqual(path.stat().st_size, artifact["artifact_size_bytes"])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    artifact["artifact_sha256"],
                )
                self.assertEqual(
                    f"artifacts/{variant['condition_id']}.gguf",
                    artifact["artifact_uri"],
                )

    def test_resolver_fails_closed_when_a_variant_is_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = {
                condition_id: root / f"{condition_id}.gguf"
                for condition_id in ("q8_0", "q6_k", "q5_k_m")
            }
            for path in artifacts.values():
                path.write_bytes(b"fixture")

            with self.assertRaisesRegex(ValueError, "artifact paths"):
                resolver.resolve_manifest(
                    template_path=ROOT / "experiments/exp_002-quantization_llama_cpp_gguf/manifest.template.json",
                    output_path=root / "manifest.json",
                    model_revision="model-commit",
                    tokenizer_revision="tokenizer-commit",
                    runtime_version="runtime",
                    converter_revision="converter-commit",
                    artifact_paths=artifacts,
                    commands={condition_id: "command" for condition_id in artifacts},
                )


if __name__ == "__main__":
    unittest.main()
