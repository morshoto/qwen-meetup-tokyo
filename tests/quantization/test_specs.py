import unittest

from llm_lab.quantization import (
    ArtifactProvenance,
    QuantizationManifest,
    QuantizationVariant,
)


def artifact() -> ArtifactProvenance:
    return ArtifactProvenance(
        source_uri="hf://Qwen/Qwen3.8-27B",
        source_revision="model-sha",
        conversion_command="convert_hf_to_gguf.py --outfile model-f16.gguf",
        converter_revision="llama.cpp-sha",
        artifact_uri="file:///models/model-q8_0.gguf",
        artifact_sha256="a" * 64,
        artifact_size_bytes=123,
    )


def variant(condition_id: str = "q8_0") -> QuantizationVariant:
    return QuantizationVariant(
        condition_id=condition_id,
        label="Q8",
        format="GGUF",
        quantization_type="Q8_0",
        bits=8,
        artifact=artifact(),
        runtime_kernel="ggml",
    )


class QuantizationManifestTests(unittest.TestCase):
    def test_manifest_preserves_variant_provenance_and_controls(self) -> None:
        manifest = QuantizationManifest(
            experiment_id="exp_002",
            model_id="Qwen/Qwen3.8-27B",
            model_revision="model-sha",
            tokenizer_id="Qwen/Qwen3.8-27B",
            tokenizer_revision="tokenizer-sha",
            runtime_name="llama.cpp",
            runtime_version="llama-cpp-python-sha",
            prompt_id="prompt.qa.v001",
            task_ids=("task.literal.000001", "task.semantic.000001"),
            context_lengths=(8192, 32768),
            sampling={"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 64},
            variants=(variant(),),
            repeats=5,
        )

        record = manifest.to_record()
        restored = QuantizationManifest.from_record(record)

        self.assertEqual(manifest, restored)
        self.assertEqual("model-sha", record["variants"][0]["artifact"]["source_revision"])
        self.assertEqual(123, record["variants"][0]["artifact"]["artifact_size_bytes"])
        self.assertEqual((8192, 32768), restored.context_lengths)

    def test_manifest_rejects_duplicate_condition_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate condition_id"):
            QuantizationManifest(
                experiment_id="exp_002",
                model_id="Qwen/Qwen3.8-27B",
                model_revision="model-sha",
                tokenizer_id="Qwen/Qwen3.8-27B",
                tokenizer_revision="tokenizer-sha",
                runtime_name="llama.cpp",
                runtime_version="llama-cpp-python-sha",
                prompt_id="prompt.qa.v001",
                task_ids=("task.literal.000001",),
                context_lengths=(8192,),
                sampling={"temperature": 0.0},
                variants=(variant(), variant()),
            )

    def test_variant_rejects_untraceable_artifact_hash(self) -> None:
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            ArtifactProvenance(
                source_uri="hf://Qwen/Qwen3.8-27B",
                source_revision="model-sha",
                conversion_command="convert_hf_to_gguf.py",
                converter_revision="llama.cpp-sha",
                artifact_uri="file:///models/model-q8_0.gguf",
                artifact_sha256="not-a-hash",
                artifact_size_bytes=123,
            )


if __name__ == "__main__":
    unittest.main()
