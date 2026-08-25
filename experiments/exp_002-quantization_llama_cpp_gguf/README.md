# exp_002-quantization_llama_cpp_gguf

## Goal

Measure the capability/performance trade-off of GGUF weight quantization using
`llama.cpp` through the `llama-cpp-python` binding while keeping benchmark tasks,
prompts, runtime controls, and sampling reproducible.

The primary comparison is Q8_0, Q6_K, Q5_K_M, and Q4_K_M, with an F16
reference added when the same conversion path produces it. These are GGUF
weight formats executed by the same ggml kernel family; `Q8`, `Q6`, `Q5`, and
`Q4` are not treated as interchangeable labels for other quantizers.

## Artifact procedure and provenance

Do not commit the model weights. For every condition, create a resolved
manifest from [`manifest.template.json`](manifest.template.json) and record:

- the exact base-model and tokenizer revisions;
- the `llama.cpp` converter and quantizer revision;
- the complete conversion command;
- the artifact URI, SHA-256 digest, and byte size; and
- the runtime package/commit and kernel options.

The controlled production procedure is:

1. Pin a `llama.cpp` checkout and obtain the exact `Qwen/Qwen3.8-27B` source
   revision.
2. Convert the source weights once to GGUF F16 with
   `convert_hf_to_gguf.py`.
3. Derive Q8_0, Q6_K, Q5_K_M, and Q4_K_M from that F16 artifact using
   `llama-quantize` from the same checkout.
4. Hash and size every resulting artifact before the run. A placeholder hash
   is not valid evidence and must not be used in a resolved manifest.

The four quantized artifacts must use the same model revision, conversion
revision, tokenizer, and source procedure. Any unavailable variant is recorded
as unavailable with a reason rather than silently substituted.

## Initial questions

- How do memory, prefill throughput, decode throughput, and task accuracy change by precision?
- Which capabilities degrade first?
- Is degradation different at short (8,192) versus medium (32,768) context?

## Fixed controls

- Shared tasks: `data/tasks/core.v001.jsonl`.
- Shared prompt: `data/prompts/prompt.qa.v001.txt` (`prompt.qa.v001`).
- Context lengths: 8,192 and 32,768 tokens.
- Sampling: greedy decoding, `temperature: 0.0`, `top_p: 1.0`, seed 42, and
  `max_new_tokens: 64`.
- Runtime: the same `llama-cpp-python` version, ggml kernel options, context
  size, batch size, GPU-layer setting, and flash-attention setting for every
  variant.
- Repeats: five per task/condition/context cell.

## Measurements

The runner records:

- weight/artifact footprint: the resolved GGUF file byte size;
- peak memory: process peak RSS and its measurement method;
- TTFT: time from `generate` call until the first streamed chunk;
- prefill throughput: prompt tokens divided by the stream-derived TTFT;
- decode throughput: completion tokens divided by elapsed time after TTFT; and
- task accuracy: deterministic scoring from the shared task definitions.

The runtime records `timing_source: first_stream_chunk`. This is a portable
proxy for prefill timing, not a backend-native kernel counter. Do not compare
it with a backend's native prefill timing without labelling the difference.
Record OOM, timeout, invalid-output, and other runtime failures as statuses;
do not remove failed cells from the denominator without explanation.

## Analysis

Run `analysis.ipynb` only after a resolved manifest and processed summary CSV
are present under `results/`. It produces accuracy-vs-memory and separate
prefill/decode-vs-memory comparisons, then recommends the smallest measured
artifact within the declared accuracy tolerance of the best measured accuracy.
With missing artifacts, missing cells, or missing metrics it fails loudly.

The notebook is an analysis surface, not the benchmark runner. No conclusion
or recommendation is valid until it is based on measured raw trials tied to a
resolved manifest.
