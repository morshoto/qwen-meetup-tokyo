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

The checked-in resolver computes the immutable file fields and refuses an
incomplete artifact set:

```bash
PYTHONPATH=src python3 experiments/exp_002-quantization_llama_cpp_gguf/resolve_manifest.py \
  --template experiments/exp_002-quantization_llama_cpp_gguf/manifest.template.json \
  --output experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json \
  --model-revision MODEL_COMMIT \
  --tokenizer-revision TOKENIZER_COMMIT \
  --runtime-version 'llama-cpp-python==PACKAGE_VERSION' \
  --converter-revision LLAMA_CPP_COMMIT \
  --artifact q8_0=experiments/exp_002-quantization_llama_cpp_gguf/results/artifacts/q8_0.gguf \
  --artifact q6_k=experiments/exp_002-quantization_llama_cpp_gguf/results/artifacts/q6_k.gguf \
  --artifact q5_k_m=experiments/exp_002-quantization_llama_cpp_gguf/results/artifacts/q5_k_m.gguf \
  --artifact q4_k_m=experiments/exp_002-quantization_llama_cpp_gguf/results/artifacts/q4_k_m.gguf \
  --command 'q8_0=COMPLETE_CONVERSION_AND_QUANTIZATION_COMMAND' \
  --command 'q6_k=COMPLETE_CONVERSION_AND_QUANTIZATION_COMMAND' \
  --command 'q5_k_m=COMPLETE_CONVERSION_AND_QUANTIZATION_COMMAND' \
  --command 'q4_k_m=COMPLETE_CONVERSION_AND_QUANTIZATION_COMMAND'
```

Use the same raw output for the pilot and full run: the runner fingerprints
the resolved manifest, appends only missing trial IDs, and rejects mismatched
or out-of-scope records. The pilot is Q8_0 × 8,192 × all three tasks × one
repeat (3 trials); the complete matrix is 4 × 2 × 3 × 5 (120 trials):

```bash
PYTHONPATH=src python3 experiments/exp_002-quantization_llama_cpp_gguf/runner.py \
  --manifest experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json \
  --condition-id q8_0 --context-length 8192 --repeats 1

PYTHONPATH=src python3 experiments/exp_002-quantization_llama_cpp_gguf/runner.py \
  --manifest experiments/exp_002-quantization_llama_cpp_gguf/results/manifest.json
```

## Initial questions

- How do memory, prefill throughput, decode throughput, and task accuracy change by precision?
- Which capabilities degrade first?
- Is degradation different at short (8,192) versus medium (32,768) context?

## Fixed controls

- Shared tasks: `data/tasks/core.v001.jsonl`.
- Shared prompt: `data/prompts/prompt.qa.v001.txt` (`prompt.qa.v001`).
- Context lengths: 8,192 and 32,768 input tokens. The prompt/template
  convention is explicit in the resolved manifest as
  `context_length_semantics: input_tokens`.
- Runtime context budget: `n_ctx=33088`, which is the largest input condition
  plus `max_new_tokens=64` and a 256-token overhead margin.
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
- stream TTFT: `stream_ttft_s`, from `generate` call until the first
  streamed chunk;
- prompt-throughput proxy: `prompt_throughput_proxy_tok_s`, prompt tokens
  divided by stream TTFT;
- post-first-chunk output throughput:
  `post_first_chunk_output_tok_s`, completion tokens divided by elapsed time
  after the first chunk; and
- scored accuracy, end-to-end success (`correct / attempted`), and failure
  rate, all retained with their denominators.

The runtime records `timing_source: first_stream_chunk`. The two throughput
fields above are portable stream-derived proxies, not backend-native kernel
counters. Do not compare them with native prefill/decode timings without
labelling the difference.
Record OOM, timeout, invalid-output, and other runtime failures as statuses;
do not remove failed cells from the denominator without explanation.

## Analysis

Run `analysis.ipynb` only after a resolved manifest and processed summary CSV
are present under `results/`. It produces end-to-end-success-vs-memory and
separate prompt-throughput-proxy/post-first-chunk-output-vs-memory
comparisons, then recommends the smallest measured artifact within the
declared tolerance of the best measured end-to-end success. A condition with
runtime or invalid-output failures cannot hide those failures by reporting
only scored rows.
With missing artifacts, missing cells, or missing metrics it fails loudly.

The notebook is an analysis surface, not the benchmark runner. No conclusion
or recommendation is valid until it is based on measured raw trials tied to a
resolved manifest.
