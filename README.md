# qwen-meetup-tokyo

**Quantization, Long Context, and Agent Reliability in Qwen3.8-27B**

Research and presentation repository for a Qwen Meetup Tokyo talk investigating when a local LLM starts to break under long context, quantization, and agentic workloads.

## Repository layout

```text
.
├── docs/                         # research plan, methodology, literature, findings
├── data/                         # shared prompts, tasks, fixtures, corpora
├── experiments/                  # numbered, self-contained experiments
│   ├── _template/
│   ├── exp_001-context_measurement/
│   └── exp_002-quantization_llama_cpp_gguf/
├── src/llm_lab/                  # reusable local-LLM research library
├── tests/                        # tests for reusable library code
├── pyproject.toml
└── LICENSE
```

## Design principles

- Experiments are numbered, self-contained research units.
- Shared prompts and benchmark fixtures live in `data/`, not inside reusable library code.
- Each experiment owns an `analysis.ipynb` for visualization and deeper interpretation.
- `src/llm_lab/` is model-agnostic reusable infrastructure; Qwen-specific behavior should be implemented through adapters/configuration rather than baked into the package structure.
- Experiment code should be thin: reusable logic belongs in `src/`.
- Raw measurements, processed results, and figures should remain traceable to the experiment that produced them.

See `experiments/README.md`, `data/README.md`, and `src/llm_lab/README.md` for conventions.
