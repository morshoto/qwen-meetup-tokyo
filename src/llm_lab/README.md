# llm_lab

Reusable, model-agnostic infrastructure for local LLM research.

Planned module boundaries:

```text
llm_lab/
├── models/          # model metadata, adapters, capabilities, registries
├── runtimes/        # MLX, llama.cpp, vLLM, etc. backend adapters
├── generation/      # common request/generation interfaces
├── context/         # context construction, placement, token accounting
├── quantization/    # quantization metadata and comparison helpers
├── agents/          # tool schemas, trajectories, agent harness primitives
├── evaluation/      # scorers, task runners, repeated trials
├── datasets/        # loaders for versioned data/ task definitions
├── telemetry/       # latency, throughput, memory, power/runtime metadata
├── analysis/        # reusable aggregation/statistics/plot preparation
└── utils/           # small cross-cutting utilities
```

The package must not assume Qwen. Qwen-specific code should live behind model/runtime adapters so experiments can later compare other local models without restructuring the project.

Experiment directories may import this package, but reusable package code must not import experiment-specific modules.
