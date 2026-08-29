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

## Implemented interfaces

The first reusable seams are available without installing a model backend:

```python
from llm_lab.generation import GenerationRequest, SamplingConfig
from llm_lab.models import qwen38_model_spec
from llm_lab.runtimes import QwenTransformersRuntime, RuntimeConfig

model = qwen38_model_spec()
runtime = QwenTransformersRuntime()
runtime.load(model, RuntimeConfig(name="transformers", options={"device_map": "auto"}))
response = runtime.generate(
    GenerationRequest(
        prompt="Answer with one word.",
        model=model,
        sampling=SamplingConfig(max_new_tokens=32),
    )
)
runtime.close()
```

Install the optional backend only for a local model smoke test:

```bash
python -m pip install 'llm-lab[transformers]'
```

The Transformers adapter is lazy and injectable, so normal unit tests use fake
processor/model components and do not download Qwen weights.

## Evaluation flow

The reusable runner records every attempt and can feed raw JSONL directly into
the analysis helpers:

```python
from llm_lab.analysis import aggregate_jsonl, write_summary_csv
from llm_lab.evaluation import CalibratedAnswerScorer, EvaluationRunner

runner = EvaluationRunner(
    runtime=runtime,
    model=model,
    scorer=CalibratedAnswerScorer(),
    experiment_id="exp_001",
    output_path="results/raw/trials.jsonl",
)
runner.run(tasks, repeats=3, condition_id="q8:ctx65536:p050")
write_summary_csv("results/processed/summary.csv", aggregate_jsonl("results/raw/trials.jsonl"))
```

Execution failures remain trial records with a controlled status; they are not
silently removed before aggregation.

Experiment directories may import this package, but reusable package code must not import experiment-specific modules.
