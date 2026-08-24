# Engineering architecture

**Issues:** #5 and #6

The reusable code under `src/llm_lab/` should support this research program without hard-coding Qwen, one runtime, or one experiment.

## 1. Architectural rule

Experiment directories describe **what to test**. `src/llm_lab/` implements **how to run reusable mechanics**.

Dependency direction:

```text
experiments/*
     |
     v
src/llm_lab/*
     |
     +--> runtime libraries / OS interfaces
     +--> data files under data/
```

Forbidden direction:

```text
src/llm_lab/*  --> experiments/exp_XXX/*
```

Reusable library code must never import one numbered experiment.

## 2. Planned package boundaries

```text
src/llm_lab/
├── models/
├── runtimes/
├── generation/
├── context/
├── quantization/
├── agents/
├── evaluation/
├── datasets/
├── telemetry/
├── analysis/
└── utils/
```

### `models/`

Responsibilities:

- model identifier/revision metadata;
- tokenizer identifier/revision;
- declared capabilities;
- context-limit metadata;
- chat-template/model adapter where necessary;
- model registry/config parsing.

Avoid placing runtime execution code here.

Possible interface shape:

```python
@dataclass(frozen=True)
class ModelSpec:
    id: str
    revision: str | None
    tokenizer_id: str | None
    max_context_tokens: int | None
    modalities: tuple[str, ...]
```

Exact API is not fixed yet; this illustrates separation of concerns.

### `runtimes/`

Responsibilities:

- load/unload a model artifact;
- tokenize/generate through a backend;
- expose backend timing fields where possible;
- normalize runtime configuration;
- identify runtime package/version;
- optional memory hooks.

Examples of future adapters may include MLX-family runtimes, llama.cpp-compatible runtimes, or other local inference engines. The first implementation should target the runtime actually used for Qwen3.8-27B experiments.

### `generation/`

Define backend-independent request/response objects.

Suggested concepts:

```python
GenerationRequest
GenerationResponse
TokenUsage
GenerationTiming
SamplingConfig
```

The response should contain **raw text and metadata**, not a task-specific correctness decision.

### `context/`

Responsibilities:

- token-aware filler/context construction;
- evidence placement;
- prompt token accounting;
- context-window validation;
- multi-span evidence metadata;
- agent-history measurements.

This package is central to exp_001/003/004.

Required property: given the same seed/config/tokenizer revision, context construction should be reproducible.

### `quantization/`

Responsibilities:

- normalized quantization metadata;
- artifact provenance;
- bit/format labels;
- helper comparison metadata.

Do **not** assume `q4` is enough metadata. Store method/format/grouping/runtime-relevant details where available.

### `agents/`

Responsibilities:

- tool schema types;
- tool registry;
- deterministic test tools;
- trajectory/message representation;
- agent loop primitives;
- stop conditions;
- trajectory serialization.

Keep the first agent harness deliberately minimal. We are measuring model behavior, not building a production agent platform.

### `evaluation/`

Responsibilities:

- task interface;
- scorer interface;
- trial execution;
- repeated-run orchestration;
- status/error records;
- task-specific metrics.

Generation and scoring must be separate so outputs can be rescored without rerunning the model.

### `datasets/`

Responsibilities:

- load shared files from `data/`;
- validate schemas;
- resolve task IDs;
- produce typed task/context objects.

Do not embed large datasets in Python source.

### `telemetry/`

Responsibilities:

- wall-clock timing;
- runtime-reported prompt/decode timing;
- memory measurement abstraction;
- environment snapshot;
- optional energy measurement;
- model/runtime load timing.

Every metric should include measurement provenance/method.

### `analysis/`

Responsibilities:

- raw result loading;
- validation;
- aggregation;
- confidence intervals;
- effective-context calculations;
- common plotting-table preparation;
- failure taxonomy aggregation.

Notebooks should call this package rather than duplicate statistical calculations across experiments.

### `utils/`

Only small truly cross-cutting helpers. Do not turn this into a dumping ground.

## 3. Core interfaces

### Runtime adapter

A useful minimum contract:

```python
class Runtime(Protocol):
    def load(self, model: ModelSpec, config: RuntimeConfig) -> None: ...
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...
    def close(self) -> None: ...
```

Async/batched variants can be introduced later if needed.

### Task

```python
class Task(Protocol):
    id: str
    def build_request(...) -> GenerationRequest: ...
```

### Scorer

```python
class Scorer(Protocol):
    def score(self, task, response) -> ScoreResult: ...
```

### Trial record

Trial records are append-only experiment evidence. See `data-and-result-contracts.md`.

## 4. CLI / execution shape

The project should eventually support a simple command such as:

```bash
python -m llm_lab.run \
  --config experiments/exp_001-context_measurement/config.yaml
```

or an equivalent installed CLI.

The exact CLI name is secondary; critical behavior is:

1. resolve config;
2. snapshot git/runtime/model metadata;
3. load task instances;
4. execute trials;
5. append raw result records safely;
6. generate or update processed summaries;
7. never overwrite prior main-run evidence without explicit force/versioning.

## 5. Configuration

Experiment YAML should be declarative.

Avoid putting logic such as “if position == 0.5 do X” inside config files. Config describes parameter values; Python implements algorithms.

A resolved config snapshot should be written with results so later edits to the source YAML do not change the interpretation of old runs.

## 6. Runtime reproducibility

For every run record or batch manifest, capture:

- runtime name/version;
- Python/package lock state if available;
- model artifact source/revision;
- quantization metadata;
- tokenizer revision;
- repository git SHA;
- operating system/hardware summary.

## 7. Error handling

Expected runtime problems include:

- out of memory;
- prompt exceeds runtime limit;
- invalid tokenizer/model combination;
- backend crash;
- malformed tool call;
- timeout;
- unsupported metric.

Errors should become structured result statuses rather than exceptions that erase the trial history.

The runner may still fail fast on configuration/programming errors before trials begin.

## 8. Result writing

Recommended pattern:

```text
results/
├── manifest.json
├── resolved-config.yaml
├── raw/
│   └── trials.jsonl
├── processed/
│   ├── summary.parquet
│   └── summary.csv
└── figures/
```

The repository currently ignores `results/raw/` by default. If raw data remains outside Git, `manifest.json` must point to the durable location/hash or describe regeneration.

## 9. Testing strategy

### Unit tests

Should not require downloading a 27B model.

Test:

- token-aware placement using a fake/simple tokenizer;
- schema validation;
- score normalization;
- result serialization;
- aggregation/effective-context calculation;
- tool-call parsing;
- failure record behavior.

### Integration tests

Use a tiny local model, mock runtime, or deterministic fake runtime to validate end-to-end runner behavior.

### Hardware/model smoke tests

Run manually or under a clearly tagged environment. They should not be mandatory for normal CI.

## 10. Notebook boundary

`analysis.ipynb` is for exploration and communication:

- load processed/raw results;
- validate missing cells;
- inspect distributions;
- call reusable analysis functions;
- build presentation-quality figures;
- annotate representative examples.

It should **not**:

- own the inference loop;
- contain the only scorer implementation;
- be required to construct benchmark prompts;
- hide manual changes to result data.

## 11. Engineering done-state before exp_001 main run

At minimum:

- one local runtime adapter works;
- common generation request/response exists;
- shared tasks can be loaded;
- context generator can hit target token length/position;
- scorer produces deterministic records;
- raw JSONL writes safely;
- environment/model metadata is captured;
- notebook can load processed results;
- smoke test completes at short and one long context.
