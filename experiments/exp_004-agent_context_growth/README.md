# exp_004-agent_context_growth

## Goal

Measure whether a local Qwen agent can reuse a critical fact discovered during
an earlier tool interaction after its own trajectory accumulates deterministic
distractor observations. The primary controls are trajectory length, the
relative position of the critical observation, and the practical quantization
variants selected from the resolved `exp_003` manifest.

## Controlled task

Each task has a fixed objective, one critical observation, and a versioned list
of unrelated deterministic tool outputs. The harness places the critical
observation after a controlled number of distractor observations, then appends
the remaining distractors before asking the model for the final answer. The
trajectory length includes the critical observation. The same task seed,
control, observation list, and environment fingerprint are reused across
selected quantization variants.

The model emits strict JSON actions:

```json
{"action":"tool","name":"discover_fact","arguments":{}}
{"action":"answer","value":"middleware/auth.ts"}
```

Every model action, controller observation, tool result, parse failure, and
recovery is retained in the raw trial record.

## Protocol and provenance

[`config.yaml`](config.yaml) is the committed protocol. It is JSON-compatible
YAML so the runner can load it with the Python standard library in a clean
checkout. The default selected variants are `q8_0` and `q4_k_m`; their complete
artifact provenance is inherited from a resolved `exp_003` manifest rather than
re-running `exp_002`. The source manifest's task IDs describe the upstream
context/quantization study; exp_004's agent task IDs come from its own committed
agent catalog.

Real pilot/main runs require that source manifest, local artifacts, and a
compatible `llama-cpp-python` runtime. The runner verifies artifact paths for
non-fixture execution and records the effective runtime options.

## Smoke, pilot, and main

Smoke uses the deterministic fixture backend and is a harness check only. It
must never be copied into `docs/findings.md` as model evidence. Pilot and main
use the selected source artifacts when available. Runtime failures stay in raw
results and attempted denominators.

Example smoke command:

```bash
PYTHONPATH=src python3 experiments/exp_004-agent_context_growth/runner.py \
  --source-manifest experiments/exp_003-context_x_quantization/results/manifest.json \
  --output experiments/exp_004-agent_context_growth/results/raw/smoke-trials.jsonl \
  --manifest experiments/exp_004-agent_context_growth/results/manifests/smoke.json \
  --processed experiments/exp_004-agent_context_growth/results/processed/smoke-summary.csv \
  --phase smoke --backend fixture
```

## Analysis

The notebook requires a resolved run manifest and processed/raw results. It
rejects fixture-only inputs, validates the complete declared dimension product,
and plots final success and critical-fact reuse against trajectory length and
critical-information position. It also reports tool-call validity, repeated
actions, recoveries, input-token totals, and observable failure categories:
`retrieval`, `state_tracking`, `tool_planning`, `runtime`, and `success`.

Results are descriptive. The notebook does not claim that a failure category is
causal or statistically significant without a predeclared repeated measurement
design and sufficient observations.

## Findings handoff

After a real measured run, record these conclusions in `docs/findings.md` only
with the manifest, sample counts, processed summary, and figures attached:

- Long-horizon degradation: observed, not observed, or insufficient evidence;
- **Lost in the Agent**: observed, not observed, or insufficient evidence; and
- the exact model/runtime/quantization, task family, trajectory-length range,
  critical-position conditions, and failure categories supporting the statement.

Until then, `docs/findings.md` remains `exp_004: not yet measured`.
