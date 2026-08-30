# Data and result contracts

**Issues:** #4 and #6

This document defines shared data structures. Exact serialization can evolve, but identifiers and provenance must remain stable enough for cross-experiment analysis.

The committed v1 fixture uses `data/tasks/core.v001.jsonl`,
`data/prompts/prompt.qa.v001.txt`, and `data/corpora/synthetic.v001.json`.
`TaskCatalog` validates the machine-checkable task metadata, while
`SyntheticContextGenerator` records the seed, tokenization convention, target
length, and evidence offsets used to reproduce a context instance.

Issue #6 formalizes the execution side of this contract: `EvaluationTask` adapts
catalog definitions into generation requests, `ExpectedAnswerScorer` scores
responses independently, `TrialResult` serializes one execution, and
`JsonlResultWriter` appends records without allowing duplicate trial IDs.
`aggregate_jsonl` produces notebook-ready summaries without counting unscored
runtime or scorer failures as accuracy observations.

Issue #24 adds the opt-in `CalibratedAnswerScorer` for separating exact
correctness, answer-bearing correctness, and output-format validity. Existing
`expected.v1` scores are not rewritten.

## 1. Directory roles

```text
data/
├── prompts/
├── tasks/
├── fixtures/
└── corpora/
```

### `data/prompts/`

Versioned system prompts, question templates, tool instructions, and task templates.

Prompt wording is part of the experimental method. Do not keep the only copy in a notebook.

### `data/tasks/`

Machine-readable benchmark task definitions and expected answers/success conditions.

### `data/fixtures/`

Small deterministic files/repositories/tool outputs used in tests and controlled agent tasks.

### `data/corpora/`

Reusable filler/context material or generators' source material.

Large generated contexts should usually be reproducible from seed rather than committed as hundreds of megabytes of text.

## 2. Stable identifiers

Use IDs that survive file movement.

Suggested patterns:

```text
task.literal.000001
task.semantic.000001
task.multihop.000001
corpus.synthetic.000001
prompt.qa.v001
prompt.agent.v001
```

Every trial should record these IDs plus content hashes/revisions when practical.

## 3. Task schema — common fields

Illustrative JSON/YAML shape:

```yaml
id: task.literal.000001
type: literal_retrieval
version: 1
question: "What is Project Aurora's access code?"
expected:
  type: exact
  value: "ZX-4817"
evidence:
  - text: "The access code assigned to Project Aurora is ZX-4817."
metadata:
  seed: 1234
  source: generated
  license: project-generated
```

Required concepts:

- stable task ID;
- task type;
- question/instruction;
- expected result or success condition;
- evidence spans or references;
- provenance/generation metadata.

## 4. Literal retrieval task

Goal: test basic access to evidence.

Properties:

- answer should be unambiguous;
- evidence has high lexical overlap with question;
- distractor text must not contain alternative valid answers;
- answer should not be guessable from general knowledge.

Prefer synthetic random identifiers/names to reduce memorization.

## 5. Semantic retrieval task

Goal: require meaning rather than string matching.

Properties:

- question and evidence should not share the answer phrase trivially;
- inference should require one clear semantic mapping;
- avoid ambiguous commonsense tasks where multiple answers are defensible;
- store accepted normalized concepts/synonyms where exact text is not appropriate.

A semantic task should still be machine-scoreable if possible.

## 6. Multi-hop task

Goal: combine multiple required evidence spans.

Example structure:

```yaml
id: task.multihop.000001
type: multi_hop
question: "What is the access code for the building containing the Orion project manager's office?"
expected:
  type: exact
  value: "8392"
evidence:
  - hop: 1
    text: "The Orion project is managed by Clara."
  - hop: 2
    text: "Clara's office is in Building Seven."
  - hop: 3
    text: "Building Seven uses access code 8392."
```

Store hop identities so analysis can inspect which span may have been missed.

## 7. Context-generation specification

A generated context instance should be reproducible from:

```text
task_id
corpus_id / generator version
seed
target_token_length
requested evidence position(s)
tokenizer revision
prompt-template revision
```

The context builder must record:

- actual input tokens after final prompt formatting;
- start/end token offsets for evidence spans;
- actual normalized evidence positions;
- filler segments used;
- any truncation/padding decisions.

## 8. Filler/corpus rules

Filler should be challenging enough to behave like real context but should not accidentally answer the task.

Potential sources:

- project-generated neutral prose;
- redistributable public-domain text;
- code/document mixtures in a separate task family.

Requirements:

- provenance/license documented;
- deterministic sampling by seed;
- scan for answer leakage where possible;
- avoid repeated exact patterns that make position detectable.

## 9. Agent task schema

Suggested concepts:

```yaml
id: task.agent.000001
type: agent_state_tracking
objective: "..."
tools:
  - read_file
  - search_repository
  - run_tests
environment_fixture: fixture.agent_repo.v001
success:
  type: state_assertion
  assertion: "..."
critical_observation:
  id: obs.auth_location
  expected_content: "middleware/auth.ts"
trajectory_controls:
  distractor_steps: 20
```

The environment should be deterministic for the controlled exp_004 task.

## 10. Repository task schema

Pin:

- repository URL or fixture ID;
- commit SHA/base revision;
- any patch used to introduce a regression;
- expected files/diagnosis;
- test command/success condition;
- allowed tools;
- time/turn budget.

Do not depend on a moving `main` branch.

## 11. Raw trial result schema

Recommended JSONL record concept:

```json
{
  "schema_version": 1,
  "trial_id": "exp_001:task.literal.000001:q8:ctx65536:p050:run01",
  "experiment_id": "exp_001",
  "task_id": "task.literal.000001",
  "status": "success",
  "model": {
    "id": "Qwen/Qwen3.8-27B",
    "revision": null,
    "artifact": null,
    "quantization": null
  },
  "runtime": {
    "name": null,
    "version": null
  },
  "input": {
    "target_context_tokens": 65536,
    "actual_input_tokens": null,
    "requested_evidence_position": 0.5,
    "actual_evidence_position": null,
    "prompt_id": "prompt.qa.v001"
  },
  "generation": {
    "output_text": "...",
    "output_tokens": null,
    "finish_reason": null
  },
  "score": {
    "correct": true,
    "value": 1.0,
    "scorer": "exact.v001"
  },
  "timing": {
    "ttft_s": null,
    "prefill_s": null,
    "decode_s": null,
    "total_s": null
  },
  "memory": {
    "peak_bytes": null,
    "measurement": null
  },
  "environment": {
    "repo_git_sha": null,
    "hardware_id": null,
    "os": null
  }
}
```

The Python implementation freezes this shape at `schema_version = 1`; new fields
should be additive and readers should reject unknown schema versions rather than
silently reinterpret old evidence.

### 11.1 Scoring calibration

There are two versioned scoring policies:

- `expected.v1` is the legacy policy. It keeps the existing exact and
  normalized-exact behaviour and remains available for reading old results.
- `calibrated.v1` is the Issue #24 policy. It keeps `score.correct` and
  `score.value` exact-compatible, and adds independent boolean fields:
  `exact_correct`, `answer_bearing_correct`, and `format_valid`.

For `calibrated.v1`:

- `exact_correct` is true only when the output equals the canonical answer
  (after the declared normalized-exact normalization for semantic tasks);
- `answer_bearing_correct` is true when a complete accepted answer occurs in
  the output with word boundaries. This is a lexical diagnostic, not a fuzzy
  or LLM-judged semantic score;
- `format_valid` checks the declared `expected.format`. `identifier` accepts
  only letters/digits separated by hyphens or underscores; `phrase` accepts
  alphanumeric words separated by spaces, hyphens, or underscores.

Semantic tasks must declare their finite allowed answer range in
`expected.accepted`. Values are normalized for comparison, and an answer is
not accepted merely because it is plausible or synonymous unless it is listed
in that range. The canonical answer remains in `expected.value`.

For example, with `expected.value: "ZX-4817"` and
`expected.format: "identifier"`, the output `ZX-4817.659` is:

```text
exact_correct           = false
answer_bearing_correct  = true
format_valid             = false
```

This distinguishes a model response that contains the expected answer from a
strictly correct, correctly formatted response. A wrong but well-formed value
such as `ZX-4818` has `format_valid = true` and both correctness fields false.

Every trial records the selected policy in `score.scorer`, including failure
records. `runtime_error` means generation did not complete; `scorer_error`
means generation completed but scoring raised an exception; and
`invalid_output` means the scorer received no scoreable output. These statuses
must not be collapsed into model correctness.

Processed summaries expose `exact_accuracy`, `answer_bearing_accuracy`, and
`format_validity` with their own scored denominators. Legacy records without
the additive fields retain their existing `accuracy` and are treated as
exact-only for the compatibility `exact_accuracy` column; missing calibrated
dimensions are not fabricated.

When migrating a run, keep the original JSONL immutable and run new trials with
`CalibratedAnswerScorer`. Do not relabel old `expected.v1` records as
`calibrated.v1`.

## 12. Trial ID

Trial IDs must be unique and deterministic enough to detect accidental duplicates.

Recommended components:

```text
experiment
task instance
model/quantization
context condition
position condition
repeat/seed
```

Do not use a random UUID as the only human-visible identity.

## 13. Result status

Use a controlled enum, for example:

```text
success
wrong_answer
invalid_output
runtime_error
out_of_memory
timeout
scorer_error
cancelled
```

A correct model answer can still have `status=success` and `score.correct=true`; a wrong answer may use `status=wrong_answer` or a more neutral `status=success` plus `correct=false`. Choose one convention in implementation and keep it consistent.

Recommended: reserve `status` for execution status and put correctness entirely under `score`, e.g.:

```text
status = completed | runtime_error | out_of_memory | timeout | ...
score.correct = true | false | null
```

This avoids confusing “wrong answer” with a runner failure.

## 14. Batch manifest

Each result directory should include a manifest describing:

- experiment ID;
- run/batch ID;
- start/end timestamps;
- source config path;
- resolved config hash;
- repository git SHA;
- model artifact(s);
- runtime version;
- hardware/environment summary;
- number of planned/completed/error trials;
- raw file path/hash;
- notes/exclusions.

## 15. Processed summaries

Processed files should be derivable from raw records.

Useful columns:

```text
experiment_id
quantization
task_type
context_tokens
evidence_position
n
accuracy
exact_accuracy
answer_bearing_accuracy
format_validity
accuracy_ci_low
accuracy_ci_high
median_ttft_s
median_prefill_tps
median_decode_tps
peak_memory_bytes
```

`peak_memory_bytes` is a sampled process-local RSS maximum when psutil is
available (`psutil.Process.memory_info.rss_sampled`). The stdlib fallback is a
process-lifetime `ru_maxrss` high-water mark. Do not compare sequentially loaded
quantization variants as independent memory measurements unless each variant is
run in an isolated process; always retain `memory_measurement` with the value.

Agent/repository summaries add task-success and trajectory metrics.

## 16. Figure provenance

Every figure used in the presentation should be reproducible from:

- one notebook/script;
- one processed/raw input set;
- a known repository commit.

Recommended output metadata next to figures:

```text
figure filename
source experiment
source result manifest
notebook/script
creation timestamp
```

## 17. Data review checklist

Before main runs:

- [ ] answers cannot be guessed from filler;
- [ ] all task IDs unique;
- [ ] target positions produce correct token offsets;
- [ ] no evidence is truncated accidentally;
- [ ] semantic tasks are unambiguous;
- [ ] multi-hop tasks require all intended hops;
- [ ] seeds reproduce identical generated inputs;
- [ ] licenses/provenance documented;
- [ ] small fixture set can run in tests.
