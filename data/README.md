# Shared data

Common, versioned inputs used by more than one experiment live here.

```text
data/
├── prompts/      # system prompts, task prompts, prompt templates
├── tasks/        # benchmark/task definitions and expected outputs
├── fixtures/     # small deterministic files used by tests/experiments
└── corpora/      # reusable synthetic or curated context corpora
```

Experiment-specific generated inputs should normally remain inside that experiment. Large external datasets should not be committed blindly; document their source and acquisition procedure instead.

The current deterministic task registry is `tasks/catalog.v002.json`. Its QA
source, `tasks/core.v002.jsonl`, contains ten independent tasks for each of the
literal-retrieval, semantic-retrieval, and multi-hop families. Its agent source,
`tasks/agent.v002.jsonl`, contains ten independent state-tracking tasks. Each
JSONL record has a stable ID, schema/version fields, machine-checkable expected
answers, evidence or critical observations, and `metadata.seed`,
`metadata.source`, and `metadata.license` provenance fields. Experiment repeat
counts are recorded separately from these independent task counts.

The original `tasks/core.v001.jsonl` remains available for reproducing
historical runs.

The original `tasks/agent.v001.jsonl` remains available for reproducing the
historical two-task agent pilot.

`fixtures/core.v001.json` ties the catalog to the versioned QA prompt and the
synthetic corpus manifest. The files are intentionally small enough for unit and
integration tests.
