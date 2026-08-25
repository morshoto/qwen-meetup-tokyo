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

The initial deterministic catalog is `tasks/core.v001.jsonl`. It contains one
literal-retrieval, semantic-retrieval, and multi-hop task. Each JSONL record has a
stable ID, schema/version fields, evidence, an `expected.type` scorer declaration,
and `metadata.seed`, `metadata.source`, and `metadata.license` provenance fields.

`fixtures/core.v001.json` ties the catalog to the versioned QA prompt and the
synthetic corpus manifest. The files are intentionally small enough for unit and
integration tests.
