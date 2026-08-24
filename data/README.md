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
