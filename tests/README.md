# Tests

Tests for reusable code under `src/llm_lab/`. Experiment notebooks and one-off exploratory analysis do not belong here.

Run the complete suite from the repository root with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The suite uses deterministic fake runtimes and does not require model weights.
