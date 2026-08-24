# Experiments

Every experiment is a self-contained research unit with a unique sequential ID.

Naming convention:

```text
exp_NNN-short_descriptive_name/
```

Examples:

```text
exp_001-context_measurement/
exp_002-quantization_using_approach_a/
exp_003-context_x_quantization/
exp_004-agent_context_growth/
```

## Experiment contract

Each experiment should contain at least:

```text
exp_NNN-name/
├── README.md          # question, hypotheses, method, status, summary
├── config.yaml        # reproducible parameters
├── analysis.ipynb     # plots, diagnostics, deeper analysis
└── results/
    └── README.md      # result format / provenance notes
```

Add small experiment-specific scripts or fixtures only when they cannot reasonably be generalized. Reusable implementation belongs in `src/llm_lab/`.

## Notebook role

`analysis.ipynb` is not the benchmark runner. It should load recorded outputs and focus on:

- data validation and sanity checks;
- visualizations;
- confidence intervals / repeated-run variation;
- failure-case inspection;
- comparisons across configurations;
- interpretation and candidate findings.

The notebook should be rerunnable from committed or reproducibly generated processed results.
