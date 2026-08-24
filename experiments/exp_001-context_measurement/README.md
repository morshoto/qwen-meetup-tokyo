# exp_001-context_measurement

## Goal

Measure the effective context behavior of Qwen3.8-27B across increasing context lengths and evidence positions before introducing quantization as an additional variable.

## Initial questions

- How does task accuracy change with context length?
- Is there measurable position bias / lost-in-the-middle behavior?
- Does the effective context window differ between literal retrieval, semantic retrieval, and multi-hop reasoning?

## Status

Designing methodology and benchmark inputs.

## Analysis

Use `analysis.ipynb` for plots, failure inspection, and deeper analysis after runs are recorded.
