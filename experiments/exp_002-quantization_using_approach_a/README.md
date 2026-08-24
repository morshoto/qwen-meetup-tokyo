# exp_002-quantization_using_approach_a

## Goal

Measure the capability/performance trade-off of one clearly specified quantization approach while keeping benchmark tasks and runtime controls reproducible.

`approach_a` is intentionally a placeholder until the exact quantizer/runtime is selected; the experiment should be renamed once that decision is made.

## Initial questions

- How do memory, prefill throughput, decode throughput, and task accuracy change by precision?
- Which capabilities degrade first?
- Is degradation different at short versus long context?

## Status

Quantization approach and exact model variants are still to be selected.

## Analysis

Use `analysis.ipynb` for Pareto plots, per-task degradation, memory/performance analysis, and failure-case inspection.
