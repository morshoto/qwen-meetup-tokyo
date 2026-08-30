# exp_002 v002 pilot report

**Status: Real-model pilot only.** This report records one measured pilot
condition. It is not a complete quantization comparison and does not support a
Q8/Q6/Q5/Q4 recommendation.

## Provenance

- Manifest: `results/manifest.json`
- Run fingerprint: `231ed9320044933bcdbc293612927f345e0de809ca5bf3a6f64e143a693c8ebe`
- Raw input: `results/raw/trials-v002.jsonl` (Git-ignored)
- Raw trial count: 30
- Raw SHA-256: `84eab3da1656d15df100e3fd7382ca3ab44cfaf83e32f1cd1a123b061a62ade6`
- Task catalog: `data/tasks/core.v002.jsonl`
- Task catalog SHA-256: `7631adc23aab29b0a1b06fae51267a940aa9f86d58dbe830724a3a2bc8703512`
- Scorer: `calibrated.v1`
- Artifact condition: `q8_0` / `Q8_0`
- Context condition: 8,192 input tokens
- Repeats: one per each of the 30 independent tasks
- Source revisions: each raw trial records the current runner, context,
  evaluation, generation, and llama.cpp runtime source revisions; each raw and
  processed row records its generated context SHA-256.

## Measured pilot observations

All 30 trials completed without a runtime, OOM, timeout, or invalid-output
status. Calibrated outcomes were:

| Outcome | Count / attempted |
| --- | ---: |
| Exact calibrated answers | 14 / 30 |
| Answer-bearing outputs | 30 / 30 |
| Format-valid outputs | 19 / 30 |

Median stream-derived measurements were 71.1740 seconds TTFT, 115.7094
prompt-tokens/second proxy, and 7.9671 completion-tokens/second proxy.
Native prefill/decode counters are unavailable through this binding and remain
`null`; these stream-derived values must not be treated as native kernel
metrics.

## Completion boundary

The declared v002 matrix is 4 variants × 2 context lengths × 30 tasks × 5
repeats = 1,200 trials. Only the Q8_0 × 8,192 × 30-task × one-repeat pilot
(30 trials) is measured here. The remaining 1,170 trials are unrun. No
cross-variant capability or systems-cost conclusion is valid until the main
matrix is completed with the same manifest and raw JSONL output.
