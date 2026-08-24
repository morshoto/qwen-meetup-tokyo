# Reproducibility checklist

Use this before a pilot/main run and before merging a completed experiment.

## A. Research/design

- [ ] experiment ID is unique;
- [ ] research question is written in the experiment README;
- [ ] hypothesis is falsifiable;
- [ ] independent variables are explicit;
- [ ] controlled variables are explicit;
- [ ] primary metrics are defined before seeing main results;
- [ ] planned comparisons are stated;
- [ ] stopping/resource rules are documented.

## B. Model/runtime

- [ ] model ID is exact;
- [ ] model revision/hash recorded where possible;
- [ ] tokenizer ID/revision recorded;
- [ ] quantization method/format recorded;
- [ ] artifact source or conversion command recorded;
- [ ] runtime name/version recorded;
- [ ] cache/KV precision/options recorded if relevant;
- [ ] reasoning mode/template settings recorded;
- [ ] runtime can execute a short smoke task.

## C. Hardware/environment

- [ ] chip/CPU/GPU description recorded;
- [ ] total memory recorded;
- [ ] OS version recorded;
- [ ] Python environment/package versions reproducible;
- [ ] repository git SHA recorded in run manifest;
- [ ] no personal secrets/machine-specific private identifiers committed;
- [ ] warm/cold start policy documented.

## D. Data/tasks

- [ ] task IDs unique and stable;
- [ ] prompt template versioned;
- [ ] expected answer/success criterion stored;
- [ ] task generation seed stored;
- [ ] filler/corpus provenance known;
- [ ] no obvious answer leakage in filler;
- [ ] semantic tasks reviewed for ambiguity;
- [ ] multi-hop tasks actually require intended hops;
- [ ] repository revisions pinned;
- [ ] agent tool outputs deterministic for controlled studies.

## E. Context construction

- [ ] token counts use inference tokenizer;
- [ ] target context lengths verified;
- [ ] actual input token count logged;
- [ ] requested evidence position logged;
- [ ] actual token offset/position logged;
- [ ] evidence is not accidentally truncated;
- [ ] system/chat/tool overhead accounted for;
- [ ] generated context reproducible from seed/config.

## F. Runner/scoring

- [ ] raw generation stored or durably referenced;
- [ ] scoring is separate from generation;
- [ ] scorer version recorded;
- [ ] normalization rules tested;
- [ ] runtime errors produce records instead of disappearing;
- [ ] OOM/timeouts retained in coverage audit;
- [ ] duplicate trial IDs detected;
- [ ] interrupted runs can resume without corrupting data.

## G. Telemetry

- [ ] TTFT definition consistent;
- [ ] prefill timing available or explicitly unavailable;
- [ ] decode timing available or explicitly unavailable;
- [ ] token counts used in throughput denominator stored;
- [ ] total call time stored;
- [ ] peak memory method documented;
- [ ] model-load time not mixed with warm decode benchmark unintentionally;
- [ ] agent total task time includes tool/orchestration costs where intended.

## H. Smoke run

- [ ] short context works;
- [ ] at least one long context works;
- [ ] beginning/middle/end positions verified manually;
- [ ] correct answer scores correctly;
- [ ] known incorrect answer scores incorrectly;
- [ ] processed summary can be generated;
- [ ] analysis notebook loads output without manual edits.

## I. Main run

- [ ] methodology/config frozen or versioned;
- [ ] sample-size target recorded;
- [ ] missing planned cells explained;
- [ ] batch manifest complete;
- [ ] no result rows manually deleted to “clean” plots;
- [ ] reruns are distinguishable from original trials.

## J. Analysis

- [ ] coverage/missing/error table shown;
- [ ] `n` reported;
- [ ] uncertainty reported for key rates;
- [ ] paired comparisons use matched task IDs;
- [ ] result status separated from correctness;
- [ ] failure examples trace back to trial IDs;
- [ ] figures generated from code/notebook;
- [ ] figure provenance recorded;
- [ ] hypothetical/illustrative values absent from result figures.

## K. Findings

- [ ] measured result added to `docs/findings.md` only after validation;
- [ ] interpretation separated from result;
- [ ] caveats included;
- [ ] null result retained if it answers a hypothesis;
- [ ] broad claims limited to tested conditions.

## L. Experiment completion

- [ ] experiment README updated with final method;
- [ ] resolved config available;
- [ ] result manifest available;
- [ ] processed results available/reproducible;
- [ ] `analysis.ipynb` reruns cleanly;
- [ ] important figures exported;
- [ ] findings linked;
- [ ] linked GitHub issue acceptance criteria met.
