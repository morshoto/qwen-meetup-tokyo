# Tasks

Shared benchmark and agent task definitions, including expected answers or machine-checkable success conditions where possible.

`catalog.v002.json` is the registry for the current shared task sources. It
declares the catalog version, source files, minimum independent task counts, and
the policy that experiment repeat counts are separate from independent task
counts.

`core.v002.jsonl` contains ten independent presentation-ready tasks in each QA
family: literal retrieval, semantic retrieval, and multi-hop reasoning.
`agent.v001.jsonl` contains ten independent presentation-ready agent
state-tracking tasks. Each record has a stable ID, machine-checkable expected
answer, evidence or critical observation, and provenance metadata including a
seed and license. Additive revisions use a new catalog version rather than
silently changing existing records; the original `core.v001.jsonl` remains
available for reproducing historical runs.
