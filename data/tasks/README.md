# Tasks

Shared benchmark and agent task definitions, including expected answers or machine-checkable success conditions where possible.

`core.v001.jsonl` is the first reproducible task catalog. The loader requires
unique IDs, a declared scorer under `expected.type`, at least one evidence item,
and provenance metadata including a seed and license. Additive revisions should
use a new catalog version rather than silently changing existing records.
