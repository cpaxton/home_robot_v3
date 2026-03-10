# Refactor log: 2026-03-08 — Plans moved to docs/plans/, logs updated

## Summary

All plan documents now live under **`docs/plans/`**. The logs index was updated to reference this and to list the new log entry.

## Plan files in docs/plans/

- **ARCHITECTURE_PLAN.md** – Moved from `docs/ARCHITECTURE_PLAN.md`. Multi-robot, multi-simulator refactor (emet rename, robots/simulators).
- **GRAPH_EQA_PLAN.md** – Plan for GraphEQA (graph-based EQA memory). Already moved in a prior change.
- **MAPPING_REFACTOR.md** – Mapping module layout, instance/memory split. Already moved in a prior change.

`docs/plans/README.md` lists all plan documents in one place.

## References updated

- No code or docs referenced `ARCHITECTURE_PLAN.md` by path; only the plan file’s self-reference was updated (to `docs/plans/ARCHITECTURE_PLAN.md`).
- Other docs already pointed at `docs/plans/` for GRAPH_EQA_PLAN and MAPPING_REFACTOR (see graph_eqa.md, mapping README, mapping `__init__.py`).

## docs/logs/ updates

- **README.md** – Clarified that plan documents live in `docs/plans/`. Added this log to the contents list.
- **2026-03-08_plans-and-logs.md** (this file) – Log for the plans move and logs update.
