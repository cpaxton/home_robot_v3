# LazyGraph: Dynamem find + Qwen commit on arrival

**LazyGraph** is a memory method sibling to [dynagraph](dynagraph.md): same DynaMem voxel navigation, GraphEQA EQA loop, frontier sync, and dynagraph merge/staleness — but **no streaming YoloE/VLM graph updates** during passive mapping.

Graph object nodes are created **only when navigation successfully arrives** at a target, using **Qwen** label extract on the arrival frame. Detector class names never author graph labels.

## CLI

```bash
uv run emet run lazy-graph --robot-ip 127.0.0.1
uv run emet run agent --memory-backend lazy_graph
```

Options mirror `emet run dynagraph`. Both apps call [`graph_nav_cli.py`](../src/emet/app/graph_nav_cli.py) via `configure_graph_nav` at import (process-lifetime; no monkeypatch). Tests should use `with configure_graph_nav(...)` so the previous controller is restored.

**HM-EQA harness row:** `emet-habitat run-batch --method lazy_graph` selects the
close-look-only row (`harness.habitat_eqa.lazy_graph` in
`configs/benchmarks/dynagraph.yaml`): `use_instance_graph: false`,
`use_sensor_perception: true`, same `unified_eqa` merge/staleness profile as
`dynagraph`. This is the ingest mode that keeps the scene graph at one node per
deliberate inspection — the right regime for location/state questions, at the cost of
denser per-instance count recall.

## vs dynagraph

| | dynagraph | lazy_graph |
|--|-----------|------------|
| Graph during `update()` | YoloE + optional sensor VLM stream | viewpoints only |
| Object labels | detector + VLM | **Qwen at arrival only** |
| Find | voxel + graph | **voxel-first** (committed graph as fallback) |

## Implementation

- Controller: `emet.controller.controller_lazy_graph.LazyGraphController`
- CLI: `emet.app.run_lazy_graph` → `emet.app.graph_nav_cli.configure_graph_nav` (context manager; restores on `with` exit)
- Commit helper: `emet.memory.graph_eqa.ingest.lazy_graph_commit` (shim: `emet.memory.graph_eqa.lazy_graph_commit`)
- Graph memory layout: [graph_memory.md](graph_memory.md)
- Voxel-nav parent: `emet.controller.dynamem.DynamemController`

See `docs/plans/arrival_graph.md` (design history) for exploration ledger and region-exhaustion plans (not yet implemented).
