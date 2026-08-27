# LazyGraph: Dynamem find + Qwen commit on arrival

**LazyGraph** is a memory method sibling to [dynagraph](dynagraph.md): same DynaMem voxel navigation, GraphEQA EQA loop, frontier sync, and dynagraph merge/staleness — but **no streaming YoloE/VLM graph updates** during passive mapping.

Graph object nodes are created **only when navigation successfully arrives** at a target, using **Qwen** label extract on the arrival frame. Detector class names never author graph labels.

## CLI

```bash
uv run emet run lazy-graph --robot-ip 127.0.0.1
uv run emet run agent --memory-backend lazy_graph
```

Options mirror `emet run dynagraph` (the lazy-graph app reuses the dynagraph CLI).

## vs dynagraph

| | dynagraph | lazy_graph |
|--|-----------|------------|
| Graph during `update()` | YoloE + optional sensor VLM stream | viewpoints only |
| Object labels | detector + VLM | **Qwen at arrival only** |
| Find | voxel + graph | **voxel-first** (committed graph as fallback) |

## Implementation

- Controller: `emet.controller.controller_lazy_graph.LazyGraphController`
- Commit helper: `emet.memory.graph_eqa.lazy_graph_commit`
- Graph memory layout: [graph_memory.md](graph_memory.md)
- Branch: `feature/lazy-graph`

See `docs/plans/arrival_graph.md` (design history) for exploration ledger and region-exhaustion plans (not yet implemented).
