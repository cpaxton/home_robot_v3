# Dynagraph: DynaMem navigation + GraphEQA graph lifecycle

**Dynagraph** is the same runtime stack as [GraphEQA](graph_eqa.md) (DynaMem-style **sparse voxel map** for navigation and exploration, plus **graph-based EQA memory** in `emet.memory.graph_eqa`), with optional **spatial merge** of nearby nodes that share the same primary label and **staleness pruning** of nodes that have not been reinforced recently.

Use it when you want GraphEQA-style prompts and task images, but also want a simple discrete-time lifecycle on graph nodes (similar in spirit to object aging in dense mapping stacks, without replacing the full [open-vocab scene graph](simulation.md) path used by DynaMem instance mode).

## References

- **GraphEQA** (graph memory + EQA): [paper (arXiv:2412.14480)](https://arxiv.org/abs/2412.14480), [project site](https://saumyasaxena.github.io/graph-eqa/). This repo’s re-implementation is described in [graph_eqa.md](graph_eqa.md).
- **DynaMem** (voxel semantic memory + manipulation): [paper (arXiv:2411.04999)](https://arxiv.org/abs/2411.04999), [project site](https://dynamem.github.io/). See [dynamem.md](dynamem.md) and voxel/EQA context in [eqa.md](eqa.md).

## CLI

```bash
emet run dynagraph --robot-ip 127.0.0.1
```

Options mirror `emet run graph-eqa` (robot, Discord, Rerun export, `--no-instance-graph`, `--no-sensor-perception`, etc.). Additional flags:

- **`--merge-xy-m`**: override horizontal merge distance in meters (`dynagraph_merge_xy_m` in config; `0` disables merge).
- **`--staleness-horizon`**: override how many **controller steps** a node can go without a reinforcing observation before `maintain()` drops it (`dynagraph_staleness_horizon`; `0` disables pruning).
- **`--ground-truth`**: **sim only** — build graph nodes from `emet_session["sim_object_placements"]` instead of VLM / YoloE perception. Pair with **`--export`** for a lightweight headless smoke (no full Dynamem/CLIP load). See [Ground-truth graph mode](#ground-truth-graph-mode).
- **`--compare-to-gt`**: **sim only** — on the **full** `--export` path (sensor-built graph after rotate), print alignment vs `sim_object_placements` in session.

If unset on the command line, `run_dynagraph` applies defaults (`dynagraph_merge_xy_m=0.45`, `dynagraph_staleness_horizon=256`) only when those keys are missing from the loaded parameters dict, so you can still set them in `dynav_config.yaml`.

## Configuration keys

| Key | Meaning |
|-----|---------|
| `dynagraph_merge_xy_m` | If `> 0`, a new observation whose **primary** label matches an existing node and whose XY distance is within this threshold **updates** that node (support count, running-mean XYZ, `last_seen`) instead of adding a new node/observation. |
| `dynagraph_staleness_horizon` | If `> 0`, `maintain(current_step)` removes nodes with `current_step - last_seen` greater than this value, removes their observations, renumbers `node_id`, and rebuilds edges. |

The controller passes `frame_step=self.obs_count` into the shared DynaMem→graph hook so `last_seen` stays aligned with the run’s discrete time index.

## Rerun

Live runs log graph nodes and a text tree under **`world/dynagraph/`** (`world/dynagraph/nodes`, `world/dynagraph/summary`). The Dynagraph blueprint adds a dedicated panel for that subtree alongside the usual 3D view and cameras.

### Terminal nav grid (debug)

Set **`EMET_NAVGRID_ASCII=1`** to print a cropped ASCII top-down map to **stderr** after periodic updates (same backend-neutral renderer as Dynamem: `#` obstacles, `.` explored, `@` robot, `0-9a-z` semantic glyphs with legend). Works with any robot backend that uses the shared `SparseVoxelMap` path (Stretch, Galaxea R1, etc.). Output is cropped to the explored region (same bbox as Discord share maps) at up to **320 cells** on the longest edge by default; set **`EMET_NAVGRID_MAX_SIDE=640`** for full Discord resolution.

## Ground-truth graph mode

MuJoCo ZMQ servers (Stretch and Robosuite / rby1) publish **`sim_object_placements`** in [`emet_session`](zmq_session_metadata.md): one entry per object (`cat` → label, `pos` → xyz).

| Scene | GT source |
|-------|-----------|
| Default table (`emet serve mujoco`, any `--robot`) | Packaged `scene_environment.xml` constants (table, blue cube, red cylinder) |
| Robocasa (`--use-robocasa`) | Robocasa wizard `object_placements_info` (category + pose) |
| MolmoSpaces (`--molmospaces-scene …`) | MuJoCo body scan of merged MJCF (skips robot subtree; capped body count) |

### Two workflows

1. **`--ground-truth --export`** (lightweight smoke): graph nodes are created **from GT** directly — verifies session metadata and export wiring without loading Qwen/CLIP. The alignment report is a self-check (graph vs the same placements used to build it).

2. **Normal perception + `--compare-to-gt --export`** (full stack): run the usual Dynagraph controller (rotate, voxel + sensor graph updates), then print **`compare_graph_to_placements_report`** so you can see how **sensor-built** nodes match sim GT. Use this to debug perception and graph hooks against known object poses.

Merge, staleness, and EQA require the **full interactive** path (omit `--export` or use `--export` without `--ground-truth` and exercise the question loop).

Example headless GT smoke (default table, Stretch or rby1):

```bash
# Terminal 1
emet serve mujoco --headless

# Terminal 2
emet run dynagraph --ground-truth --export /tmp/dynagraph_gt --no-rerun --cpu-only -N
```

Or: `uv run python scripts/dynagraph_ground_truth_smoke.py`.

See also [`sim_ground_truth_graph.py`](../src/emet/memory/graph_eqa/sim_ground_truth_graph.py) and [`sim_object_placements.py`](../src/emet/simulation/sim_object_placements.py).

## Code map

| Piece | Role |
|-------|------|
| [`src/emet/controller/controller_dynagraph.py`](../src/emet/controller/controller_dynagraph.py) | `DynagraphController`: `maintain` + Rerun layout after each `update`. |
| [`src/emet/memory/graph_eqa/graph_memory.py`](../src/emet/memory/graph_eqa/graph_memory.py) | `GraphEQAMemory.set_graph_timestep`, merge in `add_observation`, `maintain`. |
| [`src/emet/memory/graph_eqa/dynamem_graph_hooks.py`](../src/emet/memory/graph_eqa/dynamem_graph_hooks.py) | Optional `frame_step` forwarded to `set_graph_timestep`. |
| [`src/emet/app/run_dynagraph.py`](../src/emet/app/run_dynagraph.py) | CLI entry (`emet run dynagraph`). |

## See also

- [GraphEQA](graph_eqa.md) — baseline graph EQA without merge/staleness defaults.
- [Simulation](simulation.md) — MuJoCo / Robocasa and `emet serve mujoco`.
- [CLI](cli.md) — `emet run` apps table.
