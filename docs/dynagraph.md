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
- **`--ground-truth`**: **sim only** — build graph nodes from `emet_session["sim_object_placements"]` instead of VLM / YoloE perception. Pair with **`--export`** for a **full episode** export (rotate, voxel frames, graph, GT sidecars). See [Ground-truth graph mode](#ground-truth-graph-mode).
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

## Ground-truth graph mode (`--ground-truth`)

Use **`--ground-truth`** in simulation to build the Dynagraph scene graph from **`emet_session["sim_object_placements"]`** instead of VLM perception labels. **Voxel mapping, rotate-in-place, explore, and YoloE instance detection still run**; detections are matched to nearest GT nodes in XY and attached as observation RGB (description suffix ``|det:…``). Each control step also appends a **navigation viewpoint sample** (camera pose + RGB, no new entity node) so the graph memory records everywhere the robot observed from. Use **`--compare-to-gt`** when you want a full VLM perception graph overlaid on sim reference.

MuJoCo ZMQ servers publish placements in [`emet_session`](zmq_session_metadata.md). Each entry has a **`cat`** label, world **`pos`**, and (when the server scanned the MJCF) axis-aligned **`bounds`** from mesh/collision geoms.

| Scene | GT source |
|-------|-----------|
| Default table | Packaged `scene_environment.xml` constants, overlaid with live MuJoCo body poses when the server has the model |
| Robocasa (`--scene robocasa`) | **Full kitchen fixture scan** (sink, counter, cabinets, appliances, …) **merged** with wizard manipulable objects |
| MolmoSpaces | Per-body MJCF scan (robot subtree skipped; capped on large scenes) |

### `--ground-truth` vs `--compare-to-gt`

| Flag | Graph source | Rerun |
|------|--------------|-------|
| **`--ground-truth`** | All nodes from sim GT | **«Graph (ground truth)»** column (nodes + 3D boxes); voxel map + instance→GT association + per-step viewpoint samples |
| **`--compare-to-gt`** | Normal sensor / VLM graph | **«Dynagraph 3D»** (perception) + **«Sim GT (reference)»** (green overlay) |

The two flags are **mutually exclusive**.

### Workflows

**Export smoke** (full controller + GT sidecars; CI check):

```bash
uv run python scripts/dynagraph_ground_truth_smoke.py
uv run python scripts/dynagraph_ground_truth_smoke.py --scene ithor   # MolmoSpaces (needs wrapper)
```

**Full GT episode export** (rotate + voxel frames + `sim_object_placements.json`):

```bash
emet serve mujoco --headless                    # or --scene robocasa / --scene ithor
emet run dynagraph --ground-truth --export /tmp/dynagraph_gt --no-rerun --cpu-only
```

Exported layout: `manifest.json`, `graph.json`, `frames/`, `sim_object_placements.json`, optional `gt_alignment_report.txt`, per-frame `gt_assoc_NNNN.json` when instance masks overlap projected GT bounds.

**Batch metrics** (completeness, localization error, association recall):

```bash
uv run python scripts/eval_dynagraph_ground_truth.py --episode /tmp/dynagraph_gt
uv run python scripts/eval_dynagraph_ground_truth.py --run-live --cpu-only --output /tmp/metrics.json
```

**Interactive GT graph** (EQA / explore on known sim labels):

```bash
emet serve mujoco --scene robocasa --headless --port-offset 50
emet run dynagraph --ground-truth --port-offset 50
```

**MolmoSpaces GT** (iTHOR + per-body MJCF scan):

```bash
emet serve mujoco --scene ithor --headless --port-offset 50
emet run dynagraph --ground-truth --export /tmp/molmo_gt --port-offset 50 --no-rerun --cpu-only
```

Graph nodes and 3D bounds appear in Rerun after startup; **rotate-in-place runs by default** to seed the voxel map (use **`-N`** to skip). Use **explore** / **e** to extend the map; instance detections attach to nearby GT nodes as you move.

**Perception vs GT** (full Dynagraph stack + alignment report):

```bash
emet run dynagraph --compare-to-gt --export /tmp/dg_cmp --port-offset 50 -N
```

### Limitations

- **Static at server start:** placements are not updated if objects move during the episode.
- **Stretch sim:** GT scan uses `robot_sim.model` (MuJoCo subprocess); restart the server after code changes.
- **Coordinate frame:** `pos` / `bounds` are **MuJoCo world XYZ** (same as `camera_pose`). `gps`/`compass` are episode-relative; servers publish **`navigation_origin_xyt`** so Rerun can place the robot mesh in world.
- **Fixture grouping (Robocasa):** cabinet doors and panels merge into one entry per fixture group (e.g. `cab_1`); walls/floors are excluded.
- **Wrong GT on custom MJCF path:** if `environment.kind` stays `default_table`, you may get default-table constants — use `--scene robocasa`, `--scene ithor`, or merge via MolmoSpaces so session metadata is set.

See [`sim_object_placements.py`](../src/emet/simulation/sim_object_placements.py) and [`sim_ground_truth_graph.py`](../src/emet/memory/graph_eqa/sim_ground_truth_graph.py).

## Code map

| Piece | Role |
|-------|------|
| [`src/emet/controller/controller_dynagraph.py`](../src/emet/controller/controller_dynagraph.py) | `DynagraphController`: `maintain` + Rerun layout after each `update`. |
| [`src/emet/memory/graph_eqa/graph_memory.py`](../src/emet/memory/graph_eqa/graph_memory.py) | `GraphEQAMemory.set_graph_timestep`, merge in `add_observation`, `maintain`. |
| [`src/emet/memory/graph_eqa/dynamem_graph_hooks.py`](../src/emet/memory/graph_eqa/dynamem_graph_hooks.py) | Optional `frame_step` forwarded to `set_graph_timestep`. |
| [`src/emet/app/run_dynagraph.py`](../src/emet/app/run_dynagraph.py) | CLI entry (`emet run dynagraph`). |
| [`src/emet/app/run_interactive.py`](../src/emet/app/run_interactive.py) | Shared interactive REPL for graph-EQA and task-mode apps. |

## See also

- [GraphEQA](graph_eqa.md) — baseline graph EQA without merge/staleness defaults.
- [Simulation](simulation.md) — MuJoCo / Robocasa and `emet serve mujoco`.
- [CLI](cli.md) — `emet run` apps table.
