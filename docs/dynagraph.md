# Dynagraph: DynaMem navigation + GraphEQA graph lifecycle

**Dynagraph** is the same runtime stack as [GraphEQA](graph_eqa.md) (DynaMem-style **sparse voxel map** for navigation and exploration, plus **graph-based EQA memory** in `emet.memory.graph_eqa`), with optional **spatial merge** of nearby nodes that share the same primary label and **staleness pruning** of nodes that have not been reinforced recently.

Use it when you want GraphEQA-style prompts and task images, but also want a simple discrete-time lifecycle on graph nodes (similar in spirit to object aging in dense mapping stacks, without replacing the full [open-vocab scene graph](simulation.md) path used by DynaMem instance mode).

**CLI:** Run from the project root with **`uv run emet run dynagraph …`** (or activate `.venv` first). See [TESTING.md](TESTING.md#run-from-this-repo) if flags like **`--explore-loop`** are missing from `--help`.

**Rerun (live Dynagraph):** Enabled **by default** (unlike `emet run agent`, which needs **`--rerun`**). Use **`--no-rerun`** to disable. Optional: **`--headless`**, **`--rerun-native`**, **`--rerun-bind`**, **`--rerun-show-panels`**. Verify flags: `uv run python -m emet.app.run_dynagraph --help`.

The main **3D View** uses a **fixed world origin** (`origin=world`; see [rerun.md](rerun.md)) so the voxel map (`world/point_cloud`, `world/obstacles`, `world/explored`), boxes, and dynagraph nodes do not spin when the robot turns. **Not streamed live:** full graph tree text (`print_memory` / old “Dynagraph graph” panel — use `--export` or stdout). **Graph edge lines** and **per-node crop images/mosaic** are also off by default (`rerun.dynagraph` in dynav YAML). Tune load via `rerun.voxel_map_stride`, `rerun.mjcf_mesh_stride`, etc.

## References

- **GraphEQA** (graph memory + EQA): [paper (arXiv:2412.14480)](https://arxiv.org/abs/2412.14480), [project site](https://saumyasaxena.github.io/graph-eqa/). This repo’s re-implementation is described in [graph_eqa.md](graph_eqa.md).
- **DynaMem** (voxel semantic memory + manipulation): [paper (arXiv:2411.04999)](https://arxiv.org/abs/2411.04999), [project site](https://dynamem.github.io/). See [dynamem.md](dynamem.md) and voxel/EQA context in [eqa.md](eqa.md).

## CLI

```bash
uv run emet run dynagraph --robot-ip 127.0.0.1
```

Options mirror `emet run graph-eqa` (robot, Discord, Rerun export, `--no-instance-graph`, `--no-sensor-perception`, etc.). **Rerun is on by default** (`--no-rerun` to disable; `--rerun` is accepted as a no-op alias). Additional Dynagraph-specific flags:

- **`--merge-xy-m`**: override horizontal merge distance in meters (`dynagraph_merge_xy_m` in config; `0` disables merge).
- **`--staleness-horizon`**: override how many **controller steps** a node can go without a reinforcing observation before `maintain()` drops it (`dynagraph_staleness_horizon`; `0` disables pruning).

If unset on the command line, `run_dynagraph` applies defaults (`dynagraph_merge_xy_m=0.45`, `dynagraph_staleness_horizon=256`) only when those keys are missing from the loaded parameters dict, so you can still set them in the resolved dynav YAML (see **`--dynav-config`**).

### **`--dynav-config`** (per-robot voxel / depth)

Same semantics as **`emet run dynamem`** (see [`run_dynamem.py`](../src/emet/app/run_dynamem.py)): passes through [`resolve_dynav_config_yaml`](../src/emet/robots/__init__.py) so presets like **`dynav_innate_mars.yaml`** apply when you use **`--robot innate_mars`** without you having to name the YAML on the CLI.

### Robocasa (kitchen simulation)

Dynagraph never chooses an MJCF/Robocasa layout by itself—you run the simulator with Robocasa, then connect Dynagraph over ZMQ.

1. **Terminal 1** — MuJoCo + Robocasa + matching `--robot`:
   ```bash
   uv run emet serve mujoco --use-robocasa --robot stretch
   ```
   Substitute **`innate_mars`**, **`rby1`**, etc., to match assets (see [simulation.md](simulation.md)).
2. **Terminal 2** — Dynagraph with the **same robot key** and dynav YAML if needed:
   ```bash
   uv run emet run dynagraph --robot stretch --robot-ip 127.0.0.1
   ```
   **Robocasa sim** (ZMQ renders depth): default **`dynav_config.yaml`** or **`--perfect-depth`** (skips DA3 when sensor depth is present). **`dynav_innate_mars.yaml`** now uses **`depth_source: auto`** (sensor in sim, DA3 on real robot when depth is missing):
   ```bash
   # Sim (kitchen) — recommended
   uv run emet run dynagraph --robot innate_mars --robot-ip 127.0.0.1 --perfect-depth

   # Real robot (auto → DA3 when no ZMQ depth)
   uv run emet run dynagraph --robot innate_mars --dynav-config dynav_innate_mars.yaml --robot-ip <ROBOT_IP>
   ```

The server attaches **`navigation_origin_xyt`** in the ZMQ session; Rerun meshes and voxel fusion align when this matches the fused map frame.

#### Multi-robot Robocasa E2E (innate_mars, galaxea_r1)

Automated two-robot comparison (explored floor vs spawner walkable map, same seed):

```bash
uv run python src/test/app/run_dynagraph_multi_robot_e2e.py
```

**Full guide:** [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md) — how to run, pass criteria, result paths, and example export / detection output for quality review.

**innate_mars** and **galaxea_r1** use **`RobosuiteZmqServer`** on Robocasa (strip-and-replace MJCF + autoplace) with **`GenericZmqClient`**. Each **`--export DIR`** writes **`floor_metrics.json`** and **`scene_graph_report.txt`**.

**Stretch + Robocasa via Robosuite** (unified server with galaxea) is experimental on branch **`feature/stretch-robocasa-robosuite`**; on this branch Stretch Robocasa still uses **`stretch_mujoco`** + **`StretchZmqClient`** (same as `main`).

Manual pairwise compare:

```python
from emet.memory.floor_metrics import compare_explored_floor_metrics, load_floor_metrics

a = load_floor_metrics("/tmp/dynagraph_e2e_compare/innate_mars/graph")
b = load_floor_metrics("/tmp/dynagraph_e2e_compare/galaxea_r1/graph")
print(compare_explored_floor_metrics(a, b, rtol_area=0.35))
```

#### Pretty text vs MuJoCo “ground truth”

- **Semantic graph**: **`--export DIR`** prints the same **`format_scene_graph_pretty`** summary to stdout as it writes **`scene_graph_report.txt`** inside **`DIR`**.
- **`--print-graph`** appends another pretty snapshot after you exit interactive mode (runs in the **`finally`** handler).
- **MuJoCo body listing (sim sanity check)**:
  **`--dump-sim-ground-truth PATH_ON_SIM_MACHINE`** asks the **`emet serve mujoco`** process to serialize **named bodies**, **world XYZ**, and **approx. yaw about +Z**, then write **`PATH_ON_SIM_MACHINE`** **on the host that runs the simulator** (often the same workstation as Dynagraph).

  Typical one-machine batch:

  ```bash
  # Terminal 1 (innate_mars example; galaxea_r1 also uses RobosuiteZmqServer)
  uv run emet serve mujoco --use-robocasa --robot innate_mars --headless --seed 0

  # Terminal 2 — shared directory so both artefacts land beside each other:
  mkdir -p /tmp/dynagraph_robo
  uv run emet run dynagraph --robot innate_mars --robot-ip 127.0.0.1 \
    --explore-loop --explore-max-iters 40 --explore-max-failures 5 \
    --export /tmp/dynagraph_robo/graph \
    --dump-sim-ground-truth /tmp/dynagraph_robo/mujoco_bodies.txt
  ```

  Open **`mujoco_bodies.txt`** (or use a **`.json`** path for machine-readable rows) alongside **`scene_graph_report.txt`**. Interpretation is heuristic: VL labels (“granite countertop”) do not trivially grep to MuJoCo body names (“counter_main”), but coarse **spatial clustering** plus **kitchen layout** cues should roughly align once both are mapped into the navigation world frame (**`navigation_origin_xyt`** / voxel frame). Use **`--dump-sim-gt-include-robot`** only when debugging robot-base naming; defaults exclude **`base_link`** subtrees.

#### Object GT export and GraphObjectFusion calibration

For **tunable instance→graph fusion** (spatial + 3D bounds + optional SigLIP embeddings), use a fixed Robocasa scene:

1. **Export sim GT once** (offline kitchen load; 3D bounds + projected head **`bbox_xyxy_head`**):

   ```bash
   uv run emet export-sim-gt --robot innate_mars --seed 0 --layout 1 \
     -o /tmp/graph_fusion_calib/gt_seed0.json
   ```

2. **Capture raw detections** during one explore run (no re-sim per grid point):

   ```bash
   uv run emet run dynagraph --robot innate_mars --explore-loop --explore-max-iters 15 \
     --no-rerun --cpu-only \
     --calibration-export /tmp/graph_fusion_calib/frames.jsonl
   ```

3. **Tune defaults offline**:

   ```bash
   uv run emet tune-graph-fusion \
     --gt /tmp/graph_fusion_calib/gt_seed0.json \
     --frames /tmp/graph_fusion_calib/frames.jsonl \
     --write-config src/emet/config/agents/default_graph_object_fusion.yaml
   ```

Dynagraph loads [`default_graph_object_fusion.yaml`](../src/emet/config/agents/default_graph_object_fusion.yaml) into parameters when unset; enable in agent YAML under **`embodied_agent.graph_eqa_memory.graph_object_fusion`**. When fusion is on, legacy **`dynagraph_merge_xy_m`** on the instance path is disabled (VLM nodes unchanged). Implementation: [`graph_object_fusion/`](../src/emet/memory/graph_eqa/graph_object_fusion/), GT builder [`mujoco_gt_objects.py`](../src/emet/simulation/mujoco_gt_objects.py).

#### Rerun load (crops / seen_from lines)

Live Rerun **does not** stream per-node crop images, the RGB mosaic, or **graph edge line strips** (`near`, `on`, `seen_from`, etc.) by default—they were overloading the viewer. The graph still stores all edges and observations in memory.

On **`--export DIR`**, the exporter writes:

| Path | Contents |
|------|----------|
| `DIR/graph.json` | Nodes + all edge relations |
| `DIR/dynagraph/crops/*.png` | YoloE/instance bbox crops per object node |
| `DIR/dynagraph/crops_mosaic.png` | Labeled grid of crops |
| `DIR/dynagraph/seen_from.json` | Viewpoint → object links with world XYZ |
| `DIR/dynagraph/gallery.md` | Node table with links to crop files |

The default 3D view is anchored at ``world/robot`` but includes all ``world/**`` layers (voxel point cloud, 2D obstacle/explored maps, object boxes, dynagraph nodes). Only **crop images**, **edge line strips**, and the **crop mosaic** are off by default (viewer stability).

Opt back into those heavy channels in agent/dynav YAML:

```yaml
rerun:
  dynagraph:
    log_crops: true
    log_edges: true
```

Or set `EMET_DYNAGRAPH_RERUN_CROPS=1` / `EMET_DYNAGRAPH_RERUN_EDGES=1` (env overrides YAML when set). Defaults and other viewer keys: `src/emet/config/agents/default_rerun.yaml`.

### Autonomous frontier exploration (heuristic)

**`--explore-loop`** runs repeated **`run_exploration()`** (same as typing **`explore`** in the interactive REPL—a frontier waypoint and trajectory per iteration). Exploration **stops** when any of:

- **`--explore-max-iters`** is reached,
- **`--explore-max-failures`** consecutive frontier/navigation failures occur (no plan / blocked),
- **`--explore-timeout-s`** wall-clock seconds elapses (optional).

This is **not** a formal “100% geometric coverage guarantee.” Robocasa geometry, dilated frontiers, and planner failures can leave pockets unexplored—the flags are operational stops for scripted runs.

**Batch graph export:**

```bash
uv run emet run dynagraph --robot-ip 127.0.0.1 \
  --explore-loop --explore-max-iters 80 --explore-max-failures 5 \
  --export /tmp/dynagraph_out
```

Writes the GraphEQA memory backend layout plus **`scene_graph_report.txt`** and prints the same pretty summary to stdout (via `export_graph_eqa_dir`). Combine with **`--question "Where is …?"`** to answer one NL query after exploration (still exports the graph state afterward).

### **`--print-graph`**

Append a pretty-print snapshot of **`GraphEQAMemory`** at session end (**`finally`**), e.g. when you quit the interactive loop with an empty line. Does not imply **`--export`** unless you pass both.

## Configuration keys

| Key | Meaning |
|-----|---------|
| `dynagraph_merge_xy_m` | If `> 0`, a new observation whose **primary** label matches an existing node and whose XY distance is within this threshold **updates** that node (support count, running-mean XYZ, `last_seen`) instead of adding a new node/observation. |
| `dynagraph_staleness_horizon` | If `> 0`, `maintain(current_step)` removes nodes with `current_step - last_seen` greater than this value, removes their observations, renumbers `node_id`, and rebuilds edges. |

The controller passes `frame_step=self.obs_count` into the shared DynaMem→graph hook so `last_seen` stays aligned with the run’s discrete time index.

## Human-readable EQA answers

`--question "Where is the sink?"` prints a **short spatial sentence** (object + approximate XYZ), not “image 1”. Formatting lives in [`human_answer.py`](../src/emet/memory/graph_eqa/human_answer.py) and applies to Dynagraph, graph-eqa, and **`run_agent`** (`query_scene_graph` / `query_memory`). See [graph_eqa.md](graph_eqa.md#answer-format-human-readable).

## Rerun

Live runs log graph nodes and a text tree under **`world/dynagraph/`** (`world/dynagraph/nodes`, `world/dynagraph/summary`). The Dynagraph blueprint adds a dedicated panel for that subtree alongside the usual 3D view and cameras.

### Terminal nav grid (debug)

Set **`EMET_NAVGRID_ASCII=1`** to print a cropped ASCII top-down map to **stderr** after periodic updates (same backend-neutral renderer as Dynamem: `#` obstacles, `.` explored, `@` robot, `0-9a-z` semantic glyphs with legend). Works with any robot backend that uses the shared `SparseVoxelMap` path (Stretch, Galaxea R1, etc.). Output is cropped to the explored region (same bbox as Discord share maps) at up to **320 cells** on the longest edge by default; set **`EMET_NAVGRID_MAX_SIDE=640`** for full Discord resolution.

### Manual smoke (Robocasa + export)

- Start server as in **Robocasa** above with **`--use-robocasa`**.
- **`uv run emet run dynagraph --robot-ip 127.0.0.1 --explore-loop --explore-max-iters 20 --explore-max-failures 4 --export /tmp/graphtest`**.
- Confirm **`/tmp/graphtest/scene_graph_report.txt`** exists and is non-empty when the voxel map populated.

## Testing

| Layer | Command |
|-------|---------|
| Unit (explore loop, graph memory) | `uv run emet test src/test/app/test_dynagraph_explore.py src/test/memory/test_graph_eqa_memory.py -v` |
| Multi-robot Robocasa floor E2E | `uv run python src/test/app/run_dynagraph_multi_robot_e2e.py` |
| Manual EQA + export | [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#assessing-semantic--eqa-quality) |

Full index and known gaps (graph + EQA on known scene): [TESTING.md](TESTING.md).

## Code map

| Piece | Role |
|-------|------|
| [`src/emet/controller/controller_dynagraph.py`](../src/emet/controller/controller_dynagraph.py) | `DynagraphController`: `maintain` + Rerun layout after each `update`. |
| [`src/emet/memory/graph_eqa/graph_memory.py`](../src/emet/memory/graph_eqa/graph_memory.py) | `GraphEQAMemory.set_graph_timestep`, merge in `add_observation`, `maintain`. |
| [`src/emet/memory/graph_eqa/dynamem_graph_hooks.py`](../src/emet/memory/graph_eqa/dynamem_graph_hooks.py) | Optional `frame_step` forwarded to `set_graph_timestep`. |
| [`src/emet/app/dynagraph_explore.py`](../src/emet/app/dynagraph_explore.py) | `dynagraph_explore_until_terminated` for scripted frontier batches. |
| [`src/emet/app/run_dynagraph.py`](../src/emet/app/run_dynagraph.py) | CLI entry (`emet run dynagraph`). |
| [`src/emet/simulation/mujoco_ground_truth.py`](../src/emet/simulation/mujoco_ground_truth.py) | Text/JSON snapshots of **`mjData.body(*).xpos`** for sim validation; triggered by **`mujoco_ground_truth_dump`** ZMQ recv command. |
| [`src/emet/simulation/mujoco_gt_objects.py`](../src/emet/simulation/mujoco_gt_objects.py) | Per-object **3D AABB** + optional head **2D bbox** JSON (`emet export-sim-gt`). |
| [`src/emet/memory/graph_eqa/graph_object_fusion/`](../src/emet/memory/graph_eqa/graph_object_fusion/) | **GraphObjectFusion** + offline **`emet tune-graph-fusion`**. |

## See also

- [Testing index](TESTING.md) — master map of test docs, suites, and known gaps (graph + EQA on known scene).
- [Dynagraph Robocasa E2E](dynagraph_robocasa_e2e.md) — multi-robot floor-metrics harness and quality artefacts.
- [GraphEQA](graph_eqa.md) — baseline graph EQA without merge/staleness defaults.
- [Simulation](simulation.md) — MuJoCo / Robocasa and `emet serve mujoco`.
- [CLI](cli.md) — `emet run` apps table.
