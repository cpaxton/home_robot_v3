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

Options mirror `emet run graph-eqa` (robot, Discord, Rerun export, `--no-instance-graph`, `--no-sensor-perception`, etc.). Additional Dynagraph-specific flags:

- **`--merge-xy-m`**: override horizontal merge distance in meters (`dynagraph_merge_xy_m` in config; `0` disables merge).
- **`--staleness-horizon`**: override how many **controller steps** a node can go without a reinforcing observation before `maintain()` drops it (`dynagraph_staleness_horizon`; `0` disables pruning).

If unset on the command line, `run_dynagraph` applies defaults (`dynagraph_merge_xy_m=0.45`, `dynagraph_staleness_horizon=256`) only when those keys are missing from the loaded parameters dict, so you can still set them in the resolved dynav YAML (see **`--dynav-config`**).

### **`--dynav-config`** (per-robot voxel / depth)

Same semantics as **`emet run dynamem`** (see [`run_dynamem.py`](../src/emet/app/run_dynamem.py)): passes through [`resolve_dynav_config_yaml`](../src/emet/robots/__init__.py) so presets like **`dynav_innate_mars.yaml`** apply when you use **`--robot innate_mars`** without you having to name the YAML on the CLI.

### Robocasa (kitchen simulation)

Dynagraph never chooses an MJCF/Robocasa layout by itself—you run the simulator with Robocasa, then connect Dynagraph over ZMQ.

1. **Terminal 1** — MuJoCo + Robocasa + matching `--robot`:
   ```bash
   emet serve mujoco --use-robocasa --robot stretch
   ```
   Substitute **`innate_mars`**, **`rby1`**, etc., to match assets (see [simulation.md](simulation.md)).
2. **Terminal 2** — Dynagraph with the **same robot key** and dynav YAML if needed:
   ```bash
   emet run dynagraph --robot stretch --robot-ip 127.0.0.1
   ```
   Innate Mars in kitchen scenes often wants **`dynav_innate_mars.yaml`** (DA3 preset) when observation depth from ZMQ is missing or thin:
   ```bash
   emet run dynagraph --robot innate_mars --dynav-config dynav_innate_mars.yaml --robot-ip 127.0.0.1
   ```

The server attaches **`navigation_origin_xyt`** in the ZMQ session; Rerun meshes and voxel fusion align when this matches the fused map frame.

#### Multi-robot Robocasa E2E (stretch, innate_mars, galaxea_r1)

Automated three-robot comparison (explored floor vs spawner walkable map, same seed):

```bash
uv run python src/test/app/run_dynagraph_multi_robot_e2e.py
```

**Full guide:** [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md) — how to run, pass criteria, result paths, and example export / detection output for quality review.

All three use **`RobosuiteZmqServer`** on Robocasa (strip-and-replace MJCF + autoplace). **`stretch`** additionally needs **`EMET_STRETCH_ROBOSUITE_ZMQ=1`** so Dynagraph uses **`GenericZmqClient`**. Each **`--export DIR`** writes **`floor_metrics.json`** and **`scene_graph_report.txt`**.

Manual pairwise compare:

```python
from emet.memory.floor_metrics import compare_explored_floor_metrics, load_floor_metrics

a = load_floor_metrics("/tmp/dynagraph_e2e_compare/innate_mars/graph")
b = load_floor_metrics("/tmp/dynagraph_e2e_compare/stretch/graph")
print(compare_explored_floor_metrics(a, b, rtol_area=0.35))
```

#### Pretty text vs MuJoCo “ground truth”

- **Semantic graph**: **`--export DIR`** prints the same **`format_scene_graph_pretty`** summary to stdout as it writes **`scene_graph_report.txt`** inside **`DIR`**.
- **`--print-graph`** appends another pretty snapshot after you exit interactive mode (runs in the **`finally`** handler).
- **MuJoCo body listing (sim sanity check)**:
  **`--dump-sim-ground-truth PATH_ON_SIM_MACHINE`** asks the **`emet serve mujoco`** process to serialize **named bodies**, **world XYZ**, and **approx. yaw about +Z**, then write **`PATH_ON_SIM_MACHINE`** **on the host that runs the simulator** (often the same workstation as Dynagraph).

  Typical one-machine batch:

  ```bash
  # Terminal 1
  emet serve mujoco --use-robocasa --robot stretch

  # Terminal 2 — shared directory so both artefacts land beside each other:
  mkdir -p /tmp/dynagraph_robo
  emet run dynagraph --robot stretch --robot-ip 127.0.0.1 \
    --explore-loop --explore-max-iters 40 --explore-max-failures 5 \
    --export /tmp/dynagraph_robo/graph \
    --dump-sim-ground-truth /tmp/dynagraph_robo/mujoco_bodies.txt
  ```

  Open **`mujoco_bodies.txt`** (or use a **`.json`** path for machine-readable rows) alongside **`scene_graph_report.txt`**. Interpretation is heuristic: VL labels (“granite countertop”) do not trivially grep to MuJoCo body names (“counter_main”), but coarse **spatial clustering** plus **kitchen layout** cues should roughly align once both are mapped into the navigation world frame (**`navigation_origin_xyt`** / voxel frame). Use **`--dump-sim-gt-include-robot`** only when debugging robot-base naming; defaults exclude **`base_link`** subtrees.

### Autonomous frontier exploration (heuristic)

**`--explore-loop`** runs repeated **`run_exploration()`** (same as typing **`explore`** in the interactive REPL—a frontier waypoint and trajectory per iteration). Exploration **stops** when any of:

- **`--explore-max-iters`** is reached,
- **`--explore-max-failures`** consecutive frontier/navigation failures occur (no plan / blocked),
- **`--explore-timeout-s`** wall-clock seconds elapses (optional).

This is **not** a formal “100% geometric coverage guarantee.” Robocasa geometry, dilated frontiers, and planner failures can leave pockets unexplored—the flags are operational stops for scripted runs.

**Batch graph export:**

```bash
emet run dynagraph --robot-ip 127.0.0.1 \
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

## Rerun

Live runs log graph nodes and a text tree under **`world/dynagraph/`** (`world/dynagraph/nodes`, `world/dynagraph/summary`). The Dynagraph blueprint adds a dedicated panel for that subtree alongside the usual 3D view and cameras.

### Manual smoke (Robocasa + export)

- Start server as in **Robocasa** above with **`--use-robocasa`**.
- **`emet run dynagraph --robot-ip 127.0.0.1 --explore-loop --explore-max-iters 20 --explore-max-failures 4 --export /tmp/graphtest`**.
- Confirm **`/tmp/graphtest/scene_graph_report.txt`** exists and is non-empty when the voxel map populated.

## Code map

| Piece | Role |
|-------|------|
| [`src/emet/controller/controller_dynagraph.py`](../src/emet/controller/controller_dynagraph.py) | `DynagraphController`: `maintain` + Rerun layout after each `update`. |
| [`src/emet/memory/graph_eqa/graph_memory.py`](../src/emet/memory/graph_eqa/graph_memory.py) | `GraphEQAMemory.set_graph_timestep`, merge in `add_observation`, `maintain`. |
| [`src/emet/memory/graph_eqa/dynamem_graph_hooks.py`](../src/emet/memory/graph_eqa/dynamem_graph_hooks.py) | Optional `frame_step` forwarded to `set_graph_timestep`. |
| [`src/emet/app/dynagraph_explore.py`](../src/emet/app/dynagraph_explore.py) | `dynagraph_explore_until_terminated` for scripted frontier batches. |
| [`src/emet/app/run_dynagraph.py`](../src/emet/app/run_dynagraph.py) | CLI entry (`emet run dynagraph`). |
| [`src/emet/simulation/mujoco_ground_truth.py`](../src/emet/simulation/mujoco_ground_truth.py) | Text/JSON snapshots of **`mjData.body(*).xpos`** for sim validation; triggered by **`mujoco_ground_truth_dump`** ZMQ recv command. |

## See also

- [Dynagraph Robocasa E2E](dynagraph_robocasa_e2e.md) — multi-robot floor-metrics harness and quality artefacts.
- [GraphEQA](graph_eqa.md) — baseline graph EQA without merge/staleness defaults.
- [Simulation](simulation.md) — MuJoCo / Robocasa and `emet serve mujoco`.
- [CLI](cli.md) — `emet run` apps table.
