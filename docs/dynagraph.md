# Dynagraph: DynaMem navigation + GraphEQA graph lifecycle

**Dynagraph** is the same runtime stack as [GraphEQA](graph_eqa.md) (DynaMem-style **sparse voxel map** for navigation and exploration, plus **graph-based EQA memory** in `emet.memory.graph_eqa`), with optional **spatial merge** of nearby nodes that share the same primary label and **staleness pruning** of nodes that have not been reinforced recently.

Use it when you want GraphEQA-style prompts and task images, but also want a simple discrete-time lifecycle on graph nodes (similar in spirit to object aging in dense mapping stacks, without replacing the full [open-vocab scene graph](simulation.md) path used by DynaMem instance mode).

**CLI:** Run from the project root with **`uv run emet run dynagraph …`** (or activate `.venv` first). See [TESTING.md](TESTING.md#run-from-this-repo) if flags like **`--explore-loop`** are missing from `--help`.

**Interactive agent:** `emet run agent` defaults to **`--memory-backend dynagraph`** (same controller stack + interactive merge/staleness). See [AGENT_RUN.md](AGENT_RUN.md).

**Rerun (live Dynagraph):** Enabled **by default** (unlike `emet run agent`, which needs **`--rerun`**). Use **`--no-rerun`** to disable. Optional: **`--headless`**, **`--rerun-native`**, **`--rerun-bind`**, **`--rerun-show-panels`**. Verify flags: `uv run python -m emet.app.run_dynagraph --help`.

The main **3D View** uses a **fixed world origin** (`origin=world`; see [rerun.md](rerun.md)) so the voxel map (`world/point_cloud`, `world/obstacles`, `world/explored`), boxes, and dynagraph nodes do not spin when the robot turns. **Not streamed live:** full graph tree text (`print_memory` / old “Dynagraph graph” panel — use `--export` or stdout). **Graph edge lines** and **per-node crop images/mosaic** are also off by default (`rerun.dynagraph` in dynav YAML). Tune load via `rerun.voxel_map_stride`, `rerun.mjcf_mesh_stride`, etc.

## References

- **GraphEQA** (graph memory + EQA): [paper (arXiv:2412.14480)](https://arxiv.org/abs/2412.14480), [project site](https://saumyasaxena.github.io/graph-eqa/). This repo’s re-implementation is described in [graph_eqa.md](graph_eqa.md).
- **DynaMem** (voxel semantic memory + manipulation): [paper (arXiv:2411.04999)](https://arxiv.org/abs/2411.04999), [project site](https://dynamem.github.io/). See [dynamem.md](dynamem.md) and voxel/EQA context in [eqa.md](eqa.md).

## CLI

```bash
uv run emet run dynagraph --robot-ip 127.0.0.1
# --robot optional when sim publishes emet_robot_id on ZMQ
```

Unified config: **`--config`** (default [`configs/emet/default.yaml`](../configs/emet/default.yaml)); overrides **`--set mapping.depth_source=auto`**. Legacy **`--dynav-config`** is deprecated. See [Unified EMET configuration](emet_config.md).

Options mirror `emet run graph-eqa` (robot, Discord, Rerun export, `--no-instance-graph`, `--no-sensor-perception`, etc.). **Rerun is on by default** (`--no-rerun` to disable; `--rerun` is accepted as a no-op alias). Additional Dynagraph-specific flags:

- **`--merge-xy-m`**: override horizontal merge distance in meters (`dynagraph_merge_xy_m` in config; `0` disables merge).
- **`--staleness-horizon`**: override how many **controller steps** a node can go without a reinforcing observation before `maintain()` drops it (`dynagraph_staleness_horizon`; `0` disables pruning).
- **`--export-voxel-pickle`**: with **`--export`** / **`--dump-memory`**, additionally write **`voxel_map.pkl`** (full `SparseVoxelMapDynamem` state) into the export dir so the checkpoint restores obstacles / explored area, not just the graph.
- **`--input-path DIR`**: resume from a previous export: restores graph nodes **with staleness state** (`last_seen`, `support_count`, extents), the controller step counter (`final_step` from `manifest.json`), and — when `DIR/voxel_map.pkl` exists — the voxel map. Used per-cycle by the lifelong dynamic exploration phase ([dynamic_exploration_benchmark.md](dynamic_exploration_benchmark.md)).
- **`--ground-truth`**: **sim only** — build graph nodes from `emet_session["sim_object_placements"]` instead of VLM / YoloE perception. Pair with **`--export`** for a **full episode** export (rotate, voxel frames, graph, GT sidecars). See [Ground-truth graph mode](#ground-truth-graph-mode).
- **`--compare-to-gt`**: **sim only** — on the **full** `--export` path (sensor-built graph after rotate), print alignment vs `sim_object_placements` in session.

If unset on the command line, `run_dynagraph` applies defaults (`dynagraph_merge_xy_m=0.45`, `dynagraph_staleness_horizon=256`) only when those keys are missing from the loaded parameters dict (also the defaults in `dynav_config.yaml` and the `interactive` profile in `configs/benchmarks/dynagraph.yaml`). Paper benchmarks use other profiles from that file via `emet.eval.benchmark_dynagraph` — see [paper_benchmarks.md](paper_benchmarks.md).

### Config and robot resolution

**`--config`** loads nested YAML (`mapping`, `agent`, `robots.*`). When **`--robot`** is omitted, the client resolves robot id from config → connection profile → **ZMQ** → `stretch`. Innate Mars depth (`depth_source: auto`, DA3 fallback) comes from **`robots.innate_mars`** in the default config — no separate YAML required on the CLI.

### Robocasa (kitchen simulation)

Dynagraph never chooses an MJCF/Robocasa layout by itself—you run the simulator with Robocasa, then connect Dynagraph over ZMQ.

1. **Terminal 1** — MuJoCo + Robocasa + matching `--robot`:
   ```bash
   uv run emet serve mujoco --use-robocasa --robot stretch
   ```
   Substitute **`innate_mars`**, **`rby1`**, etc., to match assets (see [simulation.md](simulation.md)).
2. **Terminal 2** — Dynagraph (robot optional if sim is already running with matching ZMQ id):
   ```bash
   uv run emet run dynagraph --robot-ip 127.0.0.1
   ```
   **Robocasa sim** (ZMQ renders depth): default config uses **`depth_source: sensor`**. Innate Mars uses **`robots.innate_mars.mapping.depth_source: auto`** (sensor in sim, DA3 on hardware without depth). Optional **`--perfect-depth`** skips DA3 when sensor depth is present:
   ```bash
   uv run emet run dynagraph --robot innate_mars --robot-ip 127.0.0.1 --perfect-depth
   # Real robot (same default config; connection profile or --robot innate_mars)
   uv run emet run dynagraph --connection herman
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

Dynagraph loads [`default_graph_object_fusion.yaml`](../src/emet/config/agents/default_graph_object_fusion.yaml) into parameters when unset; enable in agent YAML under **`embodied_agent.graph_eqa_memory.graph_object_fusion`**. When fusion is on, legacy **`dynagraph_merge_xy_m`** on `add_observation` is disabled, but **`fallback_spatial_merge_xy_m`** defaults to the same value for a second merge tier (see Configuration keys). Implementation: [`graph_object_fusion/`](../src/emet/memory/graph_eqa/graph_object_fusion/), GT builder [`mujoco_gt_objects.py`](../src/emet/simulation/mujoco_gt_objects.py).

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

The default 3D view uses ``origin=world`` with ``contents=world/**`` (see [rerun.md](rerun.md)) so map layers stay fixed while ``world/robot`` moves. Do **not** default to ``origin=world/robot`` or the map co-rotates on in-place turns. Only **crop images**, **edge line strips**, and the **crop mosaic** are off by default (viewer stability).

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

### Graph frontier nodes (EQA-guided)

When `graph_eqa_frontier_nodes.enabled` is true (default in `dynav_config.yaml`), Dynagraph / GraphEQA:

1. **Clusters** unexplored voxel frontiers into graph nodes (`is_frontier=True`) after each `update()`.
2. **Tags** them in the EQA prompt (`IMAGE_DESCRIPTIONS`) so the VLM can pick a frontier image to explore.
3. **Biases** `sample_exploration` / `sample_frontier` toward clusters whose nearby object labels overlap the active question keywords (`keyword_score_weight`, default `2.0`).
4. **Routes** low-confidence EQA iterations to the best-matching frontier graph node before voxel sampling.

During HM-EQA / `run_eqa`, frontier nodes are re-synced **before each VLM call** and **after exploration navigation** so targets stay aligned with the growing map.

**VLM selection (Habitat bake-off):** default EQA checkpoint is **Qwen3-VL-8B int4** in
[`dynav_config.yaml`](../src/emet/config/dynav_config.yaml) (`eqa.vl_family`, `eqa.vl_hf_model_id`).
A 2026-06 canonical-6 comparison found it (5/6) outperformed both Qwen2.5-VL-3B (2/6) and
Qwen3.5-9B (3/6) on embodied MCQ EQA; see
[docs/habitat/vlm_bakeoff.md](habitat/vlm_bakeoff.md) and the paper appendix
`paper/sections/appendix/06_model_choice.tex`.

```yaml
graph_eqa_frontier_nodes:
  enabled: true
  max_nodes: 12
  min_cluster_cells: 3
  keyword_score_weight: 2.0
```

Sim smoke: `uv run python src/test/app/run_dynagraph_nav_benchmark.py --default` (GT nav + 3× `run_exploration`). Habitat sweep: `scripts/run_habitat_frontier_experiments.sh`.

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
| `graph_object_fusion.fallback_spatial_merge_xy_m` | When GraphObjectFusion is enabled, strict merge gates (XY, 3D centroid, bounds IoU, embedding) run first. If no node matches, a **fallback tier** merges into the nearest object node within this XY radius (ignores bounds/embedding). Defaults to **`0.45`** in [`default_graph_object_fusion.yaml`](../src/emet/config/agents/default_graph_object_fusion.yaml); when unset at attach time, [`setup.py`](../src/emet/memory/graph_eqa/graph_object_fusion/setup.py) copies **`dynagraph_merge_xy_m`** from the loaded dynav parameters. Set to **`0`** to disable fallback. Innate Mars hardware uses wider gates in [`graph_object_fusion_innate_mars.yaml`](../src/emet/config/agents/graph_object_fusion_innate_mars.yaml) (wired via [`dynav_innate_mars.yaml`](../src/emet/config/dynav_innate_mars.yaml)). |
| `dynagraph_staleness_horizon` | If `> 0`, `maintain(current_step)` removes nodes with `current_step - last_seen` greater than this value, removes their observations, renumbers `node_id`, and rebuilds edges. |
| `graph_eqa_frontier_nodes.enabled` | Sync unexplored frontier clusters into the graph for EQA prompts and question-guided exploration. |
| `graph_eqa_frontier_nodes.max_nodes` | Cap on simultaneous frontier graph nodes. |
| `graph_eqa_frontier_nodes.min_cluster_cells` | Minimum grid cells per frontier cluster. |
| `graph_eqa_frontier_nodes.keyword_score_weight` | Blend weight for question-keyword overlap in voxel `sample_exploration`. |

The controller passes `frame_step=self.obs_count` into the shared DynaMem→graph hook so `last_seen` stays aligned with the run’s discrete time index.

**Stationary hardware stream:** `emet stream --backend dynagraph` on a non-moving real robot can still add graph nodes every step (DA3 depth noise + GraphObjectFusion gates). See [known_issues.md](known_issues.md#dynagraph-graph-node-explosion-on-stationary-hardware-stream).

## Human-readable EQA answers

`--question "Where is the sink?"` prints a **short spatial sentence** (object + approximate XYZ), not “image 1”. Formatting lives in [`human_answer.py`](../src/emet/memory/graph_eqa/human_answer.py) and applies to Dynagraph, graph-eqa, and **`run_agent`** (`query_scene_graph` / `query_memory`). See [graph_eqa.md](graph_eqa.md#answer-format-human-readable).

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

### Tune graph object fusion (offline)

Use sim GT **3D bounds** and head **2D bboxes** from `emet export-sim-gt`, then record live detections during a short Dynagraph run, and grid-search fusion thresholds.

**Two recall numbers:** calibration scoring is **geometry-first**. `spatial_recall` counts GT bodies with any detection centroid within `match_xy_m` (default 0.55 m), regardless of YoloE label. `label_recall` additionally requires a substring match between detection and Robocasa category names — useful as a taxonomy diagnostic, but often low because YoloE returns open-vocab “best fit” strings (`cabinet`, `shelf`) while GT uses task categories (`chicken_drumstick`, `shrimp`). Low stretch `spatial_recall` on cab/counter is often a **viewpoint** issue (robot not facing manipulables), not a classifier failure.

**Methods note:** YoloE’s low `detection.confidence_threshold` is a **proposal** stage for instance masks / graph candidates (favor recall). Raising it is fine when calibration/find metrics improve or hold; don’t raise it casually to “clean up chat.” Chat captions come from the VLM and graph/voxel memory (`describe_scene`, `query_memory` / GraphEQA), with an optional separate `describe_confidence_threshold`. See [dynamem.md](dynamem.md) and [graph_eqa.md](graph_eqa.md).

```bash
# Full loop (both robots): writes /tmp/emet_fusion_tune/<robot>/ and copies tuned YAML under src/emet/config/agents/
./scripts/run_fusion_calibration_loop.sh all

# Manual steps
uv run emet serve mujoco --use-robocasa --robot innate_mars --headless --seed 0
uv run python scripts/fetch_sim_gt_from_server.py --robot innate_mars -o /tmp/gt.json
EMET_STRETCH_GENERIC_ZMQ=1 uv run emet run dynagraph --robot stretch --export /tmp/cal \
  --calibration-export /tmp/frames.jsonl --calibration-steps 36 --no-sensor-perception --cpu-only --no-rerun -N

# Assess raw detections (spatial vs label recall + taxonomy confusion table)
uv run emet eval-calibration --gt /tmp/gt.json --frames /tmp/frames.jsonl

# Grid-search fusion (objective: spatial_recall; optional --min-label-recall for strict taxonomy)
uv run emet tune-graph-fusion --gt /tmp/gt.json --frames /tmp/frames.jsonl --write-config
```

For **Stretch in Robocasa**, set `EMET_STRETCH_GENERIC_ZMQ=1` (default in the calibration loop script) so the client uses `GenericZmqClient` against the merged kitchen ZMQ server.

`--calibration-export` writes per-step instance detections (label, `xyz`, `bbox_xyxy`, optional embedding) to JSONL. `eval-calibration` reports association metrics; `tune-graph-fusion` replays frames through `GraphObjectFusion` and optimizes merge thresholds. Default fusion YAML sets `require_label_match: false` so live merging uses spatial + 3D + embedding; set `true` only for strict taxonomy experiments.

### Manual smoke (Robocasa + export)

- Start server as in **Robocasa** above with **`--use-robocasa`**.
- **`uv run emet run dynagraph --robot-ip 127.0.0.1 --explore-loop --explore-max-iters 20 --explore-max-failures 4 --export /tmp/graphtest`**.
- Confirm **`/tmp/graphtest/scene_graph_report.txt`** exists and is non-empty when the voxel map populated.

## Benchmarks

Unified episode scoring, question bank, fusion A/B, and environment smokes: **[dynagraph_benchmarks.md](dynagraph_benchmarks.md)**.

```bash
uv run emet eval-dynagraph --episode /tmp/export -o dynagraph_eval.json
uv run emet run dynagraph --export /tmp/ep --question-file src/emet/config/benchmarks/dynagraph_questions.yaml --question-env robocasa_seed0
./scripts/run_dynagraph_fusion_ab.sh innate_mars 0 20
```

## Testing

| Layer | Command |
|-------|---------|
| Unit (explore loop, graph memory) | `uv run emet test src/test/app/test_dynagraph_explore.py src/test/memory/test_graph_eqa_memory.py -v` |
| Benchmark smoke (unit) | `uv run emet test src/test/app/test_dynagraph_benchmark_smoke.py src/test/memory/test_dynagraph_staleness_disappearance.py -v` |
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
| [`src/emet/simulation/sim_object_placements.py`](../src/emet/simulation/sim_object_placements.py) | Session **`sim_object_placements`** + MJCF body scan for live sim GT. |
| [`src/emet/memory/graph_eqa/sim_ground_truth_graph.py`](../src/emet/memory/graph_eqa/sim_ground_truth_graph.py) | GT graph upsert, alignment reports, instance→GT association. |
| [`src/emet/simulation/mujoco_ground_truth.py`](../src/emet/simulation/mujoco_ground_truth.py) | Text/JSON snapshots of **`mjData.body(*).xpos`** for sim validation; triggered by **`mujoco_ground_truth_dump`** ZMQ recv command. |
| [`src/emet/simulation/mujoco_gt_objects.py`](../src/emet/simulation/mujoco_gt_objects.py) | Per-object **3D AABB** + optional head **2D bbox** JSON (`emet export-sim-gt`). |
| [`src/emet/memory/graph_eqa/graph_object_fusion/`](../src/emet/memory/graph_eqa/graph_object_fusion/) | **GraphObjectFusion** + offline **`emet eval-calibration`** / **`emet tune-graph-fusion`**. |
| [`src/emet/app/run_interactive.py`](../src/emet/app/run_interactive.py) | Shared interactive REPL for graph-EQA and task-mode apps. |

## See also

- [Testing index](TESTING.md) — master map of test docs, suites, and known gaps (graph + EQA on known scene).
- [Dynagraph Robocasa E2E](dynagraph_robocasa_e2e.md) — multi-robot floor-metrics harness and quality artefacts.
- [GraphEQA](graph_eqa.md) — baseline graph EQA without merge/staleness defaults.
- [Simulation](simulation.md) — MuJoCo / Robocasa and `emet serve mujoco`.
- [CLI](cli.md) — `emet run` apps table.
