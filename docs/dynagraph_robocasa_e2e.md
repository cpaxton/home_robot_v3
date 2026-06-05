# Dynagraph Robocasa multi-robot E2E

End-to-end comparison of **explored floor area** and **spawner walkability maps** for two robots on the same Robocasa kitchen (seed 0): **`innate_mars`** and **`galaxea_r1`**.

Experimental **Stretch + Robosuite Robocasa** (unified server, `GenericZmqClient`, lidar/render fixes) is on branch **`feature/stretch-robocasa-robosuite`**. This branch keeps Stretch on **`stretch_mujoco`** + **`StretchZmqClient`** (same as `main`).

**Testing index:** [TESTING.md](TESTING.md). **Next gap (graph + EQA on known scene):** [TESTING.md#known-gap-graph--eqa-on-a-known-scene-dynagraph](TESTING.md#known-gap-graph--eqa-on-a-known-scene-dynagraph).

**CLI:** Use **`uv run emet …`** from the project root (see [TESTING.md#run-from-this-repo](TESTING.md#run-from-this-repo)). The harness invokes `emet run dynagraph` from this repo’s `.venv` automatically.

See also: [Dynagraph overview](dynagraph.md), [Simulation](simulation.md), [Multi-robot testing plan](plans/MULTI_ROBOT_TESTING.md).

## Quick start

From the repo root (requires sim extra: `uv sync` with default groups, or `emet sync -e sim`):

```bash
uv run python src/test/app/run_dynagraph_multi_robot_e2e.py
```

Expect **~15–20 minutes** on a typical workstation (two sequential server + Dynagraph runs). Exit code **0** means all robots finished exploration and pairwise floor-area checks passed.

### Re-evaluate an existing run (no sim)

If exports already exist under the default output directory:

```bash
DYNAGRAPH_E2E_REPORT_ONLY=1 uv run python src/test/app/run_dynagraph_multi_robot_e2e.py
```

### Unit tests (fast, no sim server)

```bash
uv run emet test src/test/memory/test_floor_metrics.py -v
uv run emet test src/test/utils/test_nav_xyt_session.py -v
```

## What the harness does

For each robot, the script:

1. Starts **`emet serve mujoco --use-robocasa --robot ROBOT --headless --seed 0`**
2. Runs **`emet run dynagraph`** with **`--explore-loop --explore-max-iters 15`**, **`--export`**, **`--no-rerun`**, **`--cpu-only`**
3. Sets **`EMET_SIM_NAV_TELEPORT=1`** so frontier goals snap in sim (consistent coverage)
4. Wipes **`graph/`** before each run so stale **`floor_metrics.json`** cannot mask failures
5. Writes a combined report to **`/tmp/dynagraph_e2e_compare/comparison_report.json`**

Implementation: [`src/test/app/run_dynagraph_multi_robot_e2e.py`](../src/test/app/run_dynagraph_multi_robot_e2e.py).  
Metrics helpers: [`src/emet/memory/floor_metrics.py`](../src/emet/memory/floor_metrics.py).

## Where to find results

| Path | Contents |
|------|----------|
| **`/tmp/dynagraph_e2e_compare/comparison_report.json`** | Pairwise explored-area comparison, spawner vs explored summaries, per-robot metrics |
| **`/tmp/dynagraph_e2e_compare/{robot}/graph/floor_metrics.json`** | Explored cell count, area (m²), grid metadata, embedded spawner floor map |
| **`/tmp/dynagraph_e2e_compare/{robot}/graph/scene_graph_report.txt`** | Pretty graph export + explored-floor summary (stdout mirror) |
| **`/tmp/dynagraph_e2e_compare/{robot}/graph/frames/detections_*.json`** | Per-frame open-vocab detections (labels + world XYZ) for semantic quality review |
| **`/tmp/dynagraph_e2e_compare/{robot}/dynagraph.stdout`** | Full Dynagraph CLI log (explore steps, navigation, export) |
| **`/tmp/dynagraph_e2e_compare/{robot}/server.log`** | MuJoCo / Robocasa server log (spawn, walkable map, autoplace) |

Replace **`/tmp/dynagraph_e2e_compare`** by passing a different base directory only if you edit **`BASE`** in the harness script.

## Pass criteria

The harness exits **0** when:

1. **All three robots** produce a fresh **`floor_metrics.json`**
2. **Pairwise explored area** matches within **35% relative tolerance** (`rtol_area=0.35` in `compare_explored_floor_metrics`)
3. **Spawner scene walkable area** is identical across robots (same kitchen footprint): **`scene_walkable_area_m2`** within **5%** (typically **82.25 m²** for seed 0 / layout 1)

It does **not** require identical explored **cell counts** (local_radius and spawn pose differ slightly per robot).

### Reference run (2026-05-24)

Console summary from a passing full run:

```
=== innate_mars: running dynagraph ===
innate_mars: explored=21.77 m² scene_walkable=82.25 m² spawn_eroded=20.39 m²

=== galaxea_r1: running dynagraph ===
galaxea_r1: explored=16.59 m² scene_walkable=82.25 m² spawn_eroded=20.65 m²

=== stretch: running dynagraph ===
stretch: explored=24.32 m² scene_walkable=82.25 m² spawn_eroded=20.65 m²

Wrote /tmp/dynagraph_e2e_compare/comparison_report.json
compare innate_mars vs galaxea_r1: area_match=True cells_delta=518 (21.77 vs 16.59 m²)
compare innate_mars vs stretch: area_match=True cells_delta=255 (21.77 vs 24.32 m²)
scene_walkable_area_m2 range: 82.250 .. 82.250
```

| Robot | Explored | % of scene walkable | Spawner eroded spawn (m²) |
|-------|----------|---------------------|---------------------------|
| innate_mars | 21.77 m² | 26.5% | 20.39 |
| galaxea_r1 | 16.59 m² | 20.2% | 20.65 |
| stretch | 24.32 m² | 29.6% | 20.65 |

**Spawner parity:** all robots report **`scene_walkable_area_m2 = 82.25`** (full kitchen walkable footprint from occupancy). **Eroded spawn** maps (~20.4–20.65 m²) differ by robot footprint during autoplace.

## Example export text (`scene_graph_report.txt`)

Explore-only runs (no **`--question`**) often show **0 graph nodes** but still record navigation samples and floor metrics. Example tail from **`innate_mars`**:

```
────────────────────────────────────────────────────────
 Scene graph (Dynagraph export)
────────────────────────────────────────────────────────
 Nodes (0)
 Edges (0)
 Navigation samples (83) — camera views without a semantic graph node
   base=( -0.768,   2.397,   2.554)  anchor=(  1.984,   0.899,   0.259)
   ...
────────────────────────────────────────────────────────

--- Explored floor ---
robot='innate_mars' explored floor: 2177 cells, 21.770 m² (grid_resolution=0.100 m/cell)
spawner scene walkable map: 8225 cells, 82.250 m² (grid=0.1 m)
spawner walkable map: 2039 cells, 20.390 m² (clip_eroded=48.863)
explored / scene walkable: 26.5%
```

**`stretch`** and **`galaxea_r1`** exports follow the same layout; see their directories under **`/tmp/dynagraph_e2e_compare/`**.

## Assessing semantic / EQA quality

The E2E harness focuses on **geometric exploration parity**. For **label and EQA quality**, inspect per-robot artefacts:

### Per-frame detections

After export, open **`graph/frames/detections_NNNN.json`**. Example kitchen labels at frame 10:

**innate_mars** — range hood, paper towel, lamp, doorframe, …

```json
[
  {"label_short": "range hood", "xyz": [1.00, -0.79, 0.46]},
  {"label_short": "paper towel roll", "xyz": [0.80, -0.79, 0.41]},
  {"label_short": "lamp", "xyz": [3.11, -0.75, 1.14]}
]
```

**galaxea_r1** — windowsill, divider, scissors, …

```json
[
  {"label_short": "windowsill", "xyz": [2.01, -3.04, -0.14]},
  {"label_short": "scissors", "xyz": […]}
]
```

**stretch** — fewer detections at some poses (camera height / FOV); e.g. blinds, bathroom stall.

Compare label plausibility, XYZ spread in world frame, and consistency across robots on the **same seed**.

### Single EQA question (manual, per robot)

Run one robot at a time with a natural-language question after exploration:

```bash
# Terminal 1
uv run emet serve mujoco --use-robocasa --robot innate_mars --headless --seed 0

# Terminal 2
uv run emet run dynagraph --robot innate_mars --robot-ip 127.0.0.1 --no-rerun --cpu-only \
  --explore-loop --explore-max-iters 15 \
  --question "Where is the sink?" \
  --export /tmp/dynagraph_q/innate_mars
```

Use default **`dynav_config.yaml`** (`depth_source: sensor`) in Robocasa sim so ZMQ depth is used. **`dynav_innate_mars.yaml`** forces DA3 and is for the real robot (or sim without hardware depth); it can crash during rotate-in-place in kitchen sim (`Umeyama alignment` / degenerate poses).

Stdout includes the **GraphEQA answer** and an updated **`scene_graph_report.txt`** with **Nodes (N)** when observations merge into the graph. Repeat with **`--robot galaxea_r1`** for side-by-side EQA comparison.

## Robot-specific notes

| Robot | Robocasa server | Client | Dynav tuning |
|-------|-----------------|--------|--------------|
| **innate_mars** | `RobosuiteZmqServer` (planar autoplace) | `GenericZmqClient` | `dynav_parameter_overrides` on `RobotSpec` |
| **galaxea_r1** | `RobosuiteZmqServer` (freejoint autoplace) | `GenericZmqClient` | Default `dynav_config.yaml` |
| **stretch** (this branch) | `MujocoZmqServer` / stretch_mujoco | `StretchZmqClient` | Default `dynav_config.yaml` |
| **stretch** (experimental) | `RobosuiteZmqServer` on `feature/stretch-robocasa-robosuite` | `GenericZmqClient` | See that branch |

Navigation uses **world-frame** planning (`navigation_origin_xyt` from ZMQ session); see [`controller_dynamem.py`](../src/emet/controller/controller_dynamem.py) (`_planning_base_xyt`, `_robot_nav_xyt`).

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `No such option: --explore-loop` | Shell `emet` is not this repo’s install — use **`uv run emet`** from project root ([TESTING.md](TESTING.md#run-from-this-repo)) |
| Harness exit **1**, `area_match=False` | Run-to-run variance; re-run full harness or increase explore iters; check **`dynagraph.stdout`** for nav timeouts |
| Stale metrics / wrong robot numbers | Old **`graph/`** not wiped — fixed in harness; delete **`/tmp/dynagraph_e2e_compare`** and re-run |
| **innate_mars** 0 explore successes | Nav frame bug (fixed); verify **`explore-loop: step N/15 ok`** in stdout |
| **stretch** Robosuite migration | Use branch **`feature/stretch-robocasa-robosuite`** (not enabled on this branch) |
| Port **4401** in use | `uv run emet kill-mujoco-server` |

## See also

- [Dynagraph](dynagraph.md) — CLI, explore loop, export layout
- [GraphEQA](graph_eqa.md) — graph memory and EQA semantics
- [Testing backends](plans/TESTING_BACKENDS.md) — unit/smoke tests for memory backends
