# Testing index

Central map of **how to run tests**, **what each suite validates**, and **where detailed write-ups live**. From the repo root, use **`uv sync`** then **`uv run emet test`** (see [cli.md](cli.md#testing)).

## Run from this repo

Use **`uv run emet …`** (or `source .venv/bin/activate` then bare `emet`) from the **project root** so commands pick up **this checkout’s** code and virtualenv.

| Symptom | Fix |
|---------|-----|
| `No such option: --explore-loop` on `emet run dynagraph` | Wrong `emet` on PATH (another clone or old install). Run `which emet`; use **`uv run emet`** from this repo. Confirm with `uv run python -m emet.app.run_dynagraph --help` (should list `--explore-loop`) |
| Sim logs show `Sim navigation: driving toward` (no `[sim_nav]`) | **`emet` from another checkout** (e.g. `home_robot_v4`). From this repo: `cd ~/src/home_robot_v2 && emet serve …` (yellow “re-running: uv run emet”) or `./scripts/emet-v2 serve …` |
| rotate_in_place sends goals hundreds of meters away | Usually stale sim + wrong `nav_world`; run [test_rotate_in_place_robocasa_nav.py](../src/test/simulation/test_rotate_in_place_robocasa_nav.py) after nav changes |
| Tests skip sim unexpectedly | Default runs include sim; set `RUN_SIM_TESTS=0` or `uv run emet test --no-sim` to skip |
| Missing MuJoCo / Robocasa | `uv sync` with default groups, or `./install.sh --sim` |

## Quick commands

| Goal | Command |
|------|---------|
| **Simulation smoke battery (7 tracks)** | `./scripts/run_simulation_smoke_battery.sh` — see [simulation_testing_plan.md](simulation_testing_plan.md) |
| Motion planning (offline) | `uv run emet test src/test/motion/algo/test_rrt.py src/test/motion/test_arm_rrt.py src/test/motion/test_voxel_obstacle_planning.py -q` — see [motion_planning.md](motion_planning.md) |
| **No-NN sim pick/place** | `EMET_SIM_NAV_TELEPORT=1 uv run python scripts/scripted_sim_pick_place.py --start-sim --sim configs/sim/default_table_rby1.yaml --manip-mode kinematic` — GT+MuJoCo only ([motion_planning.md](motion_planning.md#no-neural-nets-smoke-sim-only)) |
| **Molmo grasp oracle MP** | `uv run emet test src/test/perception/grasps/ src/test/motion/test_arm_manip_profile.py -q`; optional smoke `scripts/scripted_molmo_grasp_mp.py` ([motion_planning.md](motion_planning.md#molmospaces-grasp-oracle-multi-robot)) |
| Full suite (sim on by default) | `uv run emet test` |
| Skip sim (faster CI-style) | `uv run emet test --no-sim` |
| Verbose | `uv run emet test -v` |
| Memory backend smokes | `uv run emet test src/test/memory/test_memory_backends_smoke.py -v` |
| Multi-robot Dynagraph floor E2E | `uv run python src/test/app/run_dynagraph_multi_robot_e2e.py` |
| Dynagraph unit (explore loop, graph memory) | `uv run emet test src/test/app/test_dynagraph_explore.py src/test/memory/test_graph_eqa_memory.py -v` |
| GraphObjectFusion + GT export (fast) | `uv run emet test src/test/memory/test_graph_object_fusion.py src/test/simulation/test_mujoco_gt_objects.py -v` |
| Dynagraph benchmark smoke (unit) | `uv run emet test src/test/app/test_dynagraph_benchmark_smoke.py -v` |
| Dynagraph staleness / disappearance | `uv run emet test src/test/memory/test_dynagraph_staleness_disappearance.py -v` |
| Unified Dynagraph eval CLI | `uv run emet eval-dynagraph --episode /tmp/export` |
| SQA3D benchmark (unit) | `uv run emet test src/test/benchmarks/sqa3d/ -v` |
| SQA3D ScanNet embodied smoke | `uv run python scripts/run_sqa3d_scannet_smoke.py` |
| SQA3D EM@1 scoring | `uv run emet eval-sqa3d -p preds.jsonl --split val` |
| Graph fusion calibration (one scene) | `emet export-sim-gt` → `emet run dynagraph --calibration-export` → `emet tune-graph-fusion` (see [dynagraph.md](dynagraph.md#object-gt-export-and-graphobjectfusion-calibration)) |
| GraphEQA human-answer formatter | `uv run emet test src/test/memory/test_graph_eqa_human_answer.py -v` |
| Manual Dynagraph EQA + export (Robocasa) | See [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#single-eqa-question-manual-per-robot) |
| Dynagraph Robocasa question CLI (CI) | `uv run emet test src/test/app/test_dynagraph_robocasa_question_cli.py -v` |

Environment: **`RUN_SIM_TESTS=0`** skips sim integration tests. Heavy VLLM download tests use marker **`vllm_load`** and are excluded from default runs — see [plans/TESTING_VLLM_LOAD.md](plans/TESTING_VLLM_LOAD.md).

---

## Documentation by topic

| Topic | Doc | What it covers |
|-------|-----|----------------|
| **Motion planning (base + arm)** | [motion_planning.md](motion_planning.md) | RRT / A\* on voxel maps, kinematic arm RRT-Connect, offline tests |
| **Memory backends (SVM, DynaMem, GraphEQA)** | [plans/TESTING_BACKENDS.md](plans/TESTING_BACKENDS.md) | Test matrix, red-cylinder / Robocasa spin integration, backend unit tests |
| **Dynagraph multi-robot Robocasa E2E** | [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md) | Floor-area parity (stretch / innate_mars / galaxea_r1), spawner maps, artefact paths |
| **GraphEQA manual Robocasa** | [graph_eqa.md](graph_eqa.md#testing-graph-eqa-in-robocasa) | Interactive questions in kitchen sim (Gemini / encoder) |
| **MolmoSpaces** | [plans/2025-03-10_molmospaces_testing.md](plans/2025-03-10_molmospaces_testing.md), [molmospaces.md](molmospaces.md) | Wrapper venv, CLI smoke, optional integration; **multi-robot matrix below** |
| **Simulation smoke battery (paper)** | [simulation_testing_plan.md](simulation_testing_plan.md) | Seven-track sequential Habitat + Robocasa + Molmo + SQA3D validation |
| **Multi-robot sim plumbing** | [plans/MULTI_ROBOT_TESTING.md](plans/MULTI_ROBOT_TESTING.md) | Registry, MJCF load, GenericZmqClient (partially superseded by unit tests) |
| **CLI / install smoke** | [cli.md](cli.md#testing), [simulation.md](simulation.md#2-test-the-setup) | `emet test`, serve smoke |
| **SQA3D + ScanNet EQA** | [sqa3d.md](sqa3d.md) | Situated QA loaders, EM@1 eval, Open3D mesh replay |
| **Agent / LLM** | [AGENT_RUN.md](AGENT_RUN.md#testing), [llm_agent.md](llm_agent.md#testing-the-llm-agent) | Agent loop, component tests |
| **Refactor logs (not test specs)** | [logs/README.md](logs/README.md) | Historical change notes |

There is **no other single “master” file** today; this page is the hub. Feature-specific docs link back here where useful.

---

## Automated test map (by validation goal)

### Geometry / navigation / floor coverage

| Validates | Test / harness | Notes |
|-----------|----------------|-------|
| Offline RRT / arm RRT on voxel-shaped grids | [test_voxel_obstacle_planning.py](../src/test/motion/test_voxel_obstacle_planning.py), [test_arm_rrt.py](../src/test/motion/test_arm_rrt.py) | Unit; [motion_planning.md](motion_planning.md) |
| Explored floor vs spawner walkable (3 robots, Robocasa) | [run_dynagraph_multi_robot_e2e.py](../src/test/app/run_dynagraph_multi_robot_e2e.py) | [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md); **does not** assert graph nodes or EQA |
| Nav world ↔ session frame | [test_nav_xyt_session.py](../src/test/utils/test_nav_xyt_session.py) | Unit |
| **rotate_in_place** world (x,y) stays at spawn (innate_mars + Robocasa) | [test_rotate_in_place_robocasa_nav.py](../src/test/simulation/test_rotate_in_place_robocasa_nav.py) | Sim; asserts `spawn_compose` goals, not `nav_world` |
| **VLM-free multi-env nav/explore** (Robocasa L1, OVMM Robocasa L2, Molmo iTHOR 0) | [test_multi_env_nav_explore_smoke.py](../src/test/simulation/test_multi_env_nav_explore_smoke.py) | Sim; no Qwen/VL worker — `rotate_in_place` + one `run_exploration`; asserts `|Δxy|≥0.08m` or explored-cell growth. Helper: [`nav_explore_smoke.py`](../src/emet/eval/nav_explore_smoke.py). Run when no other MuJoCo/EGL job is live: `uv run emet test -v src/test/simulation/test_multi_env_nav_explore_smoke.py` |
| Robocasa freejoint spawn (Stretch / Galaxea, collision autoplace) | [test_robocasa_freejoint_spawn.py](../src/test/simulation/test_robocasa_freejoint_spawn.py) | Sim; `EMET_ROBOSUITE_AUTOPLACE` + `find_robocasa_freejoint_xyz` (no startup nav rollback) |
| Robosuite nav frame unit (spawn compose, jump guard) | [test_robosuite_nav_world_clamp.py](../src/test/simulation/test_robosuite_nav_world_clamp.py) | Unit |
| Floor metrics export | [test_floor_metrics.py](../src/test/memory/test_floor_metrics.py) | Unit |
| Red cylinder localize (DynaMem, default scene) | [test_red_cylinder_in_sim.py](../src/test/mapping/test_red_cylinder_in_sim.py) | Sim; stretch + innate_mars |
| Robocasa memory after spin | [test_robocasa_memory_after_spin.py](../src/test/simulation/test_robocasa_memory_after_spin.py) | Sim |

### Scene graph / object labels (model-derived)

| Validates | Test / harness | Notes |
|-----------|----------------|-------|
| **OpenVocab scene graph** in Robocasa (dedup, kitchen labels) | [test_scene_graph_robocasa.py](../src/test/scene_graph/test_scene_graph_robocasa.py) | Uses `SceneGraphProcessor` + `cpu_scene_graph`; **not** GraphEQAMemory / Dynagraph path |
| Open-vocab unit behaviour | [test_open_vocab_scene_graph.py](../src/test/scene_graph/test_scene_graph_in_sim.py) | Default scene spin |
| GraphEQA memory unit (merge, staleness, query) | [test_graph_eqa_memory.py](../src/test/memory/test_graph_eqa_memory.py) | Mock EQA clients; package map: [graph_memory.md](graph_memory.md) |
| GraphEQA + default scene (red/blue) | [test_graph_eqa_default_scene_sim.py](../src/test/memory/test_graph_eqa_default_scene_sim.py) | Sim RGB + **injected** graph labels; mocked `query_answer` |
| Per-frame detections on export | Dynagraph `--export` → `graph/frames/detections_*.json` | Manual review; see [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#assessing-semantic--eqa-quality) |

### EQA / answering questions

| Validates | Test / harness | Notes |
|-----------|----------------|-------|
| Mocked EQA on constructed graph | [test_graph_eqa_default_scene_sim.py](../src/test/memory/test_graph_eqa_default_scene_sim.py) | Known scene; labels not from full VL pipeline |
| Interactive Robocasa EQA | Manual — [graph_eqa.md](graph_eqa.md#testing-graph-eqa-in-robocasa) | Requires API keys / models |
| **Dynagraph graph build + NL answer on known scene** | [`test_dynagraph_benchmark_smoke.py`](../src/test/app/test_dynagraph_benchmark_smoke.py) (unit); sim harness [`run_dynagraph_benchmark_smoke.py`](../src/test/app/run_dynagraph_benchmark_smoke.py) | Question bank + `eqa_results.json`; full sim via `RUN_DYNAGRAPH_BENCHMARK_SMOKE=1` |

### Infrastructure / CLI / controllers

| Validates | Test / harness |
|-----------|----------------|
| `emet` CLI, `run dynagraph` help | [test_cli.py](../src/test/cli/test_cli.py) |
| Controller smoke (SVM, DynaMem, GraphEQA, Dynagraph imports) | [test_controller_smoke.py](../src/test/controller/test_controller_smoke.py) |
| Multi-robot registry / MJCF | [test_multi_robot.py](../src/test/simulation/test_multi_robot.py) |
| Dynagraph explore loop helper | [test_dynagraph_explore.py](../src/test/app/test_dynagraph_explore.py) |

### MolmoSpaces multi-robot (serve, merge, navigation)

Robot registry and aliases: [robots/supported_robots.md](robots/supported_robots.md). MolmoSpaces install, merge, and troubleshooting: [molmospaces.md](molmospaces.md#quick-verification-developers).

**Quick serve smoke** (repo root, sim + wrapper installed; first iTHOR load can take 1–2 minutes):

```bash
# Merge + ZMQ load; expect no MJCF errors and autoplace log (not stuck at origin)
for r in stretch rby1 innate_mars xlerobot; do
  echo "=== $r ==="
  timeout 120 uv run emet serve mujoco --scene ithor --robot "$r" --headless
done
```

| Robot | `--robot` | Server stack | Serve / merge tests | Nav / mapping tests |
|-------|-----------|--------------|---------------------|---------------------|
| **Stretch** | `stretch` | `MujocoZmqServer` (Stretch stack) | `RUN_MOLMOSPACES_TESTS=1` → [test_molmospaces_ithor_base_settle.py](../src/test/molmospaces/test_molmospaces_ithor_base_settle.py) (rby1-oriented settle; stretch uses same merge path) | `RUN_STRETCH_MOLMO_DYNAMEM=1` → [test_stretch_molmospaces_dynamem_floor_map.py](../src/test/molmospaces/test_stretch_molmospaces_dynamem_floor_map.py) |
| **Galaxea R1** | `rby1`, `galaxea_r1` | `RobosuiteZmqServer` | [packages/emet_molmospaces/tests/test_rby1_scene.py](../packages/emet_molmospaces/tests/test_rby1_scene.py) (merge + step); `RUN_MOLMOSPACES_TESTS=1` → [test_molmospaces_ithor_base_settle.py](../src/test/molmospaces/test_molmospaces_ithor_base_settle.py) | `RUN_MULTI_ROBOT_NAVGRID=1` → [test_multi_robot_molmospaces_navgrid_similarity.py](../src/test/molmospaces/test_multi_robot_molmospaces_navgrid_similarity.py) |
| **Innate Mars** | `innate_mars`, `maurice` | `RobosuiteZmqServer` (planar) | [test_multi_robot.py](../src/test/simulation/test_multi_robot.py) (spec/MJCF); wrapper merge: `test_merge_innate_mars_into_minimal_scene_no_double_floor` | Robocasa Dynagraph E2E: [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md); Molmo navgrid: `RUN_MULTI_ROBOT_NAVGRID=1` (above); `EMET_NAVGRID_COMPARE_ROBOTS=stretch,rby1,innate_mars` + [scripts/tier4_multi_robot_navgrid_compare.py](../scripts/tier4_multi_robot_navgrid_compare.py) |
| **XLeRobot** | `xlerobot` | `RobosuiteZmqServer` (planar) | wrapper `test_merge_xlerobot_into_minimal_scene`, `test_merge_stretch_into_scene_with_memory` (merge `memory` + robot `njmax` guard) | `RUN_XLEROBOT_DYNAMEM=1` → [test_xlerobot_dynamem_default_table_smoke.py](../src/test/molmospaces/test_xlerobot_dynamem_default_table_smoke.py); `RUN_XLEROBOT_MOLMO_DYNAMEM=1` → [test_xlerobot_molmospaces_dynamem_floor_map.py](../src/test/molmospaces/test_xlerobot_molmospaces_dynamem_floor_map.py); `RUN_STRETCH_XLEROBOT_NAVGRID=1` → [test_stretch_xlerobot_navgrid_similarity.py](../src/test/molmospaces/test_stretch_xlerobot_navgrid_similarity.py) |

**Cross-robot Dynagraph / DynaMem nav benchmark** (default table + Robocasa + MolmoSpaces tiers):

```bash
EMET_SIM_NAV_TELEPORT=1 uv run python src/test/app/run_dynagraph_nav_benchmark.py --robot xlerobot --dynamem --default --molmo
```

See [dynagraph_nav_benchmark.md](dynagraph_nav_benchmark.md). Molmo iTHOR GT query on train index 0 is **sink** (not sofa).

**Dynagraph + Stretch on MolmoSpaces iTHOR** (manual): terminal A `emet serve mujoco --scene ithor --robot stretch`, terminal B `emet run dynagraph`. If you see `Timeout waiting for navigation step` while the sim teleports, check branch `fix/stretch-molmo-nav-wait` (relative `move_base_to` must not double-compose with server `nav_relative`). Rerun `world/robot` uses `gps`/`compass` + `navigation_origin_xyt`; if the base marker is frozen but the voxel map moves, confirm ZMQ obs `gps` is updating and session has `navigation_origin_xyt`.

**Wrapper-only pytest** (no live iTHOR assets):

```bash
uv run pytest packages/emet_molmospaces/tests/ -q
```

---

## Known gap: graph + EQA on a known scene (Dynagraph)

**Status (2026-07):** Unit coverage landed in [`test_dynagraph_known_scene_attach.py`](../src/test/memory/test_dynagraph_known_scene_attach.py) — red cylinder / blue cube instance items and fusion detections must attach as object nodes with allowlisted labels (no GPU). World-change invalidation: [`test_dynagraph_staleness_disappearance.py`](../src/test/memory/test_dynagraph_staleness_disappearance.py) + [`test_lifelong_checkpoint_invalidate.py`](../src/test/eval/test_lifelong_checkpoint_invalidate.py). Operator notes: [experiments/dynagraph_dynamic_memory.md](experiments/dynagraph_dynamic_memory.md). Full Robocasa/MuJoCo E2E with live YoloE remains a stronger integration check.

**Remaining E2E gap:** Explore-only exports can still show **`Nodes (0)`** in `scene_graph_report.txt` when detections exist but the instance→graph hook did not run. Prefer graph-health fields in Habitat `metrics.json` / dynamic-explore cycle rows (`graph_health`) and `uv run python scripts/summarize_graph_health.py …`.

| Step | Detail |
|------|--------|
| Unit (CI) | `uv run emet test src/test/memory/test_dynagraph_known_scene_attach.py --no-sim` |
| Scene E2E | Default MuJoCo table or Robocasa seed 0 with `--export` + `--question` |
| Assert graph | `graph_health.n_object ≥ 2` or allowlisted labels in `scene_graph_report.txt` |
| Health triage | `scripts/summarize_graph_health.py` → `blowup` / `fragmentation` / `empty_graph` / `ok` |

Until full E2E is green in CI, use **manual** `--question` runs (documented in [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md#assessing-semantic--eqa-quality)) and inspect **`detections_*.json`** + **`scene_graph_report.txt`**.

---

## See also

- [Dynagraph](dynagraph.md) — CLI, explore loop, export layout
- [GraphEQA](graph_eqa.md) — graph memory design and manual Robocasa testing
- [plans/README.md](plans/README.md) — design plans (architecture, GraphEQA plan, mapping refactor)
