# Motion planning

How EMET plans **base** and **arm** motion. Both share the same planners under [`emet.motion.algo`](../src/emet/motion/algo/) and (for collision) the agent **voxel / 2D obstacle map** — not CuRobo and not Molmo / MuJoCo mesh geometry for agent-time collision.

| Layer | Space | Default planner | Collision |
|-------|--------|-----------------|-----------|
| **Base nav** | XYT on `SparseVoxelMapNavigationSpace` | `a_star` or `rrt_connect` (config) | Footprint vs `get_2d_map()` obstacles |
| **Arm (kinematic manip)** | Joint space (torso + arm) | `rrt_connect` | Link XY vs same 2D obstacle grid (`VoxelMapArmCollisionChecker`) |

Related: [dynamem.md](dynamem.md) (voxel maps), [molmospaces.md](molmospaces.md#mobile-manipulation-sim-teleport--kinematic) (sim pick/place modes), [llm_agent.md](llm_agent.md) (tuning `motion_planner` on real Stretch), [TESTING.md](TESTING.md), [attempt_ledger.md](attempt_ledger.md) (structured nav / closer-look outcomes).

**Closer look (CHAT `aim_arm_at`):** [`emet.controller.manipulation.closer_look.aim_wrist_at_phrase`](../src/emet/controller/manipulation/closer_look.py) localizes a phrase and aims the wrist/EE when kinematic aim is available; structured failure codes feed the opt-in action-outcome ledger. `take_ee_picture` is gated on a successful aim grant (one capture per aim).

**Nav attempt status:** [`emet.controller.nav_attempt`](../src/emet/controller/nav_attempt.py) maps `NavAttemptResult` → stable `status_code` / ledger fields used by CHAT diagnostics and agentic place cards.

---

## Package layout

```
src/emet/motion/
  algo/           # RRT, RRT-Connect, A*, Shortcut, SimplifyXYT
  base/           # ConfigurationSpace, Planner, PlanResult
  arm_rrt.py      # Joint-space RRT-Connect wrapper for kinematic manip
  arm_manip_profile.py  # Per-robot EE / joint / gripper / home profiles (discovered from MJCF)
  voxel_arm_collision.py
  mujoco_arm_ik.py
  base_goal_rank.py     # Multi-goal base XY on navigable grid (one A*/Dijkstra)
  utils/simple_env.py   # 2D box obstacle toy env (unit tests)

src/emet/mapping/voxel/
  voxel_map.py    # SparseVoxelMapNavigationSpace (base XYT + footprint)
```

Entry points:

- Base: `get_planner(algo, space, validate_fn)` → used by Dynamem / instance-memory controllers.
- Arm: `plan_arm_joint_path(...)` in [`arm_rrt.py`](../src/emet/motion/arm_rrt.py), called from [`KinematicPickPlaceExecutor`](../src/emet/controller/manipulation/kinematic_pick_place.py).

---

## Base navigation (voxel map)

1. Build / update the sparse voxel map from RGB-D.
2. `SparseVoxelMapNavigationSpace` samples free XYT and checks the robot **footprint** against dilated obstacles (`is_valid`).
3. Planner (`motion_planner.algorithm` in dynav / mapping YAML): typically **`a_star`** or **`rrt_connect`**, optionally wrapped in `Shortcut` / `SimplifyXYT`.

Config block (see [dynav_config.md](dynav_config.md)):

```yaml
motion_planner:
  step_size: 0.05
  rotation_step_size: 0.1
  algorithm: "rrt_connect"   # rrt | rrt_connect | a_star
  shortcut_plans: false
  simplify_plans: false
```

Preset files: `src/emet/config/default_planner.yaml`, `a_star_planner.yaml`, `sim_planner.yaml`.

**Important:** A\* is wired to `space.voxel_map.get_2d_map()` (navigable = explored ∧ ¬obstacle). RRT\* uses the same `is_valid` footprint check.

### Clearance-safe path simplification (anti-stuck)

Dynamem always post-processes A\* trajectories with `AStar.clean_path` / `clean_path_for_xy`. Line-of-sight is **clearance-aware**: Bresenham + dense samples reject any cell that is obstacle, unexplored, or below `min_clearance_m` (same hard gate as search). That stops LOS chords from cutting corners into walls after a clearance-preferring plan.

Before exec, `_filter_unsafe_nav_traj` also checks mid-segments between waypoints. On `aborted_waypoint_timeout`, planner failure, or Habitat navmesh stuck/noop, Dynamem marks the goal in `_habitat_blocked_goals` so multi-goal / uncover explore skips it and replans elsewhere.

Arm RRT-Connect wraps [`Shortcut`](../src/emet/motion/algo/shortcut.py); every mid-config from `ConfigurationSpace.extend` must validate (prefer no shorten over an unsafe chord).

**Sim impact note:** HM-EQA agentic with `eqa.habitat_perfect_nav` routes `navigate_to_target_pose` / explore through **navmesh** (not A\* `clean_path`). Clearance-safe simplify mainly helps Molmo/Robocasa Dynamem and any Habitat path that falls back to voxel A\*. Stuck-goal marking still helps Habitat agentic when navmesh returns noop/stuck so frontiers are not re-picked. Expect modest holdout lifts vs baseline; do not treat a flat accuracy curve as a motion-planner failure.

```bash
uv run emet test src/test/motion/test_a_star_clearance.py src/test/motion/algo/test_rrt.py src/test/controller/test_nav_abort_blocked.py --no-sim -q
```

---

## Two `manip_mode` namespaces

Do not conflate OVMM harness flags with chat-agent YAML:

| Namespace | Flag / key | Values | Used by |
|-----------|------------|--------|---------|
| **OVMM harness** | `--manip-mode` | `skip` \| `oracle` \| `sim` \| `attempt` | [`eval_ovmm_full.py`](../scripts/eval_ovmm_full.py) → [`ovmm_full.py`](../src/emet/eval/ovmm_full.py). `sim` snaps object freejoints via ZMQ; no DynamemTaskExecutor. |
| **Chat / scripted agent** | `agent.manip_mode` / `EMET_MANIP_MODE` | `teleport` \| `kinematic` | [`DynamemTaskExecutor`](../src/emet/controller/task/dynamem/dynamem_task.py) after `_merge_chat_agent_manip_parameters` in the agent loop. |

**Stretch MuJoCo default:** when the server advertises `sim_set_body_pose` and visual-servo is **off**, chat pick/place uses **GT teleport** (`prefer_sim_teleport_manip`). Pass **`--visual-servo` / `-V`** on `emet run agent` to keep the Stretch AnyGrasp / visual-servo path. OVMM `sim` always teleports object bodies regardless of `-V`.

Details for OVMM: [ovmm_full_benchmark.md](ovmm_full_benchmark.md#ovmm---manip-mode--chat-agentmanip_mode). Molmo agent modes: [molmospaces.md](molmospaces.md#mobile-manipulation-sim-teleport--kinematic).

---

## Arm kinematic planning (RRT-Connect)

Used when `agent.manip_mode: kinematic` (or `EMET_MANIP_MODE=kinematic`) on robots that advertise `kinematic_manip` (e.g. rby1).

Pipeline per EE target:

1. Sync base freejoint + joints into an offline MuJoCo model.
2. **Position IK** → goal joint vector `q_goal`.
3. **Joint-space path** `q_start → q_goal`:
   - Default **`rrt_connect`** (`agent.manip_planner` / `EMET_MANIP_PLANNER`).
   - Optional `linear` interpolation.
   - On RRT failure, falls back to linear (still rejected if voxel collision is on).
4. Stream actuator waypoints over ZMQ; optional **`sim_attach_body`** for grasp.

| Knob | Values | Default |
|------|--------|---------|
| `agent.manip_mode` / `EMET_MANIP_MODE` | `teleport` \| `kinematic` | `teleport` |
| `agent.manip_collision` / `EMET_MANIP_COLLISION` | `none` \| `voxel` | `none` |
| `agent.manip_planner` / `EMET_MANIP_PLANNER` | `rrt_connect` \| `rrt` \| `linear` | `rrt_connect` |

### Voxel collision for the arm

[`VoxelMapArmCollisionChecker`](../src/emet/motion/voxel_arm_collision.py) samples named link XY after FK and queries the **same** 2D obstacle grid as base nav.

Grid indexing matches `GridParams` / `SparseVoxelMap`:

```text
grid_i = floor(world_x / resolution + grid_origin[0])
grid_j = floor(world_y / resolution + grid_origin[1])
```

`from_voxel_map(voxel_map)` reads `get_2d_map()` and prefers `voxel_map.grid.grid_origin` (cell indices). Synthetic tests may use `convention="world_offset"` (origin in meters).

---

## Offline tests (no sim / no GPU)

```bash
# Core planners (2D toy env)
uv run emet test src/test/motion/algo/test_rrt.py -q

# Arm RRT helpers + resolve
uv run emet test src/test/motion/test_arm_rrt.py -q

# Voxel-map-shaped grids: base RRT around wall; arm RRT vs linear; from_voxel_map convention
uv run emet test src/test/motion/test_voxel_obstacle_planning.py -q

# IK smoke + TAMP packing / attach protocol helpers
uv run emet test src/test/motion/test_rby1_mujoco_arm_ik.py src/test/motion/test_kinematic_tamp_helpers.py -q
```

| Test file | What it proves |
|-----------|----------------|
| `test_rrt.py` | RRT / RRT-Connect / Shortcut on `SimpleEnv` |
| `test_arm_rrt.py` | Config resolve; 2D wall; arm joint RRT without map |
| `test_voxel_obstacle_planning.py` | `GridParams` indexing; `from_voxel_map`; base RRT on fake SVM; arm RRT avoids wall; multi-goal A* rejects sealed frontier |
| `test_rby1_mujoco_arm_ik.py` | Offline MuJoCo position IK |
| `test_svm.py` (mapping) | Heavier: real pickle map + `plan_to_instance` / frontier (optional data) |

## No-neural-nets smoke (sim only)

GT placements + MuJoCo — **no** LLM / VLM / SigLIP / YOLO. Script: [`scripts/scripted_sim_pick_place.py`](../scripts/scripted_sim_pick_place.py).

```bash
# Teleport (oracle GT snap) — table + rby1
uv run python scripts/scripted_sim_pick_place.py --start-sim \
  --sim configs/sim/default_table_rby1.yaml --manip-mode teleport \
  --object "red cylinder" --receptacle "blue cube"

# Kinematic (IK + RRT-Connect + attach)
EMET_SIM_NAV_TELEPORT=1 uv run python scripts/scripted_sim_pick_place.py --start-sim \
  --sim configs/sim/default_table_rby1.yaml --manip-mode kinematic \
  --object "red cylinder" --receptacle "blue cube"
```

Expect `OK: measured object move…` and `displacement_m` ≳ 0.05. Offline planner unit tests (no sim): see [Offline tests](#offline-tests-no-sim--no-gpu) above.

## Agent tool stub (no LLM)

CI-safe proof that the CHAT agent tool path drives manip / kinematic MP **without** loading an LLM or VLM. Inject a canned `tool_calls` list through [`_dispatch_tool_calls`](../src/emet/agent/loop.py) (e.g. `find_objects` then `pick_place` → `find` / `pickup`+`place`). Test: [`src/test/agent/test_agent_manip_tool_sequence.py`](../src/test/agent/test_agent_manip_tool_sequence.py).

```bash
uv run emet test src/test/agent/test_agent_manip_tool_sequence.py -q
```

Manual no-LLM letter path (put both args on the same line so the loop does not call `input()`):

```bash
emet run agent --no-llm -c "M bowl table"
```

Scripted MuJoCo smokes above still bypass the agent loop; use the stub test for agent↔MP wiring, and the scripts for behavior under sim.

## Task search (no-NN TAMP)

Deterministic beam-style search over `approach → grasp → place` (no Qwen / chat agent). Grasp candidates are ranked by offline position IK **at the approach base pose** before execution. Multi-option smoke plants IK-unreachable decoys first so ranking must skip them.

```bash
# Offline (no sim) — includes decoy-skip unit tests (plant helper: tamp/smoke_grasps.py)
uv run emet test src/test/controller/task/test_task_search.py src/test/visualization/test_manip_figures.py -q

# Multi-option table + rby1 kinematic smoke + figures
EMET_SIM_NAV_TELEPORT=1 MUJOCO_GL=egl \
  uv run python scripts/scripted_tamp_pick_place.py --start-sim \
  --sim configs/sim/default_table_rby1.yaml \
  --object "red cylinder" --receptacle "blue cube" \
  --plant-infeasible-grasps --cpu-only --skip-oracle

# Same + third-person MP4 (action/goal overlays)
EMET_SIM_NAV_TELEPORT=1 MUJOCO_GL=egl \
  uv run python scripts/scripted_tamp_pick_place.py --start-sim \
  --sim configs/sim/default_table_rby1.yaml \
  --object "red cylinder" --receptacle "blue cube" \
  --plant-infeasible-grasps --cpu-only --skip-oracle --record-mp4
# → <figures-dir>/third_person.mp4

# Sourccey: table objects at z≈0.6 m (floor picks are out of reach). RoboCasa
# PickPlace must pin obj_main — the GT category is not "obj". Pass `--rerun`
# for the web viewer (http://127.0.0.1:9090?url=ws://127.0.0.1:9877); the
# process holds 30s after the plan so a failed IK still leaves something to look at.
# Approach is a **side** standoff (`tamp_approach=side`); the rby1 front pose is unreachable.
# `--start-sim` sets EMET_SIM_NAV_TELEPORT so approach snaps instead of a 30s drive.
# Spawn restores table-object freejoints after sourccey_home (otherwise the cube falls to z≈0).
uv run python scripts/scripted_tamp_pick_place.py --start-sim \
  --sim configs/sim/default_table_sourccey.yaml --manip-mode kinematic --skip-oracle --rerun
uv run python scripts/scripted_tamp_pick_place.py --start-sim \
  --sim configs/sim/robocasa_pick_place_sourccey.yaml --object obj --receptacle cab \
  --object-gt-body obj_main --manip-mode kinematic --skip-oracle --rerun
# Paper stills: kitchen-orbit chase MP4 + stills/{start,approach,grasp,place,final}_*.png
uv run python scripts/scripted_tamp_pick_place.py --start-sim \
  --sim configs/sim/default_table_sourccey.yaml --manip-mode kinematic --skip-oracle --record-mp4
```

Expect `chosen_grasp` on a `reachable=True` candidate (not decoy index 0), `execute success=True`, and `displacement_m` ≳ 0.05. Figures under `~/runs/emet/tamp_pick_place/<stamp>/` (or `--figures-dir`). With `--record-mp4`, the sim streams a **chase camera** off ``base_link`` (FREE cam, kitchen-orbit defaults) plus a nadir **overhead** still and onboard POV (head / wrist) under ``stills/``. Tune chase with ``EMET_SIM_THIRD_PERSON_{DISTANCE,AZIMUTH,ELEVATION,LOOKAT_Z}``; overhead with ``EMET_SIM_OVERHEAD_{DISTANCE,LOOKAT_Y,LOOKAT_Z}``.

### Frontier multi-option (explore)

Base multi-goal planning: one shared A*/Dijkstra on the navigable grid toward a set of XY goals; nearest reachable wins (sealed / unreachable goals are ignored). Prefer this over K independent RRT ranks.

**Live Dynamem explore** (`mode=exploration`): collects top-K frontier XYs via [`collect_explore_frontier_candidates`](../src/emet/motion/frontier_goals.py) (graph frontiers + voxel heuristics), maps each with `sample_navigation`, then calls `AStar.plan(start, goals=[...])`. Object-localize / navigation mode stays single-goal. Habitat navmesh explore is unchanged (helper returns `[]` for Habitat clients).

```bash
uv run emet test src/test/motion/test_voxel_obstacle_planning.py::test_multi_frontier_goals_pick_reachable_reject_sealed -q
uv run emet test src/test/motion/test_frontier_goals.py src/test/motion/test_voxel_obstacle_planning.py::test_astar_plan_goals_kwarg_sets_goal_index --no-sim -q
```

API: [`plan_xy_multi_goal`](../src/emet/motion/base_goal_rank.py) / `AStar.plan(..., goals=[...])` (sets `PlanResult.goal_index`).

Figures: `topdown`, `ee_path_xz`, `joint_traj`, `plan_tree` (matplotlib / Agg). Optional live debug overlays live under `world/manip/…` in Rerun when a visualizer is attached; **figures are the paper path**.

**Known limitation:** MuJoCo IK is currently **position-only** — grasp orientation from the Molmo oracle sets the approach standoff axis but is not enforced at the EE.

## MolmoSpaces grasp oracle (multi-robot)

Fake grasp predictor over ZMQ using on-disk MolmoSpaces NPZ grasps (`~/.cache/molmospaces/assets/grasps`). Robot-agnostic world-frame poses; execution dispatches by server caps:

| Cap | Executor |
|-----|----------|
| `kinematic_manip` | [`KinematicPickPlaceExecutor`](../src/emet/controller/manipulation/kinematic_pick_place.py) + [`ArmManipProfile`](../src/emet/motion/arm_manip_profile.py) |
| `sim_set_body_pose` only | Teleport object to grasp XYZ ([`sim_teleport_to_grasp_pose`](../src/emet/simulation/sim_manipulation.py)) — Stretch etc. |

### ArmManipProfile discovery (no hardcoded table)

`ArmManipProfile.for_robot()` first checks the small explicit table (`rby1` / `galaxea_r1`
shared arms + gripper fingers), then falls back to **discovery from the robot's spec +
vendored MJCF** via `discover_from_spec()`. This is what makes every registry robot with an
arm pick-planable without per-robot wiring:

- **Arm joints** by side convention — `left_`/`right_` prefix (sourccey, rby1), `*_L`/`*_R`
  suffix (xlerobot), or a single un-suffixed chain (innate_mars, franka_fr3).
- **EE body** = deepest terminal body scored by gripper/jaw/finger/ee/hand tokens
  (`Moving_Jaw`, `ee_link`, `fr3_link7`, …).
- **Gripper contact bodies** = finger/jaw bodies (or EE fallback) for fake-grasp contact checks.
- **Actuator map + home q** from the spec MJCF (`actuator_trnid`, `qpos0`).

### End-to-end pick verification (no physics, no EGL)

`src/test/motion/test_arm_manip_profile.py` parametrizes every supported pick robot
(sourccey, xlerobot, rby1, innate_mars, franka_fr3) and verifies, per arm:

1. **Discovery resolves** — profile has joints, EE body, and a gripper contact body; every
   name exists in the MJCF.
2. **IK reaches the target** — `solve_position_ik_multiseed` drives the EE to a point a few
   cm in front of the arm's home EE, within tolerance.
3. **Fake-grasp contact** — an annotated gripper body lands within the object's contact
   radius. Grasp *physics* (attachment/force) is not simulated; "contact" means the gripper
   is at the object, so a mis-wired EE/joint/gripper map fails without a live robot or EGL.

```bash
# Unit (no sim)
uv run emet test src/test/perception/grasps/ src/test/motion/test_arm_manip_profile.py -q

# Oracle process (optional; smoke script spawns it)
uv run emet grasp-oracle --bind tcp://127.0.0.1:5558

# rby1 kinematic smoke
EMET_SIM_NAV_TELEPORT=1 MUJOCO_GL=egl \
  uv run python scripts/scripted_molmo_grasp_mp.py --start-sim \
  --sim configs/sim/molmospaces_ithor_train_0.yaml \
  --port-offset 194 --object bowl --cpu-only

# stretch teleport (same oracle)
EMET_SIM_NAV_TELEPORT=1 MUJOCO_GL=egl \
  uv run python scripts/scripted_molmo_grasp_mp.py --start-sim \
  --sim configs/sim/molmospaces_ithor_train_stretch_0.yaml \
  --port-offset 195 --object bowl --cpu-only
```

See also [molmospaces.md](molmospaces.md#mobile-manipulation-sim-teleport--kinematic).

---

## What we deliberately do not use for agent collision

- **CuRobo** on Molmo / GT MJCF — geometry would not match the real robot’s map.
- **MuJoCo contact** for agent pick/place success — kinematic attach is a sim proxy; teleport remains the OVMM oracle.

Collision awareness for planning should stay on the **agent voxel map** so sim and hardware share one world model.
