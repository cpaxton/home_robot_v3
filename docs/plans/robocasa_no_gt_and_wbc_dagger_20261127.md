# Plan: RoboCasa PandaOmron No-GT Perception, Proprioceptive Retention Gates, and Task-Posture WBC DAgger

Date: 2026-11-27
Branch / PR: `feat/omron-benchmark-harness` (PR #273 → `main`)
Baseline qualification: [`docs/experiments/robocasa_pandaomron_pick_place_20261127.md`](../experiments/robocasa_pandaomron_pick_place_20261127.md)

---

## 1. Ground-Truth (GT) Provenance Audit

Every simulator read in the current RoboCasa PandaOmron stack falls into one of four categories. The low-level WBC actor contract is **already 100% GT-free**; Scene GT usage is confined to the outer task-harness planner, execution gates in both controller modes in `scripts/run_wbc_robocasa_pick_place.py` and diagnostic helpers in `src/emet/rl/manipulation/wbc_robocasa_panda.py`.

### 1.1 Zero-GT Actor & Control Contract (Keep as-is)

| Component | File / Method | Inputs Used | GT Status |
|---|---|---|---|
| WBC observation & encoding | `wbc_env.py:encode_observation`, `wbc_robocasa_panda.py:_observation` | Reset-anchored base pose/twist, joint `q/qd/tau`, hand FK pose, active EE/base target, gripper `q/qd`, robot tactile `MuJoCoContactSensor` | **0 GT** (proprioceptive only) |
| Dynamic base & torso actuation | `wbc_robocasa_panda.py:step`, `_body_twist_to_joint_vel`, `reanchor_origin`, `hold_ee_in_base_frame` | Robot joint axes (`xaxis`), yaw anchor (`xanchor`), robot `dof_frictionloss`, robot FK | **0 GT** (robot URDF/MJCF parameters + encoders) |
| Wrist-to-EEF rigid offset & arm IK | `run_wbc_robocasa_pick_place.py:T_hand_eef`, `wbc_robocasa_panda.py:solve_ee_ik` | Robot `robot0_right_hand` and `gripper0_right_eef` FK ($R_z(-\pi/2), (0,0,0.097\,\text{m})$) | **0 GT** (fixed robot kinematics; `solve_ee_ik` runs with `--teacher` and for expert labeling with `--save-rollout`) |
| Task language prompt | `env.get_ep_meta()["lang"]` | Natural-language instruction (e.g. *"Pick the pear from the counter and place it in the sink."*) | **0 GT** (standard benchmark task specification) |

### 1.2 Privileged Execution Gates → Proprioceptive Replacements (Phase 1)

| Privileged Call Site | Current GT Read | Why Used | Sensor-Only / Proprioceptive Replacement |
|---|---|---|---|
| `wbc_robocasa_panda.py:600` `finger_target_forces(target_body)` & `:566` `contact_breakdown(target_body)` (called at `run_wbc_robocasa_pick_place.py:355, 444, 821, 920, 980`) | Filters `data.contact` by `model.body(target_body).id` | Distinguishes finger-on-object load from finger-on-counter load | `MuJoCoContactSensor.read(data).finger_force` (per-finger tactile load `(left_N, right_N)`, zero body IDs) **+** encoder jaw-stall gap `q_left - q_right > 0.008 m` under closed command `(0.0, 0.0)` |
| `run_wbc_robocasa_pick_place.py:845, 857` `lift_start / lift_end = _body_pose(env, object_body)` | Reads `obj_main` Z rise (`>= 0.01 m`) over 3 cm lift | Verifies object lifted off counter rather than slipping | **Proprioceptive free-air jaw-stall + tactile hold** after lifting `+3 cm` (when an object slips under `ctrl=(0, 0)`, the jaws close immediately to `< 0.002 m` and tactile load drops to `0 N`) |
| `wbc_robocasa_panda.py:419` `teleport_base_target()` (`--teleport-base`) | Solves planar `qpos` directly | Optional unloaded teacher shortcut | Already `False` by default and unused in all 11/11 qualified runs; forbid when `--no-gt` is set |

### 1.3 Privileged Task-Target Perception → Observed RGB-D Provider (Phase 3)

| Privileged Call Site | Current GT Read | Why Used | Observed (No-GT) Replacement |
|---|---|---|---|
| `run_wbc_robocasa_pick_place.py:623, 679` `_body_pose(env, args.object_body)` | `sim.data.get_body_xpos("obj_main")` | Base approach target & initial object centroid | Multi-camera RGB-D (`robot0_agentview_left/right`, `robot0_eye_in_hand`) open-vocabulary grounding + depth back-projection (Dynagraph / `observed_capture.py`) |
| `run_wbc_robocasa_pick_place.py:116, 681` `_body_geometry(env, args.object_body)` | Collidable mesh vertices & triangle edges of `obj_main` | Feeds `_plan_topdown_grasp` (`_slice_2d` narrow-axis yaw & counter/palm-safe height) | Segmented 3D object surface point cloud (from wrist + agentview depth + camera extrinsics) passed directly into `_plan_topdown_grasp` (point-cloud path requires zero body IDs) |
| `run_wbc_robocasa_pick_place.py:271, 933` `_receptacle_place_target(env, receptacle_body)` | `fixture.get_int_sites(relative=False)` / `_body_pose(env, receptacle_body)` | Sink basin / open cabinet shelf interior bounds & front normal | Observed receptacle bounding box & approach normal from RGB-D semantic segmentation + Dynagraph fixture nodes |

### 1.4 Post-Episode Benchmark Scoring Only (Keep Isolated in Eval Scorer)

- `run_wbc_robocasa_pick_place.py:1103–1106`: `env._check_success()`, `displacement_m`, and `receptacle_xy_error_m` run **only after Stage 5 (`retreat`) completes** to score the episode. They never enter planning or control.

---

## 2. Staged Execution Plan

### Phase 0 — Land PR #273 Infrastructure (`feat/omron-benchmark-harness`)
- **Goal:** Merge the dynamic-base teacher, geometry-aware grasp planner, `--json-out` / `--save-rollout` exporters, unit tests, docs, and CoRL Appendix `\ref{app:robocasa_wbc_pick_place}` onto `main`.
- **Gate:** Tier 0 CPU suite (`test_wbc*.py` 133/133 + TAMP CI check) green; `scripts/check_generated_artifacts.py` and `paper/build.sh` clean.

### Phase 1 — Proprioceptive Retention Gate & Hard `--no-gt` Runtime Guard
- **Goal:** Remove all GT body-ID reads from execution gating (`close`, `lift`, `retract`, `transport`) when running in actor / `--no-gt` mode, and enforce a fail-closed guard that raises if any `privileged_*` helper is called under `--no-gt`.
- **Changes:**
  1. Add `RoboCasaPandaBackend.proprioceptive_grasp_state()` returning `(left_finger_N, right_finger_N, jaw_gap_m)` from `MuJoCoContactSensor.read(data).finger_force` and `qpos[_grip_q]` (zero target body IDs).
  2. Gate `close`, `lift_check` (post-3 cm free-air lift), and `transport_check` on `left_N > 1.0`, `right_N > 1.0`, and `0.006 <= jaw_gap_m <= 0.077`.
  3. Add `--no-gt` flag + `RuntimeError` guard on `contact_breakdown`, `finger_target_forces`, `teleport_base_target`, `_body_pose`, `_body_geometry`, and `_receptacle_place_target`.
- **Gate:** Re-run the 11 Sink + Cabinet teacher episodes using **proprioceptive gating only** and confirm `11 / 11` match the GT-gated outcome with zero false positives on deliberate air-grasp / slip negative controls.

### Phase 2 — Task-Posture RL-WBC Policy Fine-Tuning (BC + DAgger)
- **Goal:** Close the `2.19 rad` top-down wrist-posture OOD gap on `pandaomron_robust/best.pt` so the **learned neural actor** (`--teacher` off) executes the full dynamic pick-place sequence.
- **Changes:**
  1. **Direct-plant curriculum expansion (`wbc_pandaomron.py`):** Add a task-posture target mode (`topdown` $R_z(-\pi/2)$ across $\psi \in [\psi_b, \psi_b + \pi]$, $55^\circ$ cabinet-entry pitch, vertical descents, and $0.02\text{--}0.34\,\text{m}$ torso heights) alongside existing `_READY_Q ± 0.15 rad` targets so DAgger retains clean-suite accuracy.
  2. **RoboCasa teacher rollout collection:** Run `run_wbc_robocasa_pick_place.py --teacher --save-rollout ...` across 40–60 Sink + Cabinet seeds via `emet jobs` (`--need-mib 6000`).
  3. **DAgger fine-tune (`wbc_dagger.py`):** Initialize from `pandaomron_robust/best.pt`, train on mixed direct-plant + RoboCasa task rollouts + on-policy actor-visited states relabeled by `backend.teacher_action()`.
- **Gate:**
  - Existing 5-robot / `pandaomron_robust` clean WBC benchmark suite does not regress beyond tolerance (`-0.02` aggregate).
  - Learned policy (`--teacher` off, privileged task targets labeled `policy_with_gt_targets`) achieves `>= 80%` `env._check_success()` on held-out Sink & Cabinet seeds.

### Phase 3 — Observed RGB-D Target Provider & Unassisted No-GT Benchmark
- **Goal:** Replace privileged object & receptacle pose/mesh queries with camera observations (`use_camera_obs=True`) for true unassisted evaluation (`policy_observed_no_gt`).
- **Changes:**
  1. Enable `robot0_agentview_left`, `robot0_agentview_right`, and `robot0_eye_in_hand` RGB-D + camera extrinsics in `run_wbc_robocasa_pick_place.py` when `--target-source observed`.
  2. Localize target object & receptacle from `lang` via the shared VLM/detector grounding pipeline + depth back-projection into world/robot-frame 3D point clouds; run an active wrist-camera look-down refinement after `base_approach` to get a dense top-down point cloud of the target object before calling `_plan_topdown_grasp(points, edges=empty, ...)`.
  3. Run with `--no-gt` active so only `env.get_ep_meta()["lang"]` is read before `retreat`, and `env._check_success()` runs strictly post-episode.
- **Gate:** Report separate, provenance-tagged rows in `paper/data/whole_body_control/` and `docs/experiments/` for:
  1. `teacher_gt_targets` (ceiling),
  2. `policy_gt_targets` (control-only isolation), and
  3. `policy_observed_no_gt` (full unassisted mobile manipulation benchmark).
