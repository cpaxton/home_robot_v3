# PandaOmron RoboCasa multi-step mobile pick-place and grasp retention

Date: 2026-11-27
Source revision: `7a2fbeed` (PR #273, `feat/omron-benchmark-harness`)
Structured ledger: [`paper/data/whole_body_control/robocasa_pandaomron_pick_place_20261127.json`](../../paper/data/whole_body_control/robocasa_pandaomron_pick_place_20261127.json)

Scope: end-to-end physical mobile-manipulation qualification above the PandaOmron WBC contract in RoboCasa (`PickPlaceCounterToSink` and `PickPlaceCounterToCabinet`).

> **Provenance boundary:** Task-level object and receptacle geometry in `--teacher` runs come from simulator metadata (`privileged_gt_metadata`), while all robot motion executes in continuous native MuJoCo contact physics with **zero `qpos` teleportation** and **zero object welds/latches**. Do not present teacher-path numbers as observed/no-GT actor benchmark scores.

---

## 1. Root causes isolated and resolved

Earlier iterations tracked end-effector waypoints to `<1.5 cm` yet failed to capture or retain objects across lift and base transport. Four targeted diagnostic probes (`scripts/probe_wbc_grasp_retention.py`, `scripts/probe_wbc_grasp_frame.py`, `scripts/probe_wbc_grasp_ik.py`, `scripts/probe_wbc_grasp_select.py`) isolated five distinct physical and kinematic causes:

1. **Wrist-to-EEF frame offset (`robot0_right_hand` vs `gripper0_right_eef`).**
   `top_down_grasp_T` specifies `gripper0_right_eef`, whereas the WBC contract commands `robot0_right_hand`. In Robosuite's composed `PandaOmron` MJCF, `T_hand_eef` has fixed rotation $R_z(-\pi/2)$ and translation $(0, 0, 0.097\,\text{m})$. Commanding `top_down_grasp_T` directly to `robot0_right_hand` left the gripper $90^\circ$ misaligned and shoved light objects (`30 g` cans) 10–13 cm before closing.
2. **Wrist joint-limit saturation vs base-relative grasp yaw + warm-started IK.**
   With world-fixed grasp yaw, damped-least-squares IK converged with `robot0_joint7` pinned at `+2.897 rad` (`0.0 rad` joint margin), stalling Jacobian-transpose Cartesian control 25–33 cm short. Centering candidate grasp yaws around $\psi_{\text{base}} + \pi/2$ matches `_READY_Q[6] = +0.785 rad` with `0.85 rad` joint margin, and adding warm-started damped-LS IK (`RoboCasaPandaBackend.solve_ee_ik`) gives `<1 mm` teacher tracking across the full $180^\circ$ undirected yaw span.
3. **Finger-closing axis (`EEF X`, not `EEF Y`) + exact `panda_gripper.xml` clearances.**
   In `panda_gripper.xml`, pad separation in `gripper0_right_eef` lies along **EEF X** (`[-1, 0, 0]`, world $[\cos\psi, \sin\psi, 0]$), with **79.0 mm** clear inner-pad opening at `q = [0.04, -0.04]`, palm collision (`hand_collision`) **31.0 mm** above EEF, and fingertip collision **9.2 mm** below EEF. Vectorized triangle-edge plane slicing (`_slice_2d`) across collidable geoms handles sparse 2-ring cylinder meshes (`can`) and 8-corner boxes (`bar_soap`), choosing the deepest counter/palm-safe height (`z_lo`) and the narrowest horizontal pinch axis.
4. **Why carried objects dropped during base transport: discrete `qpos` stepping vs 500 Hz physics integration.**
   Stepping `data.qpos[_base_q]` kinematically every `50 ms` (`apply_base_qpos`) teleported the `8 mm`-thick rubber pads `5–15 mm` in zero time while the free-joint object remained stationary—opening an air gap on one pad and burying the opposite pad into the mesh (`185 N` Baumgarte impulse). Driving the base dynamically through `sim.step()` at `500 Hz` (`<0.45 mm` per `2 ms` substep) while locking the retracted arm target in the moving base frame (`hold_ee_in_base_frame(True)`) preserves `13–20 N` bilateral pinch force throughout 1 m carries.
5. **Omron holonomic base joint order & Coulomb stiction (`omron_mobile_base.xml`).**
   In `omron_mobile_base.xml`, `joint_mobile_forward` and `joint_mobile_side` precede `joint_mobile_yaw` (anchored at `(-0.21, 0, 0)`), so they slide in the spawn frame rather than the rotated chassis heading. In addition, the planar joints declare `frictionloss="250"` against `<velocity kv="1000..1500">` (a `0.25 m/s` deadzone), and `joint_torso_height` declares `frictionloss="1000"`. Mapping chassis twist via `data.xaxis` + `data.xanchor` and adding smooth `dof_frictionloss` feedforward eliminates both deadzones, enabling accurate non-teleport base driving and full `0.02–0.34 m` torso coordination under open upper-cabinet doors.

---

## 2. Privileged teacher qualification (`PickPlaceCounterToSink` & `PickPlaceCounterToCabinet`)

All 11 episodes below run with **100% dynamic base approach and dynamic base transport** (`--teleport-base` off, zero object welds), gated at execution time by bilateral target finger force (`>1 N` per finger for `0.9 s`), a `3 cm` verification lift (`>=1 cm` measured object rise), post-transport retention force, and RoboCasa's official `env._check_success()` (`OU.obj_inside_of` and `OU.gripper_obj_far > 0.25 m`).

| Task | `--obj-groups` (sampled object) | Pinch width | Close force (L / R) | Lift rise | Post-transport (L / R) | Displacement | Receptacle XY err | `env._check_success()` |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `PickPlaceCounterToSink` | `can` (`seed=0`, can) | 5.73 cm | 20.0 / 20.0 N | +3.24 cm | 20.0 / 20.0 N | 1.022 m | 3.19 cm | **True** |
| `PickPlaceCounterToSink` | `pear` (`seed=0`, pear) | 7.10 cm | 19.4 / 19.6 N | +3.24 cm | 18.2 / 18.6 N | 1.038 m | 6.26 cm | **True** |
| `PickPlaceCounterToSink` | `bar_soap` (`seed=0`, bar soap) | 6.07 cm | 20.2 / 19.7 N | +3.26 cm | 19.7 / 19.7 N | 0.995 m | **0.12 cm** | **True** |
| `PickPlaceCounterToSink` | `egg` (`seed=0`, egg) | 4.76 cm | 19.9 / 20.0 N | +3.24 cm | 19.8 / 19.9 N | 1.004 m | 1.99 cm | **True** |
| `PickPlaceCounterToSink` | `all` (`seed=0`, can) | 6.03 cm | 19.6 / 19.6 N | +3.23 cm | 20.1 / 19.9 N | 0.402 m | 7.63 cm | **True** |
| `PickPlaceCounterToSink` | `all` (`seed=1`, 25.5 cm spray) | 3.70 cm | 19.3 / 19.6 N | +3.18 cm | 18.1 / 18.1 N | 0.350 m | 3.64 cm | **True** |
| `PickPlaceCounterToSink` | `all` (`seed=2`, shaker) | 3.51 cm | 17.0 / 17.0 N | +3.24 cm | 16.5 / 16.5 N | 0.426 m | **0.08 cm** | **True** |
| `PickPlaceCounterToCabinet` | `can` (`seed=0`, can) | 6.04 cm | 19.8 / 20.0 N | +3.25 cm | 19.9 / 19.9 N | 0.540 m | **0.95 cm** | **True** |
| `PickPlaceCounterToCabinet` | `pear` (`seed=0`, pear) | 6.17 cm | 19.4 / 19.5 N | +3.14 cm | 18.6 / 17.4 N | 0.538 m | 6.45 cm | **True** |
| `PickPlaceCounterToCabinet` | `egg` (`seed=0`, egg) | 4.71 cm | 19.4 / 19.7 N | +3.21 cm | 18.8 / 18.2 N | 0.542 m | 3.04 cm | **True** |
| `PickPlaceCounterToCabinet` | `all` (`seed=0`, sugar cube) | 2.84 cm | 13.3 / 16.5 N | +3.21 cm | 13.2 / 13.2 N | 0.559 m | 1.56 cm | **True** |

Summary: **7 / 7** on `PickPlaceCounterToSink` and **4 / 4** on `PickPlaceCounterToCabinet` (**11 / 11 overall**).

---

## 3. Zero-shot learned RL-WBC checkpoint probe (`pandaomron_robust/best.pt`)

Running the promoted direct-plant reach policy (`logs/eval/wbc_multirobot_20261126/pandaomron_robust/best.pt`) without `--teacher` on `PickPlaceCounterToSink` (`can`, `seed=0`):

| Phase | Measured outcome | Interpretation |
|---|---|---|
| Dynamic `base_approach` | `xy_error = 1.75 cm`, `yaw_error = 0.087 rad` (237 steps) | Policy transfers zero-shot to RoboCasa dynamic base navigation once `xaxis` twist mapping and stiction feedforward are active. |
| Stationary base hold (`reanchor_origin` + `hold_base`) | Base drift `[+0.5 cm, -1.1 cm, -0.018 rad]` | Re-anchoring `observation.base_pose` to `(0, 0, 0)` after base arrival eliminates stationary base spin. |
| Top-down `pregrasp` | `pos_err = 29.6 cm`, `quat_err = 2.19 rad` → fails `close_gate` | Expected distribution gap: `PandaOmronWbcBackend` trained only on `±0.15 rad` joint offsets around `_READY_Q`, never seeing $90^\circ$ top-down grasp postures or loaded carries. |

---

## 4. Reproducible commands & rollout export

Run via `emet jobs` (`--need-mib 6000` on 8 GiB GPUs) and record structured JSON / `.npz` rollouts:

```bash
uv run emet jobs run --name omron-sink-teacher --need-mib 6000 -- \
  env MUJOCO_GL=egl uv run python scripts/run_wbc_robocasa_pick_place.py \
    --teacher --env-name PickPlaceCounterToSink --obj-groups all --seed 0 \
    --json-out ~/runs/emet/omron_pick_place/sink_s0.json \
    --save-rollout ~/runs/emet/omron_pick_place/sink_s0_rollout.npz

uv run emet jobs run --name omron-cab-teacher --need-mib 6000 -- \
  env MUJOCO_GL=egl uv run python scripts/run_wbc_robocasa_pick_place.py \
    --teacher --env-name PickPlaceCounterToCabinet --obj-groups all --seed 0 \
    --receptacle-body cab_main \
    --json-out ~/runs/emet/omron_pick_place/cab_s0.json \
    --save-rollout ~/runs/emet/omron_pick_place/cab_s0_rollout.npz
```

---

## 5. Next steps

Detailed execution plan and line-by-line GT provenance audit: [`docs/plans/robocasa_no_gt_and_wbc_dagger_20261127.md`](../plans/robocasa_no_gt_and_wbc_dagger_20261127.md).

1. **Task-distribution BC / DAgger fine-tuning for RL-WBC:** Collect multi-seed `(observations, teacher_actions)` `.npz` rollouts from `run_wbc_robocasa_pick_place.py --teacher --save-rollout ...` (and expand `PandaOmronWbcBackend` target sampling to cover top-down and $55^\circ$ cabinet postures), then DAgger-finetune `pandaomron_robust` while gating no-regression on the clean 5-robot WBC suite.
2. **Observed (no-GT) target provider + strict no-GT gate:** Feed segmented RGB-D point clouds (`robot0_eye_in_hand` + `robot0_agentview_*` via Dynagraph / depth back-projection) into `_plan_topdown_grasp` (which operates purely on 3D surface points/segments) and enforce a hard no-GT guard on actor/benchmark evaluation paths.
3. **Broader seed/layout & hollow-rim object coverage:** Sweep 20–50 seeds across RoboCasa kitchen layouts/styles, add single-wall rim pinching for wide hollow bowls/pans (`>7.9 cm` outer diameter), and wire articulated door/drawer opening (`open_fixture` / `close_fixture`).
