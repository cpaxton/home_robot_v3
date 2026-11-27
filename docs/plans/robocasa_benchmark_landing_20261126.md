# RoboCasa benchmark landing + regression gates

Date: 2026-11-26 (updated 2026-11-27)
Status: #264, #265, #267, and #266 are merged on `main` (`cd1bfcee`). PR #273 adds the
dynamic-base PandaOmron multi-step pick-place harness and passes privileged-teacher
qualification (`11/11` `env._check_success()` across Sink and Cabinet; see
[`docs/experiments/robocasa_pandaomron_pick_place_20261127.md`](../experiments/robocasa_pandaomron_pick_place_20261127.md)).
Learned-actor task-posture DAgger and observed/no-GT target perception remain open before
claiming a no-GT benchmark row.

This document separates **landing code** from **claiming a benchmark result**. The
new PandaOmron RoboCasa path can land as infrastructure before every benchmark row
is green, but no PR may claim a task pass until the task gate below has run.

## 1. Stack and landing order

| Order | PR | What it lands | Landing gate |
|---|---|---|---|
| 1 | #264 | open-fixture state + task-side placement base | merged (`fbf09184`) |
| 2 | #265 | PandaOmron bring-up, base-writer/gripper/placement fixes | Tier 0 + Tier 1 pass; explicit accept of the Panda placement gap |
| 3 | #267 | semantic/GT audit + failure provenance | Tier 0 pass after rebase |
| 4 | #266 | WBC contact observation + strict coverage + migration | Tier 0 + Tier 1 pass; no false hard-posture claim |

Rules:

- Do not merge #267 or #266 before #265. They are stacked deliberately.
- A green CI run is necessary but not sufficient for a benchmark claim.
- After each squash merge, rebase the next branch onto `main`; do not rely on a
  deleted base branch to preserve the diff.
- The hold decision for #265 is about the **placement benchmark**, not about
  whether the bring-up code is reviewable.

## 2. Benchmark inventory

| Benchmark | Kind | Current evidence | Gate |
|---|---|---|---|
| WBC clean multi-robot suite | old | `logs/eval/wbc_multirobot_20261126/manifest.json`; 5 robots, 50 episodes/subtask × 3 seeds | Full rerun before promotion. Smoke only for compatibility. |
| TAMP contract suite | old | `.github/workflows/tamp.yml`; 14 files | Every PR. All pass. |
| rby1 RoboCasa sink | old | teacher-path sink pick-place; `configs/sim/robocasa_pick_place_rby1_sink.yaml` | Run before claiming RoboCasa regression-free. |
| TAMP clutter GT+MCTS battery | old | `docs/experiments/tamp_clutter.md`; 24-case battery | Benchmark refresh, not every PR. |
| OVMM / HM-EQA paper rows | old | `docs/paper_benchmarks.md` | Only when shared agent/memory/TAMP paths change; declared refresh. |
| PandaOmron RoboCasa reach | new | `docs/experiments/wbc_robocasa_20261126.md`; 1.3 cm / 0.029 rad smoke; 20/20 loaded sweeps (1.13 cm joint, 0.87 cm Cartesian mean) | Adapter/reach gate; not task success. |
| PandaOmron RoboCasa pick-place | new | not yet green | Task gate below. |

“Old benchmarks” here means the existing WBC clean suite, the TAMP contract suite,
the verified rby1 RoboCasa sink, and the paper benchmark refresh. A change confined
to the new Panda adapter still needs Tier 0 and the WBC compatibility smoke.

## 3. Regression tiers

### Tier 0 — CPU contracts (every PR)

```bash
# Exact CI contract list
MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' \
  uv run pytest -q \
  src/test/controller/task/tamp/test_native_revalidation.py \
  src/test/controller/task/tamp/test_native_adapter.py \
  src/test/controller/task/tamp/test_native_approach.py \
  src/test/controller/task/tamp/test_native_release_budget.py \
  src/test/controller/task/tamp/test_observed_adapter.py \
  src/test/controller/task/tamp/test_transfer_provider.py \
  src/test/controller/task/tamp/test_mobile_case_config.py \
  src/test/controller/task/tamp/test_mobile_map_boundary.py \
  src/test/controller/task/tamp/test_mobile_regression_report.py \
  src/test/motion/test_release.py \
  src/test/motion/test_local_scene_store.py \
  src/test/motion/test_observed_admission.py \
  src/test/motion/test_observed_contact.py \
  src/test/motion/test_observed_support_contact.py

# WBC contract/training/coverage
uv run pytest -q src/test/motion/test_wbc*.py
```

Expected: all pass. No tolerance.

### Tier 1 — controller compatibility (stack changes)

```bash
# Layout v2 -> v3 migration and old WBC checkpoints
uv run python scripts/eval_wbc_policy.py \
  --robot pandaomron \
  --checkpoint logs/eval/wbc_multirobot_20261126/pandaomron/best.pt \
  --output /tmp/wbc_regression/pandaomron \
  --episodes 2 --seeds 30000 --device cpu --allow-missing-contract

uv run python scripts/eval_wbc_policy.py \
  --robot mars \
  --checkpoint logs/eval/wbc_multirobot_20261126/mars/best.pt \
  --output /tmp/wbc_regression/mars \
  --episodes 2 --seeds 30000 --device cpu --allow-missing-contract
```

- Smoke: no load error, no crash, finite metrics, `contact_state_rate == 1.0`.
  On the #266 head (2026-11-26): Panda 2 episodes seed 30000 = 0.833 aggregate;
  Mars = 1.000. These are compatibility checks, not performance gates.
- Full gate: 50 episodes/subtask × seeds 30000/40000/50000 for all five robots.
- Regression tolerance against `logs/eval/wbc_multirobot_20261126/manifest.json`:
  aggregate may not drop more than **0.02**, and no task may drop more than
  **0.05**, without an explicit explanation in the PR. A migrated v2 checkpoint
  sees zero contact inputs; it is not evidence for contact-aware control.
- New checkpoints embed the task contract; `--allow-missing-contract` is only
  for legacy layout-v2 artifacts. Without it, a mismatched target range, base
  goal probability, hold-base flag, or subtask set is refused.
- **Collision availability is measured only for Mars and direct Panda.** Stretch,
  Sourccey, RB-Y1, and the RoboCasa adapter report `collision_measured=False`
  rather than a false zero; do not gate `max_collision_rate` on those plants
  until a contact monitor is wired. Panda active cases have measured contact.

Also run the touched-controller pack:

```bash
uv run pytest -q \
  src/test/controller/test_kinematic_base_frame.py \
  src/test/robots/test_panda_omron.py \
  src/test/motion/test_placement.py \
  src/test/controller/task/test_task_search.py \
  src/test/controller/test_kinematic_lift_evidence.py \
  src/test/motion/test_placement_surfaces.py
```

### Tier 2 — RoboCasa old row + new reach (GPU/EGL, detached)

Use `emet jobs`; do not block an agent turn on RoboCasa.

```bash
# Old rby1 sink row. This is the recorded green command (job
# robocasa-sink-green2): all three ops complete, release verifies, pear ends in
# the sink, displacement 0.807 m. GT teacher is explicit and tagged.
NEED_MIB=12000 uv run emet jobs run --name robocasa-rby1-sink-regression --need-mib 12000 -- \
  env EMET_SIM_NAV_TELEPORT=1 EMET_GT_TEACHER=1 EMET_ROBOCASA_OBJ_GROUPS=pear \
  uv run python scripts/scripted_sim_pick_place.py --start-sim \
  --sim configs/sim/robocasa_pick_place_rby1_sink.yaml --manip-mode kinematic \
  --settle-s 2 --retries 3 --object pear --receptacle sink \
  --tool-calls-json '[{"name":"scene_tasks","arguments":{}},{"name":"plan_pick_place","arguments":{"task_ref":"$task_ref"}},{"name":"execute_pick_place_plan","arguments":{"plan_ref":"$plan_ref"}}]'

# New Panda reach adapter
NEED_MIB=12000 uv run emet jobs run --name robocasa-panda-reach --need-mib 12000 -- \
  uv run python scripts/smoke_wbc_robocasa_panda.py \
  --checkpoint logs/eval/wbc_multirobot_20261126/pandaomron/best.pt \
  --backoff 0.45 --layout 1 --style 1

# Loaded reach sweep (joint targets or Cartesian offsets)
NEED_MIB=12000 uv run emet jobs run --name robocasa-panda-sweep --need-mib 12000 -- \
  uv run python scripts/sweep_wbc_robocasa_panda.py \
  --checkpoint logs/eval/wbc_multirobot_20261126/pandaomron/best.pt \
  --target-mode offset --episodes 20 --output /tmp/wbc_robocasa_sweep
```

Acceptance:

- Old rby1 sink still reports task success, measured release, and the expected
  object displacement. A code-only smoke is not a benchmark pass.
  The recorded command uses `EMET_GT_TEACHER=1` for task grounding: label it a
  **teacher-path mechanical success**, not a no-GT benchmark pass. A no-GT row
  requires the same `scene_tasks → plan → execute` path with the teacher off.
- Panda reach tracks the target within the documented tolerance and reports no
  final-step kitchen contact. This is a tracking result, not pick-place success.

### Tier 3 — benchmark refresh

Run the TAMP clutter battery, OVMM/HM-EQA rows, and the frozen-main existing
benchmark refresh only when the PR touches their shared paths or when a benchmark
claim is being updated. Preserve failures and pending cases; do not pool a smoke
with a paper row.

## 4. New RoboCasa benchmark path

### Phase A — reach adapter (done)

`RoboCasaPandaBackend` drives the native PandaOmron `PickPlaceCounterToCabinet`
scene and tracks a free-space target. Evidence: 1.3 cm / 0.029 rad smoke; a
20-episode loaded joint-target sweep is 20/20 at 1.13 cm mean / 1.67 cm p95,
and a 20-episode Cartesian-offset sweep is 20/20 at 0.87 cm mean / 1.54 cm p95.
This is reach tracking, not task success.

The backend now also has a privileged joint-PD teacher with gravity bias and a
Cartesian PD fallback, plus `scripts/sweep_wbc_robocasa_panda.py` for loaded
scene sweeps and NPZ teacher labels. Use those labels for BC/DAgger only if a
farther object/cabinet target exits the reach tolerance.

### Phase B — task-target bridge

- Feed object/receptacle/grasp targets from the task/TAMP layer into
  `set_ee_target`; the Cartesian PD teacher already supports arbitrary world
  poses.
- Feed base goals for counter approach and retreat.
- Keep the actor on measured observations; GT task targets may be teacher-side
  only and must be tagged.
- Gate: target reaches the pre-grasp pose without contact in both
  `PickPlaceCounterToSink` and `PickPlaceCounterToCabinet`; run a farther-target
  sweep and fine-tune from the saved NPZ only if the tail crosses 2 cm.

### Phase C — carry, contact, and place

- Use the measured wrist wrench/finger loads from layout v4 during carry.
- Require `contact_state_rate == 1.0` for carry/contact cases.
- Gate: lift, transport, release, and support are measured; no GT body IDs in the
  actor path.

### Phase D — merged-scene adapters

- Stretch, Sourccey, and RB-Y1 RoboCasa configs use the merged-scene path, not a
  native Robosuite robot.
- Reuse the direct-write adapter against the merged MJCF, or add a merged backend
  that exposes the same contract.
- Gate: each robot's old RoboCasa row still passes before its new row is claimed.

### Phase E — scoring and no-GT audit

- Task success must be measured with object displacement/support/release, not
  just reach tracking.
- The actor receives no object IDs, GT poses, or privileged contact labels.
- Teacher/critic GT is tagged `privileged` and excluded from assessment.
- Gate: one held-out sink and one held-out cabinet run, with the teacher off
  where no-GT assessment is claimed.

## 5. Risks and explicit non-claims

- **#265 Panda placement is not green.** Payload/robot/self-collision while
  carrying remains. The bring-up may land as infrastructure; it is not a benchmark
  pass.
- **Migrated v2 checkpoints have zero contact inputs.** They are compatibility
  baselines, not contact-aware policies.
- **Unmeasured collision is not zero collision.** Stretch/Sourccey/RB-Y1 and the
  RoboCasa adapter currently expose `collision_measured=False`; their reports
  must fail a collision gate instead of passing it.
- **The hard-posture train split is empty.** Records 014–018 were successful
  grasp attempts from place failures; `hard_posture` stays `planned` until a
  valid `reached=False` split is regenerated.
- **Tier 1 smoke is not a performance gate.** Two episodes catch load/crash
  regressions; the 50×3 comparison is the actual old-benchmark gate.
- **rby1 sink is the old benchmark anchor, but the recorded run is teacher-path.**
  Do not replace it with the Panda reach smoke, and do not report it as a no-GT
  result. `EMET_GT_TEACHER=1` is task grounding; the actor/assessment path still
  needs the teacher-off run.
- **The local Sourccey `tamp_approach="side"` edit is not part of the PR stack.**
  It makes the existing front-approach test fail and must be reviewed separately.

## 6. TODO checklist

- [x] Land #264.
- [ ] Land #265 after Tier 0 + Tier 1 and an explicit placement-gap decision.
- [ ] Rebase and land #267 after #265; rerun Tier 0.
- [ ] Rebase and land #266 after #267; rerun Tier 0 + Tier 1.
- [ ] Run the full WBC clean suite (50×3, five robots) after #266 and record the
  comparison to `logs/eval/wbc_multirobot_20261126/manifest.json`.
- [ ] Run the old rby1 sink row via `emet jobs` and record success/displacement.
- [ ] Implement the Panda task-target bridge and base goals.
- [ ] Add the merged-scene adapters for Stretch/Sourccey/RB-Y1.
- [ ] Run the new sink/cabinet task gate with the teacher off where claimed.
- [ ] Regenerate a valid `reached=False` hard-posture train/test split.
