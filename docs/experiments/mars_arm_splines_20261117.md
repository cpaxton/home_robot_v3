# Mars random smooth arm distribution

The visible Herman elbow/wrist wave exposes a distribution gap: the analytic
controller tracks the admitted FK path (1.61 mm peak), while both retained
learned actors fail. Recent ordinary hand training changes only a few arm
coordinates by roughly 0.006–0.008 rad, often together with base motion. It does
not cover the folded posture and large elbow/wrist reversals in that wave.
This experiment changes data coverage first; the existing reward coefficients,
actor/history/preview contract and hard force/wheel/stop guards remain.

## Distribution and curriculum

`configs/benchmarks/mars_arm_splines_v1.json` declares 32 training and 16 audit
identities per level, with separate seed namespaces (460M and 470M). Four reset
anchors cover ready, raised, left reach and legal folded-right postures. These
are synthetic resets, not physical staging paths from Herman's captured pose.
Uniform reset jitter is 0.04/0.08/0.12 rad; joint-knot deviations are bounded by
0.12/0.30/0.55 rad for the three levels. Speed fractions are 25/40/60% of the
Mars profile limits. No added force/domain randomization in this first ablation.

Each case deterministically samples four joint knots, with a 50% return-to-start
choice. A natural cubic has continuous position, velocity and acceleration,
including nonzero tangents at interior knots. A rest-to-rest quintic phase
retimes it using conservative analytical chain-rule bounds for velocity,
acceleration and jerk. Joint extrema include between-knot overshoot. Private
FK supplies the runtime Cartesian hand path; joint witnesses never become
actor commands or observation fields. There is a one-second initial hold,
then motion, then a three-second settling deadline including two required
quiet seconds. The base target is stationary and fully masked; the base is a
free joint and policy wheel authority remains available. Gripper stays open.

Each path receives collision/support checks at 25 ms spacing and a 10 mm joint
margin. These are sampled geometry checks, not continuous collision or dynamic
feasibility certificates. Up to 64 deterministic sampling attempts are allowed;
every rejected proposal is recorded. Exhaustion fails the case and remains in
the report. Initial invalid anchors were replaced, rather than loosening gates.

| Family | Requested physical control ticks |
| --- | ---: |
| Existing general hand-active | 10% |
| Existing base-only | 10% |
| Existing goal changes | 10% |
| New random arm splines | 20% |
| Existing unloaded TAMP prefixes | 50% |

The outer general/TAMP ratio stays 50/50. This is Mars-only and unloaded;
loaded retention, grasp closure and complete pick/place are not trained here.
All audit identities stay out of PPO. The explicit level starts at zero;
advance only after development success reaches 85% with zero unsafe contact or
wheel lift. The audit set reports generalization and cannot select a level.

## Tools and intermediate validation

Generate the target gallery and teacher-bound report:

```bash
PYTHONPATH=src .venv/bin/python -m emet.rl.manipulation.arm_spline_benchmark \
  --level 0 --teacher --output logs/eval/arm-splines-v1-level0
```

`gallery.html` is a self-contained, orbitable 3D **target** display with all paths,
time scrubbing, hand orientation axes and per-case certificates/rejections. It
is not a video of policy behavior. `targets.json` and `report.json` stay ignored
under `logs/`; no room imagery or large recordings are committed.

The first four exploratory teacher trials (one per posture) passed: peak hand
position errors 0.91/1.97/0.71/1.20 mm, rotation errors
0.0037/0.0125/0.0040/0.0095 rad; no unsafe contact, wheel lift or joint-limit
pushing, all four stops confirmed. These preceded the shortened post-path
settling window and are exploratory, not a final qualification receipt.

Tests cover deterministic sampling, exact extrema, spline continuity and
kinematic bounds, FK state isolation, existing actor/critic compatibility,
teacher execution from each posture, finite deadlines, audit exclusion and
source/manifest/population binding. Full preflight checks all 48 level-zero
identities and preserves failures. PPO launch requires a fresh source-bound
regression receipt plus the exact teacher report; training cases need at least
85% teacher success and zero force/wheel safety failures. Audit failures remain
visible; they are not used to tune the curriculum.

## Bounded training and comparison

`emet.rl.manipulation.arm_spline_training` resumes the older retained actor,
critics and Adam moments, verifies weights against the actor bundle, and resets
physics/RNG explicitly. It starts with 4,096 critic-only decisions, then a
bounded 65,536-decision pilot total, ordinary exploration, decaying actor LR
1e-5 to approximately 2e-6, unchanged reward, and offline W&B. This is not a
reward-isolated comparison against runs with different optimizer settings;
it is the next bounded data-coverage experiment. No automatic extension,
checkpoint promotion or hardware execution. Report actual tick fractions and
completed episodes, not just planned fractions or total PPO decisions.

After the fixed checkpoint, compare teacher, initializer and final actor on the
same spline identities, with matching reset/reference hashes, and rerun ordinary,
goal-change, coupled and TAMP cases. Include the previously frozen elbow wave.
Report path and endpoint errors, base drift, joint-limit pushing, jerk, contact,
wheel unloading and stop confirmation. Improved spline performance alone does
not qualify the whole-body controller or Herman deployment.

## Launch record

Implementation is `614dfcd6` on draft PR #241. The updated source passes all
29 new tests (no skips), including four teacher executions with the final
shortened deadline and the existing 0.5 wheel-feedback gain. The preceding
56-test combined pass predates that final alignment; do not sum these as
independent coverage. Runtime package parity, Ruff and diff checks pass.
A local Chromium fixture test verifies gallery initialization, animated pixel
changes, pointer orbit, disabled playback for rejected cases and mobile width;
that fixture checks the UI, not robot geometry or policy performance.

Transient user service `emet-arm-spline-training` runs one numerical worker on
CPU 15, native threads one, CUDA hidden, 6 GB memory bound, offline W&B, and a
12-hour maximum. It uses the frozen `logs/runtimes/arm-splines-614dfcd6` snapshot,
so later worktree edits cannot change its source. It is currently in the full
regression phase; teacher preflight and PPO are queued, not yet completed.

- Pipeline state/logs: `logs/recovery/arm-spline-development-v1/`.
- Fresh regression receipt, only if passed:
  `logs/eval/arm-spline-regressions-v1/full.receipt.json`.
- Full level-zero teacher report/gallery: `logs/eval/arm-spline-preflight-v1/`.
- Bounded pilot, only after both checks pass: `logs/manipulation/arm-splines-v1/`.
- Matched initializer/final spline comparison: `logs/eval/arm-spline-paired-v1/`.

The pipeline stops on any failed prerequisite and does not silently increase
the training budget. Ordinary/coupled/goal-change/TAMP and frozen-wave evaluation
remain required after the spline comparison. No success, qualification or
hardware-readiness claim is made for the queued checkpoint.
