# Whole-body training mixture

Updated 2026-11-17. This is the maintained use-case ledger for current Mars WBC
training. [ACTIVE_WORK](ACTIVE_WORK.md) owns job/branch status; experiment reports
own results. Update this page whenever sampling weights, case pools, reset
conditions or training stages change. Planned allocations are not running ones.

## Completed ordinary-run mixture

Shares are **physical 20 Hz control ticks**, not episode counts, PPO decisions
or a random 50/50 draw at every step. `ControlStepMixture` keeps episodes intact
and schedules whichever family owes ticks. Short TAMP episodes therefore occur
more often. Some executor holds bundle multiple control ticks into one PPO
decision; their rewards/continuation use the physical-tick discount.

| Use case | Share of all training ticks | What is actually trained |
| --- | ---: | --- |
| General hand-active tracking | 40% | Eight nearby FK-witness trajectories: seven coupled hand/base and one free-base hand case; masks include full/XY constraints. |
| General base-only tracking | 10% | Four cases: forward, reverse and both yaw signs; gripper open, arm action inactive. |
| Public-tool TAMP prefixes | 50% | Four training layouts/seeds, left/right and scene objects of 20/50 g. Approach/pregrasp to an unloaded grasp-pose boundary; stop before task-owned closure. |
| Dedicated general goal-transition episodes | 0% | The general pool has no scheduled two-stage updates. TAMP contains its normal executor stages/settling, which is different coverage. |
| Loaded grasp/lift/carry/place or learned grip closure | 0% | Not part of these runs. Objects in the TAMP scene do not mean a held payload. |
| Deliberate force/disturbance windows | 0% | Incidental contact is monitored, but curated perturbations are not sampled. |
| Stretch | 0% | Existing implementation/tests remain; these training runs are Mars-only. |

This is 50% general / 50% TAMP, with 80% hand-active / 20% base-only **inside
general**. It does not mean 40% isolated arm control: hand-active cases can use
the base. Current general resets use seed-dependent nearby arm configurations,
zero added domain randomization and no initial-heading randomization. Paths
are speed/acceleration/jerk-bounded; real-model limits and independent guards
remain active. TAMP fixture solver/model settings differ from the general plant;
source/model identities and scope must stay explicit.

The completed ordinary-run custom runner is authoritative:
`paper/data/whole_body_control/long_finetune_v1/run.py`, with its adjacent
`protocol.json` and `cases.json`. The older generic CLI development config's
30/50/20 base/arm/coupled stage weights do **not** describe this runner's inner
80/20 hand/base mixture. Its four TAMP training and two audit identities still
come from `configs/benchmarks/mars_whole_body_v2_development.json`.

## Actual exposure in the completed longer run

65,536 additional PPO decisions produce **77,452 physical control ticks**:

| Use case | Requested | Actual | Control ticks | Episodes started |
| --- | ---: | ---: | ---: | ---: |
| General hand-active | 40% | 40.02% | 31,000 | 62 |
| General base-only | 10% | 9.93% | 7,692 | 17 |
| TAMP prefixes | 50% | 50.04% | 38,760 | 190 |

There are 78 completed general episodes and 190 completed TAMP episodes. The
79 general starts include an unfinished final episode. The preceding short
control had only 20 completed general episodes: two per hand trajectory and
one per base case. The longer actor's lineage is 16,384 + 65,536 = **81,920
decisions**, including the original critic warmup. Do not report the additional
65,536 as a fresh initialization or an exactly resumed physical state.

Recompute a live/completed ledger without a simulator:

```bash
.venv/bin/python scripts/report_wbc_training_mix.py \
  --run logs/manipulation/long-finetune-v1/continued/run.json
```

Use `--format json` for an auditable snapshot. The reporter checks tick totals
and nested fractions; counters labeled episodes are starts. Exact completed
outcomes are in each run's `episodes.jsonl`, not inferred from the scheduler.

## What the fresh cases show

All 24 fresh cases execute and match initial states/configuration/reference
hashes across controllers. They never enter the longer run's PPO data.

| Fresh case group | Cases | Ordinary source | Longer-trained | Teacher |
| --- | ---: | ---: | ---: | ---: |
| Coupled hand/base, one stage | 8 | 2/8 | 4/8 | 6/8 |
| Hand target, base free | 8 | 7/8 | 8/8 | 8/8 |
| Base only | 4 | 2/4 | 4/4 | 1/4 |
| Two-stage goal changes | 4 | 1/4 | 0/4 | 2/4 |
| Total | 24 | 12/24 | 16/24 | 17/24 |

Safety-flagged episodes fall 4 → 1, including wheel-lift events 2 → 0; all 72
teacher/source/final stops confirm. Peak linear/yaw jerk nevertheless increases
14.88/65.57 → 28.68/78.54 in respective SI units. On reused development cases,
20/24 → 19/24 and existing coupled stays 7/10. This is a scoped generalization
improvement, not a qualified safe controller.

Transitions are **one important gap**, not the only issue: four single-stage
coupled cases still fail, one fresh contact stop remains, and the teacher itself
fails two of the four transition cases. The second commands arrive before the
first goal's required quiet hold finishes, with nominally anchored C2 references;
they are not discontinuous, arbitrary unreachable targets. Four cases do not
establish a general transition success rate.

## Completed goal-change experiment

Preserve the outer 50/50 general/TAMP balance. Within general, move 20 percentage
points from ordinary hand tracking into analogous goal-change cases:

| Use case | Completed ordinary share | Goal-change run target |
| --- | ---: | ---: |
| Regular hand-active tracking | 40% | 30% |
| Base-only tracking | 10% | 10% |
| General two-stage goal changes | 0% | 10% |
| TAMP unloaded prefixes | 50% | 50% |

The completed curriculum producer is `paper/data/whole_body_control/goal_change_curriculum_v1/run.py`,
with adjacent protocol/cases/curriculum manifests. The older ordinary runner above
is historical. Read its live ledger with `--run logs/manipulation/goal-change-curriculum-v1/curriculum/run.json`.

The new curriculum runner uses inner general weights of 60/20/20 regular-hand/base/goal-change.
Thus goal changes use only 10% of all training; 90% retains the existing
families. Generate new training seeds/paths analogous to these failures, rather
than copying the 24 confirmation identities into PPO. Keep precise hand/coarse
base contracts, reward, native limits, history and force/wheel guards fixed.
Both nominal-stage and live-state/fault invariants have implementation/tests.
The full frozen-source gate passes 1,604 tests and teacher/source admission is
complete (13/16 vs 9/16, all 32 stops confirmed). The fixed 131,072-decision run completes. Actual shares are
29.92/10.22/9.99/49.87% (46,500/15,884/15,530/77,509 ticks). Goal changes
complete four stochastic episodes per train case: 31 timeouts and one wheel-lift
stop, zero successes. Final deterministic goals regress 9/16 → 6/16 despite
regular tracking improving 19/24 → 22/24. No promotion or automatic extension. See the
[curriculum protocol](../experiments/goal_change_curriculum_20261117.md).

Subsequent additions should be explicit new rows, not silently folded into
"TAMP": larger reachable posture/reset coverage, mid-reference motion changes,
bounded disturbances, and then grasp/lift/loaded carry/place once their simulator
initialization/execution is reliable. Keep a core of ordinary tracking. Any new
weights and randomization curriculum must be recorded before training starts.

## Random arm-spline family (implemented; validation before training)

The next Mars mixture is **10/10/10/20/50** regular-hand/base/goal-change/
random-arm-spline/unloaded-TAMP control ticks. It keeps the outer 50/50 balance.
It is not the completed goal-change run above. The new family samples four
nearby legal reset postures, cubic joint geometry mapped through private FK,
reversals/returns, speed/acceleration/jerk-bounded timing and quiet holds. The
base target stays stationary without welding the base or masking wheel actions.

The manifest declares 32 train / 16 audit identities at each of three amplitude
and speed levels. Start at the smallest (0.12 rad knots, 25% speed, 0.04 rad reset
jitter), with zero added randomization. Keep reward coefficients unchanged.
Admission retains rejected samples; a fresh full teacher report and source-bound
regressions precede the bounded PPO pilot. No audit case enters training or
selects a curriculum level. [Protocol and target-gallery commands](../experiments/mars_arm_splines_20261117.md).

## Maintenance checklist

For every run record planned/actual tick fractions, exact case manifests and
seed namespaces, source checkpoint and new/cumulative decisions, randomization
and disturbance settings, episode starts/completions, and scope of manipulation.
Link the full outcome report, including all force/wheel/stop failures. Label
reused development audits separately from fresh confirmation. Keep weights,
compressed traces and source archives ignored; commit only compact evidence.

The [fresh target gallery](../../paper/data/whole_body_control/fresh_tracking_confirmation_v1/target-gallery.svg)
shows the sampled base/hand poses and orientation axes. It is for inspection;
controllers execute the continuous references, not these sparse samples.

Hardware status is maintained separately in
[Mars hardware readiness](MARS_HARDWARE_READINESS.md).
