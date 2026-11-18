# General goal-change curriculum

This implements the planned [training mixture](../rl/TRAINING_MIXTURE.md):
30% regular hand-active, 10% base-only, **10% nominal goal changes**, 50% unloaded
TAMP prefixes, scheduled by physical control ticks. It adds generic command
handoffs, not task semantics or permission to skip TAMP grasp/settling checks.

`GoalChangeTrackingEnv` constructs two admitted FK references at reset. The next
reference starts from the first nominal endpoint, independently of tracking
error. Its pose/velocity/acceleration join is continuous. Announce it after the
first planned endpoint but before its two-second quiet hold, preserve physical
state/history/reward memory, and expose the new reference before the next actor
action. No controller stop/fault is cleared. An already stopped environment
cannot step; a terminal failure cannot install the next trajectory.

The versioned `configs/benchmarks/mars_goal_changes_v1.json` has eight training
and eight new audit cases. Generator/reset seed namespaces are disjoint from the
previous fresh confirmation cases. Both coupled and free-base hand commands are
included. Only training IDs enter PPO. Both reference stages require collision,
joint-margin and support admission; all failures remain recorded. Teacher success
is a comparison, not a claimed optimum or prerequisite that filters difficult
valid paths. Experimental readiness requires admitted nominal paths, confirmed
teacher/source stops and no invalid physics; unsafe/task failures stay counted.
No success or physical guard threshold is relaxed.

The candidate starts from the longer run's actor/critic/Adam checkpoint, verifies
its checksum and exact loaded state, and explicitly resets physical state/cursors
with seed 109. Source snapshot counter is 65,536 additional decisions, lineage
81,920. Budget is **131,072 further decisions** (212,992 lineage if complete),
LR 1e-5 → 1e-6, original reward/exploration and three independent critic epochs.
No fresh-critic warmup, audit PPO data, intermediate checkpoint selection,
automatic extension, promotion or hardware motion.

The new source requires its own full regression receipt; the preceding 1,585-test
receipt is historical. New tests cover history/cadence preservation, nominal
controller-independent goals, C2/yaw-wrap continuity, horizon rejection, stop
latching and frozen-case/mixture validation. Portable snapshot/source manifests
bind the new config, generator and indirect test inputs.

Exact protocol/runners are under ignored `logs/recovery/goal-change-curriculum-v1/`;
raw checkpoints/readouts under `logs/manipulation/goal-change-curriculum-v1/`.
A source-bound full gate and fresh 16-case teacher/source admission precede PPO.
Final fixed-policy readout covers regular development, existing coupled, TAMP
prefixes and separately reported goal train/audit cases. Baseline regular/TAMP
rows are explicitly reused from the source actor; they are not fresh trials.
Native training remains one CPU14 worker, one numerical thread and hidden CUDA,
with the test worker on CPU15. Retained viewer restoration is automatic.

Herman work remains passive: [current feedback audit](herman_feedback_audit_20261117.md).
Preparing this simulator curriculum does not enable the hardware executor.

## Launch status

Source `2a2ce5c1` is committed and frozen under ignored workspace runtime storage,
with matching canonical controller/test digest. Targeted handoff, deadline,
regression-receipt and snapshot tests pass **36/36**; formatting/lint checks pass.
Managed job `20261117_111259_6729bc` now runs fresh full regressions on CPU15
and all 16 new teacher/source goal-change cases on CPU14. PPO is queued behind
those checks, not yet running or qualified. The retained viewer is paused during
native work and restored by the supervisor. Exact compact
[launch evidence](../../paper/data/whole_body_control/goal_change_curriculum_v1/README.md)
includes immutable-input digests and the full producers; raw logs remain ignored.

## Admission complete; PPO running

Frozen source `2a2ce5c1` passes **1,604 source-bound regressions**. Both nominal
stages admit for all 16 cases under both baseline controllers. Teacher/source
success is **13/16 vs 9/16**, with **32/32 confirmed stops**. Teacher has one
safety-flagged episode; source has none. No case was removed. Managed job
`20261117_111259_6729bc` has entered PPO; W&B offline ID `4przjfk9`. The budget
and fixed-final readout remain unchanged. Subsequent hardware/UI development runs
outside this frozen runtime and does not inherit its regression qualification.

## Completed fixed-budget results

The run and paired readout finish: **131,072 additional decisions**, **155,423
physical ticks**, **212,992 lineage decisions**. Audit IDs never enter PPO.

| Cohort | Source | Curriculum final |
| --- | ---: | ---: |
| Regular development | 19/24 | 22/24 |
| Existing coupled | 7/10 | 6/10 |
| Unloaded TAMP prefixes | 6/6 | 6/6 |
| Goal changes, all | 9/16 | 6/16 |
| Goal changes, training identities | 4/8 | 3/8 |
| Goal changes, audit identities | 5/8 | 3/8 |

All **56 candidate evaluation stops confirm**. Evaluation safety flags remain
three episodes across the regular/existing cohorts; goal-change evaluation has
none under either policy. Worst goal position error improves 32.05 → 27.77 mm,
yaw jerk 91.04 → 58.98 rad/s³, and contact load 12.94 → 0.196 N, but three
previously successful goal cases regress. Those new misses end at 6.27, 4.69 and
7.28 mm; all ten failed handoffs exceed the unchanged 3 mm hand position bound,
four also exceed 0.02 rad orientation. All ten satisfy the coarse base region;
all receive their second command. These are tracking timeouts, not geometry
rejections, missing updates or failed stops. Terminal metrics follow stopping;
they do not by themselves prove the active policy had reached a false equilibrium.

Actual tick shares are **29.92/10.22/9.99/49.87%**. The goal family completes only
**32 training episodes: four per training identity**, with **31 timeouts and one
wheel-lift stop**, zero stochastic successes. Thus total PPO decision count
substantially overstates transition experience. This is a measured distribution/
exploration concern, not proof that increasing its weight alone will solve it.

**Do not promote this checkpoint or extend its budget automatically.** Preserve
the ordinary-tracking/smoothness improvements, but inspect pre-stop traces of
the three regressed cases: active goals, measured/commanded velocities, actor
feedback contributions and per-tick tracking/stability rewards. Separate missing
fine correction from noisy-training terminal-credit failure before another run.
Herman remains preview/passive only. The retained simulator viewer is restored
on port 8766; no PPO job remains active. Compact complete case outcomes and
source-bound raw-report checksum are in
`paper/data/whole_body_control/goal_change_curriculum_v1/results.json`.
