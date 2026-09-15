# Shared grounding: bounded cross-task pilot

## EQA startup restoration (September 15)

All six `staged-pregrasp-eqa-20260915` cases abort with exit 134 before answering:
`GL::Context: cannot retrieve OpenGL version`. These are infrastructure failures,
not 0/6 accuracy. A bare Habitat probe passes, while importing MuJoCo first or
importing the full EQA runner reproduces the native abort. OpenCV, Torch and
Open3D individually do not reproduce it (serial import-isolation job
`20260915_153108_5a8419`). The September 13 A* change `89581c70` imports shared
grid helpers through `base_goal_rank` → `voxel_arm_collision`, unintentionally
initializing MuJoCo's GL backend inside Habitat.

`5d299c9d` defers MuJoCo to the actual FK call; grid/navigation imports no longer
initialize it. `fdb441a9` adds a full-runner-import plus real-scene rendering
preflight before the batch's model checks and episodes. Import-boundary and
on-demand MuJoCo FK regressions pass; broad suite **607 passed / 4 skipped**.
Full-stack rendering passes in `20260915_153330_611847` without package, driver,
model, or renderer-configuration changes.

EQA retry `20260915_153507_23eb03` freezes `fdb441a9` in `/tmp/emet-reach-eval`,
writing `~/runs/emet/eqa-restored-20260915`. Same q15/16/25, hybrid/Qwen-box
presets, Qwen int4, 20 planning/10 movement steps, no semantic/enriched hints,
and map/video exports. Answer results are pending; restored rendering alone
does not establish EQA accuracy or no regression. The two commits are separate
from manipulation changes so the merge-critical startup repair can be reviewed
independently.

## September 15 staged pregrasp and forward reacquisition pilot (running)

Frozen `0ecc0aa9` combines `31d1002b` (complete an upward pregrasp lift before
extension; hold base/wrist/extension during that stage and stop if it fails)
with forward-view arrival verification before the existing fallback sweep.
Candidate identity and fresh visual acceptance remain required. Neither change
introduces a scene-specific clearance, weaker success threshold, or larger
timeout. Broad checks: **598 passed / 4 skipped**. Review: PR #169, stacked on
#167. This is not general collision planning or completed cross-task acceptance.

All jobs share the exclusive GPU lock and safe CPU affinity, with a frozen
evaluation checkout and new output directories:

| Case | Job | Artifacts under `~/runs/emet/` |
| --- | --- | --- |
| Molmo same wheel-priority fixture | `20260915_084618_0fc1f5` | `staged-pregrasp-20260915/molmo` |
| Original tabletop control | `20260915_084622_88d93c` | `staged-pregrasp-20260915/tabletop` |
| Native RoboCasa can | `20260915_084626_fc3604` | `staged-pregrasp-20260915/can-native` |
| Explicit RoboCasa NoSlip=10 control | `20260915_084630_7e774e` | `staged-pregrasp-20260915/can-noslip` |
| EQA q15/16/25, hybrid and Qwen-box presets | `20260915_084704_ce9c77` | `staged-pregrasp-eqa-20260915` |

Results are pending. Simulator cases use the existing tracked-narrow preset;
the EQA rerun retains the earlier comparison's two presets, int4 Qwen, budgets,
and semantic-label exclusions. It is a matched-question historical comparison,
not a same-seed causal test or the complete four-row acceptance matrix. Maps,
videos and grounding evidence are enabled. Learned TAMP remains gated on the
room manipulation checks; no oracle battery result substitutes for that gate.

First result: Molmo on `0ecc0aa9` completes the staged pregrasp and reaches
visual servoing, clearing the earlier counter-front obstruction. It still
fails pickup/place (verified F/F, 356 wall seconds): the end effector remains
about 0.28 m short of the tomato while the arm approaches its extension limit.
The last-state audit finds no fingertip contacts; arm joints are about
0.123 m each against 0.13 m limits. The viewing pose is beyond the existing
grasp workspace's upper radial bound, but the handoff previously checked only
minimum distance. This is not permission to enlarge arm limits or skip motion
confirmation.

Follow-up `6a1579e4` checks both workspace bounds and uses the existing
collision-checked relocation planner, retaining the low-target feasible-IK
exception and rejecting measured arrivals still outside the workspace. Broad
checks: **600 passed / 4 skipped**. Frozen independently in
`/tmp/emet-reach-eval`; queued exact Molmo `20260915_085532_ac9ef2` and tabletop
`20260915_085536_b4353a` write under `~/runs/emet/workspace-handoff-20260915`.
Their live results are pending. The earlier queue, including EQA, remains on
`0ecc0aa9` without this manipulation-only follow-up; no running source changes.

## Wheel-contact candidate: first turn repaired; approach-monitor bug exposed

`52bebc6d` on `fix/room-wheel-contact-profile` gives only Stretch's drive-wheel
collision shapes material priority 1, consistent with the existing caster and
pad profiles. Four compiled contact tests cover both wheels on tabletop and
high-torsion room floors; broad checks pass **593 / 4 skipped**. This changes
contact parameter mixing, not just one friction coefficient. No floor geometry,
solver setting, policy parameter, or success threshold changes.

Exact Molmo retry `20260914_235619_0e73b1` uses a separately archived fixture
under `~/runs/emet/molmo-wheel-contact-20260915/fixture`; its manifest and freezer
verify two changed wheel priorities and unchanged masses, inertias, poses,
friction arrays, and NoSlip=4. Original artifacts remain intact. Tabletop control
`20260914_235634_4df8bf` queues on the same exclusive GPU lock and retains its
explicit NoSlip=10. Both run frozen source `52bebc6d`, Qwen int4, tracked-narrow,
and lazy graph. The room retry fails physical pickup/place (verified): it passes
the formerly blocked first turn, then stops at waypoint 2 with XY error 0.024 m
and yaw error 0.730 rad. The trace shows continued translation into the inner
approach radius, not an immobile robot. The monitor ignores XY improvement
once the previous error is within its outer 0.07 m tolerance. `4c08782d` removes
that progress gate without changing arrival tolerances, stall windows, or
deadlines; valid inner-approach and stationary wrong-heading tests pass, with
**595 broad tests passed / 4 skipped**.
The wheel-only tabletop control **passes verified physical pickup/place**; its
final reconstruction shows the cylinder upright on the block with the gripper
withdrawn. This is one positive control, not a new six-case panel. Same-fixture
room retry `20260915_000525_17f117` runs `4c08782d` under
`~/runs/emet/molmo-wheel-contact-20260915/progress-fix-policy`.

Explicit RoboCasa solver control `20260915_000550_b5b250` queues behind it on
the same exclusive lock, using `4c08782d` and an archived fixture under
`~/runs/emet/room-can-noslip-control-20260915`. Only NoSlip iterations change
from 0 to 10; the native frozen wheel priorities and all object parameters are
retained. This tests the recorded-control retention hypothesis with the learned
agent; it is **not native-physics room acceptance**.

The progress-fix room retry completes navigation and arm-facing alignment,
then fails the separated pregrasp approach (237 wall seconds, verified F/F).
The arm's commanded lift/extension are 0.885/0.333 m; the final measured values
are about 0.786/0.099 m. A non-integrating frozen-state contact audit finds both
finger pads pressing against the island front (individual recomputed normal
forces up to about 35 N). This is a real obstructed simultaneous raise/extend
motion, not grounds for increasing timeouts or force limits. The audit script
and output are retained in `progress-fix-policy/replay/arm_contact_audit.*`;
forces are recomputed diagnostics, not recorded force telemetry. Next test is
staged pregrasp motion before considering general clearance planning. Navigation
completion is not room-task acceptance; no pickup occurred.

The explicit RoboCasa solver control stops before grasping (77 wall seconds,
verified F/F): after navigation, head-sweep reacquisition selects the other can
in the sink, then the target-ambiguity gate rejects pickup. Saved images confirm
that the sink view is not the originally selected counter can. The generic
instruction "can" is also underspecified in this multi-can fixture. Thus this
rollout is **inconclusive about retention**, not evidence that NoSlip=10 repairs
or fails physical grasp. Preserve it and audit target-directed reacquisition
and task identity before another learned retention comparison. Shutdown EGL
and manager errors remain visible in the log and are not task-success evidence.

## Room follow-up: retention handoff repaired; room acceptance still pending

The native-physics RoboCasa can case fails physical pickup and times out on
`b5fd55ef`. On `1d257800`, fresh payload verification correctly stops the same
failed pickup in 178 seconds; the known-good original tabletop neighbor passes
pickup/place in 251 seconds, with its final reconstruction manually inspected.
This is one positive control, not a new six-case panel. Molmo's lower-table
case stops at its first turn, with a separate positive-contact-margin recorder
bug fixed in `65574600`. Latest broad tests: **589 passed / 4 skipped**.
See the retained room traces and diagnostics below. No room OVMM or learned
TAMP acceptance is claimed; the earlier frozen six-case result remains separate.

## Latest checkpoint: repaired-source basic gate passes 6/6

Frozen **`4f78ae62` passes all six physical pickups and placements**: two
executions each of original, mirrored and separated tabletop clutter. Jobs
`20260914_220543_2c4d00` and `20260914_221712_59ba4e` ran serially, in the
declared split order (mirrored 1, original 1, mirrored 2, original 2, separated
1/2). All six final reconstructions were manually inspected. Same seed 0,
Qwen3-VL-8B int4/SDPA, tracked-narrow, lazy graph and explicit NoSlip=10
throughout; these are execution repeats, not six independent environments.

| Fixture | Physical pickup/place | Wall seconds, repeats 1 / 2 |
| --- | --- | --- |
| Original | 2/2 | 242 / 246 |
| Mirrored neighbor | 2/2 | 251 / 237 |
| Separated neighbor | 2/2 | 264 / 263 |

This satisfies Stage B on the repaired manipulation source. It is not room
OVMM, learned TAMP, paired EQA/find acceptance or a hardware guarantee. Previous
failed panels below are retained and not pooled. The later **launch-only**
Molmo discovery fix `b5fd55ef` is not part of this frozen physical panel.
Latest broad checks including its symlink regression: **570 passed / 4 skipped**;
separate Molmo config/CLI checks: **35 passed / 1 skipped**. The first CLI test
invocation rejected a mismatched worktree-local virtualenv; rerunning with the
review source on `PYTHONPATH` from the owning environment's checkout passes.
No CLI environment guard was weakened.

## September 14 closeout: mirrored placement stops the repaired-source panel

Job `20260914_214506_a2a467` freezes `787acb6f`, Qwen3-VL int4/SDPA,
tracked-narrow, lazy graph and the same NoSlip=10 wrappers. Original clutter
passes pickup/place **2/2** (244 / 253 wall seconds); both final close-up
reconstructions were manually inspected. Mirrored repeat 1 passes pickup but
fails placement (233 seconds), so repeat 2 is **unrun**. Together with the
same-source separated fixture's earlier 2/2, this is **4/5 completed placements,
one unrun**, not Stage B acceptance. The older `ef533ed3` 6/6 is not pooled.
Artifacts: `~/runs/emet/manipulation-closeout-20260914`.

| Failed local alignment (`787acb6f`) | Sequenced alignment retry (`4f78ae62`) |
| --- | --- |
| ![Cylinder still held against the cube after failed alignment](../environments/figures/tabletop-mirrored-held-failure.png) | ![Cylinder on cube after release and gripper withdrawal](../environments/figures/tabletop-mirrored-released.png) |

These are **private-qpos reconstructions, not agent camera views**. Both use
native Stretch `scene_right_neighbor_noslip.xml`, seed 0, tracked-narrow,
Qwen3-VL-8B int4/SDPA, lazy graph, wheel-drive navigation and a 600-second
episode cap. The left trial has pickup T / place F; the right has T/T under the
independent scorer. Selected to illustrate the diagnosed failure and exact-case
retry, not to replace the full-panel scores. See the
[figure provenance](../environments/figures/README.md#mirrored-tabletop-sequencing).

The failed mirrored run correctly reports failure and does not open its gripper.
It first commands a combined lateral/downward correction with observed error
(-5.0, -24.3, -57.9) mm. The next error is (-2.1, -16.2, -1.9) mm: height is
near release, but lateral alignment has not converged. That correction times
out. Private no-integration reconstruction under
`~/runs/emet/mirrored-motion-audit-20260914` shows simultaneous object/gripper
and object/cube contact. Final measured arm extension is 0.34503 m against a
0.36973 m command, outside the unchanged 0.02 m completion tolerance. Sampled
contacts show no gripper-body/counter obstruction like the earlier sink case.
This supports a local ordering hypothesis: lowering before centering makes the
remaining correction drag a held object across its support. It is not evidence
that the printed navigation-tolerance warning caused the timeout.

Candidate `4f78ae62` centers horizontally before descending. If a fresh view
already puts a laterally misaligned object near the release surface, it aborts
instead of dragging or releasing it. Keep the existing four-observation limit,
5 cm step/reference bounds, identity/geometry checks and release thresholds.
This is a shared local sequencing change, not general clearance planning. It
requires an exact mirrored retry and neighboring control before a new panel.
Offline checks pass **567 tests / 4 skip**, including near-support no-drag
negatives. Job `20260914_220543_2c4d00` runs mirrored then original serially,
stopping on physical failure; artifacts
`~/runs/emet/placement-sequencing-control-20260914`. This is diagnostic
verification, not a replacement six-case panel or room acceptance.
The mirrored retry completes **pickup T / placement T** in 251 wall seconds
(pick 73.316 s, final stable placement 162.788 s simulation time). Its observed
corrections are (-3.8, -20.1, -55.7), (-0.3, -8.0, -44.4), then
(2.8, 0.2, -1.5) mm: horizontal convergence precedes final descent. The final
close-up reconstruction was manually inspected, with the cylinder upright on
the cube after gripper withdrawal. The original-clutter neighbor also passes
T/T (pick 71.378 s, final stable placement 161.288 s simulation time), with its
final close-up inspected. Job `20260914_221712_59ba4e` runs the remaining four
same-source repeats serially. All four subsequently pass, as summarized above.
The earlier queued
`20260914_221205_43bdac` was cancelled before execution to avoid a launcher
lock/prerequisite dependency; it contributes no experimental outcome.

General end-effector clearance planning is explicitly deferred from this PR;
the failed sink remains in the report. Accessible room OVMM, learned two-step
TAMP and paired EQA/find still gate promotion. Geometry preflight
`20260914_215413_ed37f8` generated and archived a native RoboCasa seed-0 room,
but its diagnostic renderer requested 960 pixels from a 640-pixel framebuffer.
No policy ran. Preserve that scene/metadata and render within its framebuffer;
do not regenerate until a preferred object appears. Other room fixtures remain
unprepared, not implicit successes. The actual generated client URDF used by
the tabletop candidate has SHA256
`1594765a3119bc03fa0140799ef77e5796703a7f30fa93685988e24231815fe4`,
matching the archived legacy model in the embodiment audit below.

### Room preflight: dependency dynamics, not another learned failure

Preflight v2 `20260914_220622_2cb1d3` successfully renders archived RoboCasa
seed 0 and newly generated seed 1. It stops before Molmo generation because
the evaluation worktree cannot discover the installed Molmo interpreter.
The installed interpreter itself imports successfully, but the existing
`MOLMOSPACES_PYTHON` override resolves its Python symlink to the bare interpreter
and loses the virtualenv's packages. `b5fd55ef` preserves the absolute invocation
path and the existing import-validity check. The pending retry
`20260914_222412_116148` was cancelled before execution; the corrected preflight
uses the same installed environment, not a reinstall or different agent.
Neither preflight is a room-policy result.

Corrected-launch job `20260914_223654_1d5575` then merges and compiles Molmo
index 0 but fails to reload its archived XML because relative robot asset
directories no longer resolve from the artifact directory. The private preflight
script now rebases those directories without changing geometry or dynamics.
Job `20260914_224024_b8970e` completes both indices 0 and 1 on `b5fd55ef`;
artifacts: `~/runs/emet/accessible-molmo-preflight-v2-20260914`. Archived model
mass, inertia, inertial positions and initial qpos match the initial compilation
within the script's recorded tolerances. Native NoSlip=4 is retained, distinct
from the tabletop NoSlip=10 row. No physics integration or policy was run.
Manual inspection shows the exterior overview/top-down cameras see the ceilings;
these are not useful interior task views. Interior viewpoints, robot starts,
target identities and supports still require predeclaration before acceptance.

The new RoboCasa scenes are **not yet admitted for manipulation**. Seed 0's
can/fish/canned-food masses are 2.941/2.063/4.418 kg; seed 1's
spray/baguette/cream-cheese masses are 6.304/5.860/3.383 kg. Source audit
`~/runs/emet/room-source-mass-audit-20260914` confirms EMET preserves the
in-memory native masses/inertias exactly. However, the installed
`cpaxton/robocasa` fork at `3d0bd42` already rewrites **every unspecified mesh
mode to shell** in `Kitchen.edit_model_xml`, before that source model exists.
Thus source-to-adapted equality alone does not validate authored dynamics.
The installed checkout also has unrelated local changes; do not overwrite them.

A compile-only process with that blanket rewrite removed successfully generates
a room, with authored modes and densities untouched; artifacts
`~/runs/emet/room-source-mass-authored-20260914`. It samples different objects,
so its masses are **not a paired quantitative comparison**. The separate
matched control `~/runs/emet/can-inertia-control-v2-20260914` restores only the
same can's five unspecified mesh modes in cached native XML: recompiled mass
changes from 5.467 to 0.0986 kg, with world-space mesh vertices, collision flags,
qpos and non-target masses unchanged. The 5.467 kg value is a cached-XML reload,
not the earlier 2.941 kg in-memory source; keep those serialization stages
distinct. The first control's raw mesh-frame equality check was invalid because
MuJoCo changes principal-axis storage frames with inertia mode; its failed
artifacts remain, and v2 compares transformed vertices instead.

The subsequent approved fix is isolated in
[RoboCasa PR #1](https://github.com/cpaxton/robocasa/pull/1), commit `f106e07`
against `feature/v1.24`. It removes only the blanket rewrite, preserving explicit
shell modes. Its regression checks both XML modes and compiled body dynamics:
**1 passed**. The same hunk is applied to the installed dirty checkout without
overwriting its other edits; this is not a clean dependency checkout or merged
release. Fresh seed-0/1 preflight `20260914_225528_1fa35e` completes under
`~/runs/emet/authored-room-preflight-20260914`: can 0.02976 kg and spray
0.06098 kg, with native NoSlip=0 retained. Both expanded archives reload with
the preflight's mass/inertia/initial-state equality checks. These fresh adapted
masses are not the separate cached-XML can control's 0.0986 kg measurement.
The regression also passes against the installed dependency. An initial test
invocation collected unrelated dataset tests through ROS pytest plugins;
the isolated rerun disables plugin autoload and passes (no product change).
Do not tune densities, resample
until a policy passes, or label these rooms full-OVMM evidence. General sink
clearance remains separately deferred.

Room-case preparation `20260914_225935_008662` predeclares can/spray placement
onto `counter_right_main_group_main` (the open counter to the right of the
stove), each with explicit visible/search starts. This changes the task from
the generator's counter-to-sink instruction; report it as **derived accessible
room OVMM**, not canonical RoboCasa success. Native object geometry/dynamics
and NoSlip=0 stay fixed. Fixture manifests and the installed dependency diff
are archived under `~/runs/emet/accessible-robocasa-frozen-20260914`.
The candidate remains `b5fd55ef`, Qwen3-VL-8B int4/SDPA, tracked-narrow,
lazy graph and 600-second case limits; neither coordinates nor evaluator body
names are supplied to the agent. All four cases must retain failures and unrun
cases. Head-view/start preflight precedes any policy execution.

Molmo geometry inspection `20260914_225657_5bb03e` also completes, with private
interior views and whole-object **subtree** masses in
`~/runs/emet/room-interior-audit-20260914`. Several object root bodies have zero
individual mass but positive descendant mass; they are not massless objects.
Movable tomatoes, apples and open bowls are visible. Exact tasks/starts are
still pending; these camera orbits are not agent observations or policy passes.

### First derived room rollout: can retention fails

Job `20260914_230140_40e8a7` runs frozen seed-0 visible-start can → counter to
the right of the stove on `b5fd55ef`, Qwen int4, tracked-narrow, lazy graph,
native NoSlip=0 and wheel drive. It **times out at 600 seconds** (601 measured
wall seconds, exit 124); independent score is **pickup F / placement F** with
a valid physical trace. Artifacts:
`~/runs/emet/accessible-robocasa-policy-20260914/seed_0_visible`. The other
three RoboCasa cases are unrun, not failures or passes.

Manual inspection confirms correct initial can grounding, and approach motions
complete. During closure/lift around simulation seconds 63–66, the can contacts
the fingers, briefly rises, loses contact, and lands on its original counter.
The final reconstruction shows it on its side behind the paper-towel holder.
This is not a retained pickup. `_grasp()` returns lift-motion completion; the
task consequently continues to destination search without freshly verifying
possession and never places before timeout. Fix that handoff using observed
RGB-D, not evaluator contacts. The shutdown log contains a manager BrokenPipe
traceback; afterward no matching agent process or GPU compute client remains.
The timeout bypasses the driver's final scoring command, so the saved trace
was explicitly scored afterward (the replay manifest independently agrees).

Private solver-control job `20260914_230920_5ac3db` completes a replay of the
recorded closure/lift controls with NoSlip=0/10. It is an approximate 10 Hz
control replay, not an exact thread/state replay or learned acceptance. Require
qualitative reproduction of the native failure before attributing solver effects.
The native replay reproduces can loss (final z=0.950 m); NoSlip=10 keeps it
elevated through the same controls (final z=1.096 m). Artifacts:
`~/runs/emet/room-grip-solver-control-20260914`. This supports a solver-retention
hypothesis but is not permission to silently change the native room row.

`1d257800` adds fresh observed lift/retention checks before and after the carry
transition, reusing Qwen-verified RGB-D without creating memory instances.
Unknown possession still blocks a duplicate pickup. Exact native-room retry
`20260914_232230_b2bac0` finishes in **178 wall seconds**: physical F/F remains,
but absent/ambiguous post-lift evidence now stops the task and skips placement
instead of searching until timeout. This is a corrected failure handoff, not
a repaired physical grasp. Artifacts:
`~/runs/emet/room-can-retention-guard-20260914/seed_0_visible`. Same-source
original-tabletop control `20260914_232231_d88be1` passes **T/T in 251 wall
seconds**, with both fresh payload checks and the final reconstruction inspected.
It is not a replacement six-case panel. Artifacts:
`~/runs/emet/tabletop-retention-guard-control-20260914`. Broad checks for this
handoff source: 587 passed / 4 skipped.

### Molmo: initial turn failure and contact-margin scoring bug

Tomato-to-bowl preparation `20260914_230526_03cdc9` stops because its private
floor check relies on mesh names. The floor mesh has a generic hash name.
`20260914_231450_d5e322` repeats the **same starts**, checking static ancestry
and the archived rooms' zero floor elevation, and completes all four scene/start
archives under `~/runs/emet/accessible-molmo-frozen-v2-20260914`. Head views were
inspected. The higher-table index-0 task still needs a reachability check; do
not admit it solely because the object is visible.

Lower-table index-1 visible run `20260914_231706_1626e6` on `b5fd55ef` correctly
grounds the tomato but stops at the first turn: yaw error remains 0.646 rad
and the server reports a stopped, stalled navigation command. The robot remains
upright; this is not a grasp failure or evidence that it fell over. The tool
reports failure and skips placement. Original physical score is **unverified:
no settled baseline**, not a verified successful or failed physical pickup.
Artifacts: `~/runs/emet/accessible-molmo-policy-20260914/index_1_visible`.

Private no-integration audit `~/runs/emet/molmo-contact-audit-20260914` finds
force-bearing target/support contacts at positive separation (about 1 mm within
the active margin). The old recorder discards them because distance is positive.
`65574600` records active positive-normal-force contacts, with force/distance and
contact-rule provenance; inactive proximity does not count. Physical lift and
stability thresholds are unchanged. New margin/gap tests pass; broad checks
are **589 passed / 4 skipped**. Applying the recorder to frozen states with
recomputed forces establishes a baseline and still scores F/F, under
`~/runs/emet/molmo-contact-recorder-control-20260914`. This is an explicitly
reconstructed diagnostic, not a relabeled original rollout. Original artifacts
and scores remain unchanged; the two live handoff checks use `1d257800`, not
this later evaluator commit.

The turn failure remains separate. The scene has overlapping floor contacts
and high native torsional friction; neither is yet established as the cause.
Fixed-turn diagnostic `20260914_233017_b37556` completes a comparison of native
contact mixing with authored wheel-material priority, holding the recorded
initial state, controls and solver fixed. In six simulation seconds, native
mixing turns only **0.122 rad**, versus **2.111 rad** with wheel priority 1;
both stay upright (minimum base up-axis z > 0.9999). No floor geometry is
removed. Artifacts: `~/runs/emet/molmo-turn-control-20260914`. Priority changes
contact parameter mixing, not just torsional friction, so this does not isolate
a single friction coefficient. It supports a robot contact-profile correction
followed by an exact-case and tabletop regression; production floor/material
settings remain unchanged at this checkpoint. Shutdown manager BrokenPipe
errors are also retained, not suppressed as part of this physics investigation.

## September 13: basic manipulation gate passed, cross-task gates pending

**V7 passes 6/6 independently scored physical pickups and placements.** Managed
job `20260913_195911_6c9e7c` ran all six serially on frozen
`ef533ed3fced418b191161a7e74835afed94de47`, with the same tracked-narrow agent,
Qwen3-VL int4/SDPA, lazy graph, task budgets and NoSlip=10 wrappers throughout.
All six final qpos reconstructions were manually inspected. Artifacts:
`~/runs/emet/manipulation-panel-v7-20260913`; each case includes exact source,
configs, command, process status, physical trace/score, wrist evidence and replay.

| Fixture | Physical pickup | Physical placement | Wall seconds, repeats 1 / 2 |
| --- | --- | --- | --- |
| Original | 2/2 | 2/2 | 242 / 263 |
| Separated neighbor | 2/2 | 2/2 | 258 / 259 |
| Mirrored neighbor | 2/2 | 2/2 | 220 / 222 |

This satisfies Stage B's predeclared gate. These are repeated executions at
seed 0, not six independent environments, and all use explicit numerical-physics
and aperture variants. No production contact-solver default is promoted.
Earlier failures and development retries are not pooled. The key final repair
corrects an 8.7 mm camera-to-grasp mismatch and publishes rendered poses from
the same state as RGB-D, without changing object-specific offsets or tolerances.
General clutter robustness, single-room OVMM, learned TAMP and paired EQA
acceptance remain pending.

Post-panel-source work is separate: `82eaa6f3` adds private ordered distinct-object
scoring; `56ba1df7` adds an open-sink room fixture; `fd6c0041` retains native
simulation capture stamps through the clients. These are not part of frozen v7.
Latest offline checks: 440 passed / 4 skipped in the broad suite, 26 focused
camera/timing/codec tests and 18 simulator-configuration tests. Ordered scoring
controls are synthetic, not learned-TAMP successes. Follow-on containment and
driver changes bring the broad suite to **442 passed, 4 skipped**; rescoring
the six frozen v7 traces preserves all original physical results and times.

### First open-room diagnostic (after the basic gate)

Preflight `20260913_202959_f17d9b` on `fbb38412` successfully generates the
declared RoboCasa `PickPlaceCounterToSink` layout/style 1 for seeds 0/1. Targets
are pear / spray bottle. Both rooms use native NoSlip=0, unlike tabletop v7.
Generated XML, metadata and exact preflight script are archived under
`~/runs/emet/open-sink-room-preflight-20260913`.

`dfa19572` adds evaluator-only oriented basin regions so rim contact alone does
not satisfy an inside-sink task. `54097954` parameterizes the existing shared
pilot's natural-language instruction, seed and physical evaluator, preserving
tabletop defaults. Job `20260913_203846_45e788` runs the first **integration
diagnostic**: pear into sink, seed-0 adapter startup, same Qwen int4/lazy
tracked-narrow agent and 600-second cap. Private target/support bodies and basin
regions are not policy inputs. Artifacts: `~/runs/emet/open-sink-pear-diagnostic`.
That attempt failed **before simulator startup**: the runtime asset check looked
at the evaluation worktree's empty submodule, while the wizard imported a complete
editable RoboCasa installation from the original checkout. No robot action or
physical trace was produced; this is an infrastructure exclusion, not a policy
failure. `e3c48a9c` resolves runtime asset checks against the imported package and
skips repair writes when it is already complete. Read-only checks pass all five
asset predicates; 22 focused path/config/driver tests pass. Exact-case retry
`20260913_204353_cface0` on `e3c48a9c` completed with **no physical pickup or
placement**, artifacts `~/runs/emet/open-sink-pear-runtime-assets`.
The adapter's existing open-floor spawn ranking moved the robot 2.034 m from
RoboCasa's suggested base pose (hint XY 1.140/-0.751, actual 2.933/-1.712).
Its ranking favors the walkable-region centroid before distance to the hint;
this is not evidence that every closer pose was in collision. Accordingly this
is an **adapter-start search diagnostic**, not a visible-target control or the
literal RoboCasa suggested start. Spawn selection was not changed for this run.

Search reached a visible pear after about 406 seconds of tool execution. The
final head capture `grounding-a99600d886184050a1071a3eb0e39078.png` was manually
inspected: the pear is at the right edge of the counter. Qwen selected its
355-pixel measured surface from two proposals. Pickup nevertheless failed
before motion with `Manipulation requires a unique query candidate`: the
handoff counted remaining unverified voxel search hypotheses as competing
objects. A focused regression reproduces that failure. The correction prefers
non-invalidated grounded references over search-only hints, still reacquires a
new image, and still rejects multiple grounded references. It does not delete
unexplored hypotheses or relax semantic/depth checks. The focused grounding,
handoff and manipulation suite passes **123 tests**, including absent/stale
reacquisition and true ambiguity controls. A new frozen exact-case retry is
required; this software fix is not yet a room manipulation success.

Retry `20260913_205646_cf97b0` on `33fa549f` was **cancelled and excluded from
the paired comparison**: despite `--sim-seed 0`, startup generated a **can**
instead of the declared pear. Its different search route is not evidence for
or against the handoff fix. The unchanged pear instruction was incompatible
with that generated task. Logs/images/partial trace remain under
`~/runs/emet/open-sink-pear-grounded-handoff`; do not render that trace against
the earlier pear XML or score the can as though it were the requested target.

Generation-only probe `20260913_210843_a03324` confirms the mismatch. It runs
seed 0 twice per process, with `PYTHONHASHSEED=0` and `1` in separate processes:
hash-0 generates pears at different counter positions in its two repeats;
hash-1 generates cans. Results and exact script:
`~/runs/emet/robocasa-seed-repeat`. Thus merely fixing Python's hash seed is
not sufficient. The installed dependency contains unordered counter-region
deduplication (`list(set(valid_geoms))`); this is a suspected contributor,
not yet a demonstrated complete cause or a fixed dependency. Freeze generated
scene/task metadata for paired evaluations rather than trusting a seed alone.

`cfb8558a` expands robot includes when writing generated XML, fixing a separate
provenance hazard: the previous saved scene referenced `stretch_temp_abs.xml`,
which later generations overwrite. Generation regression
`20260913_211223_c6b15d` passes after deliberately overwriting that include and
reloading the archived model. External mesh/texture assets remain required.
The explicit visible-start control is prepared from the archived **pear** room,
not a new random generation; record its start intervention separately from
adapter-start search and retain native NoSlip=0.

Frozen visible-control preflight `20260913_211328_f13863` creates
`~/runs/emet/open-sink-frozen-visible-20260913`: expanded scene XML, sim config,
manifest/hash, exact preparation script and four head renders. The explicit
base start is XY **(1.0, -1.3)**, yaw **pi/2**, with native NoSlip=0. The pear
and sink are visible in the preflight sweep. These are kinematic diagnostic
renders (including rangefinder visualization), not agent observations or
physical execution. The robot start is an intervention, not a search success.

Learned run `20260913_211456_985f71` on `98b4c708` reaches fresh pear grounding
and the grasp operation, then fails **before closure** on the aperture-standoff
guard; independent physical result is **F/F**. Artifacts:
`~/runs/emet/open-sink-pear-frozen-visible`. The last recorded pose and measured
head target imply 0.594 m target-to-grasp separation, predominantly vertical in
wrist coordinates, but only 0.0368 m optical-depth difference. The old guard
treated optical-depth ordering as proximity, incorrectly rejecting the high
counter geometry. `2e988c71` measures distance to the complete finger-marker
closing segment in 3D, retaining the same 6 cm margin. Near-finger and between-
jaw targets still reject; this remains target-local clearance, not full-scene
collision planning. Focused tests: 68 pass; expanded offline suite: **489 pass,
4 skip**. Exact frozen-fixture retry `20260913_212148_9413e0` under
`~/runs/emet/open-sink-pear-aperture-3d` passes aperture adjustment (marker span
0.164 to 0.141 m, observed target extent about 0.097 m), then fails pregrasp
wrist visibility: **F/F**, no closure. The cached wrist image
`grounding-22788d7f82ca42689b9a05ac80cf92f9.png` shows cabinet fronts; the pear
lies above the frame. `run()` used the absolute vertical target offset when
choosing wrist pitch, mirroring above-pivot targets downward. `98dbfee7` keeps
the vertical sign; the existing below-pivot calculation is unchanged. A
run-level regression fails before the fix for an above-pivot target and passes
afterward, with a lower-target control. Focused suite: **70 pass**; expanded
offline suite: **491 pass / 4 skip**. Live signed-height retry
`20260913_213256_6271da` on `24c59668`, under
`~/runs/emet/open-sink-pear-signed-height`, exposes the pear and passes four
wrist semantic/geometry checks, but still ends **F/F before closure**. The
pregrasp remains below the counter surface; only the top of the pear is visible.
On the fifth frame SAM's proposed measured surfaces miss the pear and Qwen
selects a counter-edge patch. The 3D association gate correctly rejects it.
The actual RGB and all three final candidate images were manually inspected;
do not describe this as a valid target rejected by an overstrict threshold.

`516b3619` constrains the separated pregrasp to target height or above and
horizontal/downward insertion, retaining the same standoff normalization and
IK guards. This addresses below-support viewing, not full-scene collision
planning. Focused tests: **71 pass**; expanded suite: **492 pass / 4 skip**.
Same-fixture retry `20260913_214307_f3afc3`, under
`~/runs/emet/open-sink-pear-level-pregrasp`, finishes **F/F before closure**.
The full pear is now visible above the counter. In the final wrist frame
(`grounding-cff79259592347bcbb3ff81b2adcd42e`), SAM proposes both pear and baguette,
but Qwen selects the baguette; the final association guard correctly rejects it.
The RGB and both candidate panels were manually inspected.

`5fddc33d` applies the existing whole-surface 3D association check **before**
Qwen selects among tracked-object proposals. It neither changes the association
threshold nor trims proposals to manufacture a passing mask. Qwen must still
verify the retained surface, and the final association guard remains. Ordinary
search proposals are unchanged. Focused tests: **78 pass**; expanded offline
suite: **495 pass / 4 skip**.

Offline real-model replay `20260913_215320_069caf`, under
`~/runs/emet/pear-tracking-filter-replay`, tests the clear and occluded saved
frames with pear and absent-yellow-banana queries. The clear frame retains the
pear and rejects the baguette; Qwen accepts pear and abstains on banana. The
occluded frame retains one fragment and rejects two patches; Qwen abstains on
both queries. The retained clear-pear panel was manually inspected. This is
four diagnostic queries on two development frames, not held-out acceptance or
proof that association alone excludes every background fragment.

Frozen physical retry `20260913_215902_119e0e` on `5fddc33d`, under
`~/runs/emet/open-sink-pear-associated-selection`, reaches closure after tracked
approach, but finishes **physical F/F**. The robot lifts the pear briefly, then
tips and loses it. Sampled base roll reaches about 10 degrees, the object rises
at most 6.7 cm without a stable accepted pickup, and its final reconstruction
shows it on the counter. Sampled contact reconstructions during the lift show
gripper/object and base/floor contacts, not an arm/counter collision.

Two independently actionable bugs are exposed:

- The extraction requests lift **1.144 m**, beyond the actuator's **1.1 m**
  limit, and times out. `b7e030de` uses the existing conservative 1.0 m operation
  ceiling, prefers 30 cm extraction when feasible and requires at least 10 cm
  available lift travel before closing. Changed post-closure feedback cannot
  cause a lowering command. This remains an adapter motion limit, not a proof
  of collision-free extraction or successful retention.
- The generated pear weighs **4.570 kg**. The compatibility helper adds shell
  inertia to unspecified meshes, changing density from volumetric to surface
  interpretation ([MuJoCo reference](https://mujoco.readthedocs.io/en/3.3.1/XMLreference.html#body-geom-shellinertia)).
  Restoring the authored mesh modes with the same generated geometry/densities
  yields **0.07874 kg**. `926225b1` pins RoboCasa's original compiled body masses,
  COMs and inertias before adapting mesh XML, rather than assigning lighter
  object-specific densities. Full generation test `20260913_220929_746dde`
  exposed zero-mass markers gaining inferred mass; `3492cdeb` preserves those
  too. The test compares all surviving source bodies, not only the task target.

Generation/export retry `20260913_221317_3f6121` **passes**, comparing all
surviving source-body dynamics through adaptation and saved XML reload. It
also creates `~/runs/emet/open-sink-frozen-dynamics-20260913`, a separately
labelled same-geometry fixture with the three movable objects' authored inertia
modes restored and pinned: pear 0.07874 kg (was 4.570), baguette 0.07957 kg
(was 6.342), squash 0.06709 kg (was 7.119). Its manifest records source-asset
hashes and each changed mesh mode. Geometry, densities, robot dynamics, friction
and solver settings are checked against the old fixture; non-object kitchen
inertias retain their archived values. Fresh production generation preserves
all source bodies instead. The old scene and all failures remain intact.

Learned corrected-dynamics retry `20260913_221528_c641e9` on `b7e030de`, under
`~/runs/emet/open-sink-pear-restored-dynamics`, completes **physical pickup T /
placement F**. Pickup is verified at 61.834 simulated seconds (window starts
60.834). Final reconstruction was manually inspected: the pear remains between
the fingers, elevated and without other support contact. Maximum absolute
base roll/pitch over the trace is 0.20/2.84 degrees, not the earlier large tip.
The run combines the bounded-lift fix with restored object dynamics, so this
is not a one-factor ablation.

The agent stops before placement because carry retraction times out. The arm
is still retracting at about 0.1 m per simulated second, from 0.417 m toward
0.01 m; the final 14.94 simulated seconds took 39.48 wall seconds. Unlike
head/navigation/posture waits, `arm_to()` omitted the existing simulation-speed
deadline scaling. `b6ad49ce` applies that same bounded scale, retaining hardware
budgets, positional/settling checks and the scale cap. It also enforces the
deadline when feedback is missing, rather than looping indefinitely. Timing,
carry and protocol tests: **22 pass**; expanded offline suite: **519 pass /
4 skip** (also includes the exact extraction-clearance boundary regression).

Same-fixture learned retry `20260913_222451_3512ce` on `b6ad49ce`, under
`~/runs/emet/open-sink-pear-carry-deadline`, ends **F/F before closure**: lateral
servo oscillation consumes the local deadline. This failed repeat is retained;
the previous pickup is not evidence of robust room manipulation. Cached wrist
geometry shows mostly stable object centers once the arm has inserted, while
the base cycles between roughly 0.026 and 0.055 m and lateral errors alternate
around ±13 mm. Frames have distinct acquisition timestamps, not repeated IDs.

A client regression reproduces premature completion: the geometry preset's
**20 mm** base tolerance reports a **13 mm** correction complete on stationary
old feedback, before execution. `3dfcc5f4` changes only the experimental geometry
preset's base tolerance to **5 mm**, matching native translation control and
the existing servo closure precision. Coarse/default presets remain unchanged;
hardware unable to meet this precision must fail, not silently loosen the gate.
This addresses a demonstrated contract mismatch, not a proven explanation of
every mask/servo fluctuation. Focused motion/aperture tests: **29 pass**;
expanded offline suite: **520 pass / 4 skip**. Same corrected fixture retry
`20260913_223356_cf593f`, under `~/runs/emet/open-sink-pear-precision-contract`,
fails **F/F before closure** on the first fine correction: the old client
declares no motion at a measured base speed of **9.97 mm/s**, while the target
is still being approached. Tightening completion exposed this separate
instantaneous low-speed failure heuristic.

`096ab093` tracks improvement in tolerance-normalized goal error over the
existing bounded, simulation-scaled stall window instead. Slow convergence is
not a stall; stationary nonconvergence and the overall deadline still fail.
Position and stopped-motion acceptance are unchanged. Focused motion/carry/
protocol tests: **25 pass**; expanded offline suite: **522 pass / 4 skip**.
Simulator-only control `20260913_224425_cd1725`, under
`~/runs/emet/arm-precision-contract-control`, passes **9/9 motions**: initialization,
six small base corrections (including ±13 mm changes), extension and retraction.
All reported base errors are below 5 mm (maximum 4.88 mm); full retraction takes
7.05 wall seconds. This is an **empty-gripper** control on the separated-neighbor
tabletop NoSlip=10 fixture, using the real bridge/client and no VLM. It does not
test payload retention or establish room manipulation acceptance. The script,
source SHA, simulator log and measured commands/results are archived.

Learned room retry `20260913_224729_017788` on `096ab093` ran under
`~/runs/emet/open-sink-pear-progress-contract`, using the same corrected-inertia
visible room, NoSlip=0, Qwen int4 and 5 mm geometry preset. **Physical pickup T /
placement F** (pickup at 64.646 simulation seconds). Fine alignment and carry
retraction complete; the agent searches for the sink and attempts placement,
then correctly abstains when the payload is absent from its verification view.
Final and last-contact qpos reconstructions were manually inspected: the pear
has fallen onto the floor, not into the sink. This is not a pure recognition
failure and tool/process exit zero is not physical success.

The retained trace shows downward object-to-gripper drift during stationary
periods, not just turns: relative Z is about -12 mm at 72 s, -20 mm at 80 s,
and -41 mm at 105 s. Last gripper contact is at 105.038 s; by 106.87 s the pear
contacts the floor. No release was commanded. The active numerical-physics
hypothesis is steady contact creep under NoSlip=0, to be checked rather than
assumed. Private continuation `20260913_225816_7f7310` holds the recorded 72 s
actuator controls for 50 simulated seconds under NoSlip=0 and 10, serially,
with the same corrected scene and initial state. Its output/script are under
`~/runs/emet/pear-stationary-hold-control`. This is an evaluator-side causal
diagnostic, not another learned episode. Room placement and Stage C remain
unpassed; no production contact defaults have changed.

That fixed-control continuation completed both rows. **NoSlip=0 drops the pear**
(maximum object-to-gripper displacement 1.063 m; final object Z 0.026 m), while
**NoSlip=10 retains it** (maximum relative displacement 2.644 mm over 50 s;
final object Z 1.081 m). Maximum base translation is 0.236 mm and 0.0021 mm,
respectively. Thus navigation is not necessary to reproduce this particular
drop, and solver choice changes retention for the same saved grasp. This does
not establish that every failed grasp is numerical or that hardware retention
is safe. The replay freezes actuator controls instead of rerunning perception
or the complete live controller; it is not an independent task result.

Fresh variant `~/runs/emet/open-sink-frozen-dynamics-noslip-20260913` changes
only `noslip_iterations` from 0 to 10. Its archived construction script checks
**454 compiled model arrays** for exact equality and all other solver options
for equality; hashes preserve both scenes. No friction, actuator gain, object
density, robot geometry, initial pose or success threshold changes. Learned
retry `20260913_230112_fb9a85` uses the same frozen `096ab093` policy under
`~/runs/emet/open-sink-pear-progress-noslip`. Keep its explicit numerical-physics
row separate from native NoSlip=0 results and from the earlier tabletop panel.

The solver-only learned retry finishes **F/F before pickup** (106 wall seconds).
It finds the pear from a closer base pose and correctly rejects all separated
pregrasp IK candidates. This run therefore does not test live carry retention.
Offline real-kinematics reproduction at the observed 0.58 m horizontal target
distance and 0.966 m height requires **-29 mm arm extension** for a horizontal
20 cm separated pregrasp. The issue is a viewing-pose/manipulation-workspace
mismatch, not evidence against the saved-grasp contact control.

`19691f45` derives the minimum grasp distance from the robot's retracted,
horizontal-wrist FK plus the unchanged 20 cm pregrasp separation. A too-close
grounded pickup asks the existing collision-checked voxel navigator for a base
pose within hard radial bounds, preferring the configured 30 cm separation.
It does not directly command a blind backup, lower the standoff, use simulator
object coordinates, or change ordinary find/EQA sampling. Path/footprint and
measured-position checks remain required; final arm-facing alignment reacquires
the target before manipulation. This is workspace feasibility, not a full
arm collision planner. Tests cover unreachable/blocked ranges, unchanged find
behavior, navigation failure/no progress, and the actual kinematic failure;
**537 offline tests pass / 4 skip** (including the frontier-sampling suite).
Serial same-fixture retry `20260913_231231_143de9` runs frozen `19691f45` under
`~/runs/emet/open-sink-pear-workspace-noslip` with explicit NoSlip=10.

It finishes **pickup T / placement F** in 270 wall seconds, pickup at 63.734 s.
The pear stays held through the entire remaining trace: all 408 samples from
64 s through 105.548 s have gripper contact, no other contact, and maximum
relative displacement is 6.784 mm. Final reconstruction confirms it remains
between the fingers. Placement is never released: the learned sink view is
valid, but the chosen placement point is **1.017 m** from the base and the
operation rejects it as too far. The pickup relocation path was not entered
in this episode; its viewing pose already supported the separated pregrasp.

Review caught a configuration error in the newly added pickup relocation:
the real preset's nominal end-effector planning radius is **0.55 m**, not the
synthetic test fixture's 0.8 m. An object's radial approach range must include
its pregrasp separation at **both** ends. `9f0a895b` corrects that accounting
and loads the actual preset in the real-kinematics regression. No live success
is attributed to the previously unexercised path.

`67c4c90b` adds the corresponding distant-receptacle approach before placement:
retain carry posture, navigate within the existing radius/placement-step bounds
using the shared voxel planner, check measured arrival, then turn and reacquire.
Nearby placements do not add navigation; planner failure or no measured arrival
stops before manipulation. The existing local reach and observed-release gates
are unchanged. The class's step-size default now matches `configure` (0.25 m).
**540 offline tests pass / 4 skip**. Same-scene/model/preset retry
`20260913_232336_224534` runs frozen `67c4c90b` under
`~/runs/emet/open-sink-pear-place-workspace`; this remains a development diagnostic,
not the frozen cross-room acceptance panel.

That retry completes **pickup T / placement F** in 246 wall seconds (pickup
at 60.134 s). It preserves the payload but the placement navigator finds no
valid base pose within the requested 0.55–0.80 m range of the **sink center**.
It stops without release or a collision-check bypass.

CPU-only audit `20260913_233311_c8ea20` reconstructs the map from the last saved
12 agent RGB-D observations (one observation before final reacquisition). This
is **not the exact live grid cache**. The cached accepted sink mask/depth is
matched to the saved observation before extracting world support. The nearest
footprint-valid reachable pose is 0.814 m from the sink center, outside the
unchanged range; the existing placement-point calculation yields a point with
a valid base pose at `(0.90, -1.10)`, **0.701 m** away. Thus the reconstructed
map reproduces center-target rejection but admits the actual placement surface
without changing safety settings. Results, map figure, script and the 37 MB
agent checkpoint are retained at `~/runs/emet/pear-placement-map-audit`.

`b2c80dbc` shares the existing placement-point calculation between navigation
and local placement, accepting a fresh grounded point cloud before creating the
temporary receptacle instance. No sink-specific coordinates or broader reach
threshold are introduced. Large-receptacle regression and the broad suite pass
**541 tests / 4 skip**. Same-fixture learned retry `20260913_233727_19e44f`
runs frozen `b2c80dbc` under `~/runs/emet/open-sink-pear-placement-surface`.

That run completes **pickup T / placement F** in 306 wall seconds. Navigation
now reaches the planned pose `(0.901, -1.098)`, but fresh sink reacquisition
sees only the far/bottom portion of the basin. Recomputing the local goal from
that partial cloud moves it to 0.981 m away, beyond reach. No release occurs.
The saved head image and selected mask were manually inspected: this is a
partial view of the correct support, not proof that the support moved.

`7627d755` retains the planned placement point and original support-clearance
height **only after fresh semantic verification and the existing 80%/5 cm
3D association pass**. Fresh points remain fresh; old geometry is not inserted
into the reacquired instance. This assumes a static receptacle, not dynamic
tracking or collision-free empty-space certification. A new query clears the
reference, and absence/ambiguity/inconsistent association still aborts.
**545 offline tests pass / 4 skip**.

Its first live trial `20260913_235140_680b61`, under
`~/runs/emet/open-sink-pear-placement-reference`, is **pickup F / placement F**
in 145 wall seconds: it stops before pickup and does not exercise placement.
Manual review of wrist record `grounding-b89882e7244d4b559b357e963d42bc6c`
shows a clear pear and a good candidate mask, but Qwen rejects its appearance
as a potato. This verifier false negative remains in the results; no identity
threshold or prompt was changed. Unchanged repeat `20260913_235819_32930b`
uses the same frozen source, model and NoSlip=10 fixture under
`~/runs/emet/open-sink-pear-placement-reference-repeat`. A successful repeat
would test placement, not erase the failed run or establish repeatability.

The unchanged repeat completes **pickup T / placement F** (pickup 64.136 s).
The static reference survives navigation and fresh support association, and
placement starts 0.705 m from the base. Four observed vertical corrections are
−70.1, −18.9, −7.9 and −8.9 mm; the last misses the unchanged 5 mm release gate.
The robot retains the pear and does not release. Manual final reconstruction
shows it still held above the basin. A private, non-integrating reconstruction
of the recorded final state identifies **counter ↔ gripper-body contact** at
the front edge, approximately `(0.930, −0.570, 0.920)` m. This is not a reason
to loosen tolerances or command more downward force: the selected release pose
does not clear the full gripper. Trace/scene hashes and reconstructed contact
pairs are archived under `~/runs/emet/pear-placement-clearance-audit`.

Same-source neighboring tabletop control `20260914_000218_79fce1`, under
`~/runs/emet/tabletop-placement-reference-control`, also fails **F/F** before
pickup. The new horizontal-only workspace check relocates the robot to a
different side of the low target; later wrist tracking overflows its candidate
budget. This does not isolate the overflow's cause, but it exposes an unwanted
change to a previously accepted approach. The old `2e988c71` T/T control cannot
be cited as no-regression evidence for `7627d755`.

Candidate `787acb6f` shares a non-commanding pregrasp IK calculation between
the workspace check and execution. A reachable angled pregrasp no longer
triggers relocation solely because the target is inside the horizontal bound.
It also replaces the missing `link_gripper_s3_body` fallback and virtual 35 cm
local-Z offset with the active URDF's actual `link_wrist_pitch` pivot. The
invented pivot could point signed-height aiming upward at a low target.
The existing angular bias, 20 cm minimum standoff, navigation and release gates
remain unchanged. Real-kinematics low/high-target controls and the broad suite
pass **564 tests / 4 skip**. Learned tabletop retry `20260914_001546_f0b89d`
runs frozen `787acb6f` under `~/runs/emet/tabletop-actual-pregrasp-control`.
Sink gripper-clearance planning remains unresolved, not hidden by that retry.

The `787acb6f` tabletop retry completes **physical pickup T / placement T**
in 265 wall seconds (pick 73.010 s, final stable placement 172.188 s simulation
time). It does not enter the unnecessary pregrasp relocation path. Final close
and top-down reconstructions were manually inspected: the cylinder rests on
the cube after gripper withdrawal. This is one restored neighboring control,
not the full six-case panel, room acceptance or proof of general robustness.
Unchanged repeat `20260914_002232_e419bd` also passes **T/T** under
`~/runs/emet/tabletop-actual-pregrasp-repeat`, still frozen on `787acb6f`
(pick 71.480 s, final stable placement 170.388 s simulation time). Its final
close-up was manually inspected. Thus the repaired candidate has **2/2** on
this neighboring fixture; earlier-source failures remain reported above.

Before adding mesh-based clearance checks, inspect the actual model assets:
the client-generated `config/urdf/stretch.urdf` uses RE1V0 dex-wrist geometry,
whereas native simulation uses SE3/SG3. Private audit
`~/runs/emet/stretch-embodiment-model-audit` compares six recorded pear states,
without stepping physics, to the legacy and a separately generated SE3 URDF.
At these near-horizontal wrist poses, legacy grasp-origin errors are
5.61–5.63 mm; the SE3 URDF still differs by 4.77 mm. This is not a full pose-grid
calibration and does not establish either model as a drop-in replacement.
Both URDFs, trace/scene/source hashes and the audit script are archived. The
observed counter contact remains real; neither that small FK error nor the
model-name mismatch alone explains all failures. Collision-envelope support
must first select and validate the embodiment's actual geometry rather than
reusing legacy meshes or changing hardware defaults silently.

Review also exposed missing head-camera calibration in the grounding cache.
The review branch now saves camera intrinsics, pose and base pose from the
original captured frame (detection backends can discard these fields). Missing
values remain null, never invented calibration. This is diagnostic-only and
is not included in the frozen `7627d755` runs.

Neighboring tabletop regression `20260913_212612_8011e4` runs the original
NoSlip=10 fixture on `2e988c71` (before the signed-height change) with the same
tracked-narrow agent, under `~/runs/emet/tabletop-after-room-fixes`. It tests
the grounded-reference and 3D-aperture fixes against the earlier accepted task;
it is not a rerun of all six Stage B cases. It completes **physical pickup and
placement T/T** (pick 78.620 s, final stable place 166.088 s simulation time).
The final object reconstruction was manually inspected: the cylinder rests
on the cube after release. This is one neighboring regression control, not
general clutter acceptance or room/TAMP success.
This is not the eight-case Stage C panel; visible/search starts and the Molmo
counterpart remain to be frozen.

Panel history: v3 passes both original
repeats but times out during clearance-scene navigation. Panel v4 repairs that
transport and passes both clearance repeats, then exposes a delayed grasp slip
in mirrored clutter. Panel v5 on `b781c4d8` stops before pickup on its first
mirrored case: detector proposals overflow, then Qwen-box/SAM recovery yields
no supported surface. Five cases are unrun. This does not exercise the new
verified-pose handoff or tighter 5 mm final grasp gate.
Individual development successes and fourteen passed navigation moves are not
a substitute for the complete gate.

V5 artifacts: `~/runs/emet/manipulation-panel-v5-20260913`, managed job
`20260913_180405_4ed3fa`. Wrist frame
`grounding-857eb2e5483e45848ecd3619b4e8f07b` clearly shows the cylinder.
Qwen's normalized box `[420,560,500,687]` maps to pixel bounds
`[201.6,151.2,240,185.49]` on the 480×270 image, omitting the lower and right
extent. Final semantic verification is not reached. Cached mask/depth diagnosis
must distinguish bad prompting from segmentation failure before another panel.

Cached diagnostic `20260913_181328_67b30b` reproduces the recorded box: SAM's
highest-score mask contains only 51 scattered pixels, none forming a 25-pixel
depth-connected surface. All 51 have valid depth; this is not a missing-depth
failure. A manually bounded full-object box (diagnostic only, never policy input)
produces 1,454 connected pixels on the cylinder. Matched prompt check
`20260913_181543_bccba6` on the failed and preceding wrist images yields 1/2
positive recoveries with the original box prompt and 2/2 with the existing
whole-object prompt; both reject 2/2 absent-banana queries. However, the latter
still omits the cylinder's lower extent on the failed view. Do not promote a
partial identity-positive mask as complete grasp geometry or claim generalization
from these two development images. Artifacts:
`~/runs/emet/wrist-whole-box-check-20260913`.
Targeted live retry `20260913_181756_ed9bbb` retains frozen `b781c4d8` and the
unchanged setdown preset but also fails before pickup (physical false/false).
The recovery box again misses the full target; Qwen correctly rejects SAM's
106-pixel patch left of the cylinder. This is a semantic rejection of a bad
proposal, not v5's empty-surface rejection. It is not a new panel and cannot
erase v5's failure or validate the grasp handoff. Focused offline contracts:
63 passed.

### Observed-bound tracking candidate (not accepted)

`e47c0fe8` adds the separate `query_geometry_tracked_pilot.yaml` preset. During
known-target wrist tracking, project the previously verified world bounds through
the current calibrated camera pose and intrinsics, then use the box to prompt
SAM. No simulator labels, manual boxes, padding or new retries enter policy.
The original observed bounds are only proposals, not current visibility or a
complete object model. Qwen must verify the new measured surface and the existing
world-association gate must accept it. Invalid/off-camera projections fail closed.
Search/head proposals retain the control's YOLOE/SAM and bounded recovery path;
existing presets are unchanged. This is a shared geometry operation, with live
integration currently limited to the existing Stretch wrist-grasp adapter.

Cached check `20260913_182544_cedc8a` recovers **3/3** saved target views, including
both previously failed frames, with successful world association. All **6/6**
absent-banana and wrong-object blue-cube queries against the same projected region
are rejected. Manually inspected masks cover the visible cylinder in both failed
frames; v5 now has 1,482 connected pixels. Artifacts:
`~/runs/emet/wrist-projected-check-20260913`. These are correlated development
frames, not independent task successes or a new held-out set.
Broader offline regression: **423 passed, 4 skipped** (live simulation disabled).
Live development trial `20260913_182802_c70308` on `e47c0fe8` preserves wrist
tracking through the approach, then safely refuses closure: measured grasp error
stalls around 9–10 mm, outside the candidate's 5 mm gate. Physical false/false;
no carrying or release is exercised. Artifacts:
`~/runs/emet/mirrored-projected-tracking`.

The grasp servo rebuilt each IK goal from newly measured joints, retaining
steady tracking bias and compounding wrist sag. Candidate `2ef5d49f` instead
integrates fresh visual error into a persistent command reference with fixed
orientation, saturating the requested displacement at the existing 5 cm measured
step limit. The three-step no-progress stop and 5 mm closure gate remain intact.
Reset clears both reference and stall history. 114 focused grasp/place tests pass,
including a steady-bias regression. Frozen live check `20260913_183357_e4f4cc`
uses the same tracked preset and fails safely before pickup: the last verified
wrist frame is valid, but arm extension stops at approximately 0.1275 m against
a 0.1684 m command. Reconstructing the final saved qpos with MuJoCo forward
kinematics shows `rubber_tip_left` contacting the neighboring cube (`object1`),
with about 1.7 mm penetration. The target remains on the table and the robot
remains upright. This is an obstructed approach, not missing identity or a reason
to relax arm completion. Artifacts: `~/runs/emet/mirrored-tracked-reference`.

Explicit aperture ablation `80ea85c3` adds
`query_geometry_tracked_narrow_pilot.yaml`, changing only the observed marker-span
margin from 6 cm to the existing allowed minimum of 4 cm. Narrowing occurs only
at the existing depth-verified standoff, with unchanged marker checks, closure
gate, force limits and physical scoring. This is development tuning, **not**
collision-aware grasp planning or a hardware recommendation. The 6 cm control
remains intact. Job `20260913_184054_44dbd0` completes with **pickup true / place
false**, pickup at 81.170 simulated seconds. Final observed grasp-center error is
`[0.586,0.436,0.545]` mm (0.911 mm norm), below the unchanged 5 mm gate. The object
clears the table, then gradually creeps forward in the fingers: approximately
4.1 mm grasp-frame X at 86 s, 19.8 mm at 120.236 s, and 21.0 mm at last contact
122.276 s. It drops during a subsequent turn, but drift precedes that turn.
Artifacts: `~/runs/emet/mirrored-tracked-narrow`. This is not manipulation
acceptance. Focused tests: 40 passed; the preceding broad code suite passes
426 tests with four simulation skips.

Captured-state control `20260913_184804_c62875` compares 50-second stationary
holds at original and level wrist pitch from 86 s and 98 s checkpoints. Both
86 s holds retain the object but creep approximately 13 mm along grasp X. From
98 s, the original-pitch hold drops after 45.14 s without navigation; leveling
retains it but still creeps. Grip force/contact physics are unchanged.

The installed MuJoCo 3.5.0 model uses Newton, elliptic cones, `impratio=20` and
`noslip_iterations=0`. Its [modeling guide](https://mujoco.readthedocs.io/en/3.5.0/modeling.html#preventing-slip)
documents gradual soft-contact slip and the optional NoSlip postprocessor.
Solver-only control `20260913_185132_9b7aa9` repeats the 98 s stationary hold
with NoSlip=10: retention for 50 s and only approximately 31 μm relative drift.
The paired open-gripper negative drops after 0.688 s; this is not attachment.
These controls support a solver-induced contribution, not hardware grasp safety.
Both diagnostic scripts are archived beside their managed job logs.

`a4a2efe3` adds an explicit NoSlip mirrored-scene wrapper and configuration;
compiled-model tests confirm unchanged geometry, friction, actuator parameters,
initial poses and other solver settings. All three fixture tests pass. Default
scenes remain unchanged. Live row `20260913_185544_7bbd8f` keeps the tracked-narrow
agent fixed but stops before pickup on an obstructed approach. Final saved-state
reconstruction shows `rubber_tip_right` contacting the target itself; arm reaches
0.1733 m against a 0.1960 m command. The wrist mask remains valid. Replay figures:
`~/runs/emet/mirrored-tracked-noslip/replay`. The prior 12 mm hard-coded lateral
alignment threshold was inconsistent with the candidate's 5 mm closure gate.
Fix `e8b4fbcc` uses the configured grasp tolerance for both, preserving the default
12 mm behavior. Focused tests: 41 passed. Retry `20260913_190144_6ae8b9` stops on
the no-progress guard during lateral alignment, before pickup. No fingertip
contact is present in the final reconstructed state. Base commands can be
accepted inside the client's 2 cm joint tolerance while the finer motion has
barely progressed. Recorded wheel references near 0.525 demand about 31.5 N·m
(`gear=3`, `kv=20`), below the unchanged 35 N·m modeled joint friction.

Candidate `83d10c4e` derives Coulomb-friction feedforward from the loaded wheel
model rather than changing friction, minimum speed or task budgets. A common
speed fraction reserves actuator-limit headroom; reference slew and unbiased
zero/cancel behavior remain. Native small-wheel tests reproduce the uncompensated
dead zone and verify compensated velocity in both directions and with negative
gearing; 44 wheel/translation tests pass. Their physics fixture uses the native
implicit integration and rotor inertia; an initial overly stiff fixture failed
numerically and is not counted as a passing experiment.
Serial route job `20260913_191358_296e9f` passes **14/14** precision moves on
default physics (max XY 1.244 cm, yaw 0.01092 rad) and **14/14** on NoSlip
(max XY 1.223 cm, yaw 0.01184 rad). All receipts succeed without corrections.
Health still lacks full body telemetry; this is a bounded route diagnostic,
not the five-repeat navigation acceptance run or hardware validation. Broader
offline suite: **433 passed, 4 skipped**.

Physical development retry `20260913_192032_dd8d83` on frozen `83d10c4e`,
`query_geometry_tracked_narrow_pilot.yaml` and the explicit mirrored NoSlip scene
passes **pickup true / placement true**, at 81.578 and 149.288 simulated seconds.
Final observed grasp error is approximately 3.53 mm, within the unchanged 5 mm
gate. Final trace records support contact without gripper contact after retreat;
manually inspected qpos reconstructions show the cylinder held at pickup and
resting on the blue cube after release. Artifacts:
`~/runs/emet/mirrored-noslip-friction`, including `replay/manifest.json` and
picked/final object, overview and top-down images. This is one development pass,
not six-case acceptance or evidence of long-horizon manipulation.
Do not pool this numerical-physics ablation with earlier panels; any comparative
task evaluation must freeze matching physics.

### Frozen panel v6 and camera calibration repair

Job `20260913_193318_c4008b`, frozen source `b7441826`, uses the tracked-narrow
preset with explicit NoSlip wrappers for all three predeclared fixtures.
Results: original **2/2 physical pickup and placement**, separated-neighbor
repeat 1 **false/false**, three remaining cases **unrun** after the stop gate.
Original process durations are 231/236 wall seconds; pickup occurs at
76.274/84.026 simulated seconds and final placement at 158.088/162.988.
Both final object reconstructions were manually inspected. Artifacts:
`~/runs/emet/manipulation-panel-v6-20260913`.

The separated-neighbor failure is before closure: arm extension stalls at
approximately 0.180 m against a 0.202 m command. The private trace and final
qpos reconstruction show the right rubber fingertip contacting the target,
not the neighboring cube. The final wrist surface remains semantically valid
and spatially associated. Frame `grounding-a29ba1c4f88447e0833765635cc3b992`
shows the cylinder biased toward the right pad despite near-zero computed
lateral error. Comparing published camera-relative grasp geometry against the
rendered MJCF reveals a lateral displacement of **8.676 mm**, plus approximately
0.770/3.686 mm on the other camera axes. That lateral mismatch exceeds the
candidate's 5 mm servo gate. The server used URDF transforms queried after
rendering, not the actual rendered sensor/grasp geometry.

Fix `ef533ed3` captures one native state under the physics lock, renders all
RGB/depth views from that snapshot, and publishes its camera and grasp-link
world poses. The head pose includes the existing clockwise image rotation.
Only robot/sensor geometry enters observations; object state remains private.
Missing acquisition poses suppress the message rather than falling back to
later FK. Policy, contact offset, aperture, tolerance and physics are unchanged.
Twelve focused tests cover nonzero base/wrist poses, snapshot immutability,
sync/threaded RGB-D consistency and no later-FK fallback; the broader suite
remains **433 passed, 4 skipped**. This does not claim that separately queried
joint feedback or the state-only FK stream is acquisition-synchronous.
Live failed-case retry `20260913_195231_e4d996` uses the same clearance NoSlip
row under the repaired source, artifacts `~/runs/emet/clearance-camera-snapshot`.
It passes **physical pickup and placement**, at 80.966 and 171.088 simulated
seconds (255 wall seconds total). The final observed grasp error is 3.31 mm.
Saved wrist geometry now matches the native camera-relative grasp transform;
the final object reconstruction was manually inspected on the designated cube.
This development retry is not pooled with v6. Fresh v7 uses frozen `ef533ed3`
and the identical six-case order/settings declared for v6; only the snapshot
repair differs from v6 code. Stop on the first physical failure again.

#### Visual audit of the calibration failure and repair

These are unmodified saved images, not generated illustrations. The two wrist
views come from different live executions and different approach depths; they
illustrate the observed failure/recovery, not a pixel-matched causal comparison.

| Failed v6 clearance grasp (`b7441826`) | Successful failed-case retry (`ef533ed3`) |
| --- | --- |
| ![Cylinder against the right pad before closure](stretch-wrist-calibration-before.png) | ![Cylinder between the fingers at the verified grasp](stretch-wrist-calibration-after.png) |

The first image is agent wrist frame
`grounding-a29ba1c4f88447e0833765635cc3b992`; the second is
`grounding-a58a9e61adeb48519adee072345d647b`. Their SHA-256 hashes are
recorded in the [figure provenance manifest](stretch-camera-snapshot-figures.json);
original RGB-D, masks, prompts and poses
remain in the artifact directories above.

![Final cylinder resting on the designated cube after release](stretch-camera-snapshot-placement.png)

The final view is an evaluator-only **qpos reconstruction** from
`clearance-camera-snapshot/replay/final-object.png`, not an agent camera view or
another execution. Its replay manifest and independent physical trace are in
the same run directory. One successful development trial is not panel acceptance.

### Frozen panel v1 (stopped, not accepted)

Job `20260913_120543_168dcb`, source `6e54ddd1`, uses the contact/aperture preset
and predeclared original/separated/mirrored fixtures twice each, stopping on
failure. Artifacts are under `~/runs/emet/manipulation-panel-v1-20260913`.
Each completed case retains physical scoring and robot-visible replay figures.
Original repeat 1 passes (pickup 64.136 s, final placement 140.388 s); original
repeat 2 passes pickup at 63.734 s but loses finger contact near 109.63 s.
The latter stops before release when current visual grounding cannot find the
held object. Both agent processes exit zero; independent scoring rejects the
second case. Four unrun cases are **pending**, not failures or successes.

Unlike the older boundary chatter, this trace shows a progressing arc followed
by an abrupt in-place turn. The base stays upright; payload-relative drift
grows before ejection. Private replay `20260913_121747_00b35f` compares recorded
wheel targets with slew-limited targets from three measured qpos/qvel/warm-start
checkpoints. Recorded targets reproduce ejection from 107 s and 109 s (peak
object/gripper distances 0.856 and 0.798 m); the 108 s checkpoint retains it.
At 8 transmission-rad/s², all three retain it within 1.5 cm. This establishes
a local counterfactual, not learned task acceptance. The production candidate
now limits wheel-joint references to 8/3 rad/s² (independent of gearing), with
explicit cancellation bypassing the ramp. A captured 109 s checkpoint tests
both the failing recorded controls and the production profile. Grip force,
contact physics, arrival tolerances and release thresholds are unchanged.
Native regression `20260913_122750_28088e` passes both old-target negative and
production-profile positive controls (2 tests, 9.02 s). Wheel/cancellation/phase
tests pass (39). The first live route `20260913_123021_2f4538` on `663877a7`
then catches a side effect: ten ordinary moves pass, but coupled goal
`[0.4, 0, pi/2]` overshoots by 3.23 cm and stalls. Cancellation confirms rest.
The generic feedback profile assumes 1 m/s² braking, incompatible with the
new actuator ramp. `e33bb5d1` adds an inherited native Stretch profile
(v=0.09 m/s, w=0.5 rad/s, braking a=0.06 m/s² and alpha=0.4 rad/s²), leaving
other robot adapters and all tolerances/deadlines unchanged. An integrated
wheel-reference test reproduces the old handoff overshoot and checks the repair.
Full retry `20260913_123817_b165a0` passes **14/14** moves: max XY error
1.363 cm, max yaw error 0.02042 rad, zero corrections. Health remains
`incomplete_telemetry`, not full hardware acceptance.
Original learned-task retry `20260913_124152_159563` on `e33bb5d1` fails
**before pickup** (physical false/false, process zero, 87 s wall). The wrist
image visibly contains the cylinder, but the proposal/verification pipeline
rejects identity at the final approach (last accepted error about 1.7 cm).
This run does not exercise carrying. Diagnostic `20260913_124712_3cb5b1`
reproduces the cause from cached images: the cylinder contributes one measured
component, but a second proposal covering the gripper contributes eight. The
combined budget overflows **before Qwen verification**. Both proposals overlap
the target's local bounds, so proximity filtering alone cannot safely remove
the gripper. The failure audit previously overwrote the inner grounding reason
with a generic identity error; the repair now retains that reason.

The separate `query_geometry_recovery_pilot.yaml` preset opts into one Qwen-box
→ SAM alternative on the **same frame**, only for empty/overflowing proposal
geometry. Semantic rejection and invalid geometry do not trigger retries. The
ordinary final surface verification, eight-surface budget and post-selection
world association remain mandatory. The original contact/aperture-only,
detector-only and Qwen-box controls are unchanged.

Cached-model setup `20260913_125717_a45548` failed before inference because
the replay command omitted the live driver's SDPA permission. Matched int4/SDPA
retry `20260913_130039_59db80` on `eeec07fd` passes both visible-cylinder views
and rejects both absent-banana queries (**2/2 positive, 2/2 negative**). The
formerly rejected frame exercises recovery; the earlier good frame does not.
Both cylinder masks satisfy the original world-association gate (6,621 and
6,103 pixels). Saved prompts, responses and masks:
`~/runs/emet/wrist-proposal-recovery-sdpa`. These four fixed-image checks are
perception diagnostics, not cross-object generalization or manipulation passes.

Live original-task retry `20260913_164029_e59ea7` on `eeec07fd`, with the
separate recovery preset and repaired CHAT loop, finishes **pickup true / place
false**. Pickup is independently verified at 71.072 simulated seconds; the
payload survives transport. Final observed placement error is approximately
`[4.35, 2.04, 2.18]` mm and the controller completes, explicitly without physical
verification. The private trace shows the cylinder slides off the cube during
gripper opening (127.58–128.09 s), before arm retreat, and finishes on the table.
Artifacts: `~/runs/emet/grasp-proposal-recovery`, including robot-visible qpos
replay images in `replay/`. This is neither a placement pass nor a reason to
relax alignment tolerances. Saved-state opening-rate controls are the next
diagnostic; the fresh six-case panel remains gated.

Opening diagnostics `20260913_165057_7d6cbc` and
`20260913_165418_4dfb57` show contact sensitivity: slower is not monotonically
better. The proposed 0.02 m/s slide-reference profile retains support from all
four checkpoints (two from this run, two from the previous passing original
trial). However, 10 Hz recorded controls also retain support in the latter
comparison and **do not reproduce the live ejection**; do not call this causal
proof. The bridge's wall-clock opening loop is nevertheless load-dependent,
blocks dispatch and overshoots requested apertures. The candidate removes it
and routes open/close and relative gripper commands through the existing
physics-time, joint-limit-clamped position profiler. No new controller class,
task-specific release offset, contact-model change or longer timeout is added.
Seven gripper contract tests and 35 placement/aperture tests pass.

Fresh learned retry `20260913_165731_d3bff6`, frozen `a6fc3687`, passes physical
pickup (73.520 s) and placement (165.788 s), including retention after retreat.
Artifacts: `~/runs/emet/grasp-physics-gripper`. It follows a longer route than
the failed trial, so this is a new integrated pass, not an isolated causal A/B.
The same managed job first passes all 15 native profile/held-object regressions
(22.00 s). The broader offline suite passes 395 tests with four live-simulation
skips (5.56 s).

Frozen panel v2 `20260913_170342_d04c9e` stopped at **1/2 original-scene passes**
on `a6fc3687` / `query_geometry_recovery_pilot.yaml`; all four variation cases
are unrun. Repeat 1 physically picks and places (72.908 / 138.188 s), with a
visually audited upright cylinder after retreat. Repeat 2 physically picks at
73.520 s but tips off the cube during opening at 128.39–128.69 s, before retreat.
Thus physics-time opening alone is not a sufficient release repair.
Root: `~/runs/emet/manipulation-panel-v2-20260913`. It does not pool the
development pass above or panel v1. Room OVMM, learned TAMP and paired EQA remain
gated on the basic physical panel; the legacy oracle TAMP driver is not a
substitute for learned multistep evidence.

Supported-release diagnostic `20260913_171247_8bf793` holds each pre-opening
checkpoint's base/arm reference fixed for 1.5 s, optionally lowering the lift
reference by 5, 10 or 15 mm through the production profiler, then opens at the
same speed. Both no-lowering controls tip; all six lowered controls retain
support after eight simulated seconds. The 10–15 mm interventions leave the
object closer to the support center. Even the previously passing checkpoint
tips under this controlled dwell, illustrating sensitivity rather than an exact
replay of that live success. This supports testing set-down rather than more
opening-rate tuning. A private fixture retains one paired negative/positive
control in the native physics tests.

Candidate `46e380d7` adds `query_geometry_setdown_pilot.yaml`: 5 mm observed
bottom-to-support clearance and a tighter 5 mm vertical gate, versus the
preserved recovery row's 20 mm / 15 mm. Perception, grasp calibration, XY gate,
three-correction budget, 5 cm motion/reference bounds and physical scoring are
unchanged. Geometry near a support is **not contact/force verification**, and
hidden payload geometry or biased depth can still invalidate this approach.
No ground-truth contact feedback enters the policy. Sixty focused placement,
query and observation tests pass. Managed retry `20260913_171734_634eb7` runs
the native tests, then original and separated-neighbor learned controls
serially, stopping on failure; artifacts `~/runs/emet/grasp-near-support`.
The eight native tests pass (25.49 s), including suspended-release negative and
set-down positive controls. Original learned retry passes physical pickup at
73.622 s and placement at 167.688 s (235 s process wall time). Final observed
height error is 0.577 mm; the replay shows the cylinder upright on the cube,
with the gripper clear after retreat. Separated-neighbor control also passes:
pickup 71.582 s, placement 171.388 s (242 s process wall), final observed height
error 3.758 mm after two bounded corrections. These are development trials.
Full offline contracts: 403 passed, four simulation-gated skips (5.44 s).

Frozen panel v3 `20260913_172740_b4ca6c` uses `46e380d7` and the set-down preset.
It stops at **2/3 completed passes**, three unrun. Original repeats both pass
(pickup/place 73.010/137.088 s and 72.602/165.488 s). Separated repeat 1 picks at
73.214 s but times out before placement, retaining the object. It is a task
failure, not a release-policy trial. Root:
`~/runs/emet/manipulation-panel-v3-20260913`.

The failed waypoint combines a 40 cm drive with a 90° turn. The trace shows
continuous movement to approximately `(0.295, 0.204, -1.328)` for world goal
`(0.3, 0.2, -1.571)`, then a deadline stop, not a fallen robot or blocked path.
The old ten-second intermediate deadline is inconsistent with some coupled
moves under the conservative base profile. `6b020ecd` explicitly sets the
candidate's existing `find_phase_nav_step_timeout_s` option to 30 s, matching
the existing final-approach ceiling. The outer 600 s task cap, physical scoring
and arrival tolerances are unchanged. This **changes a declared system budget**;
do not report the row as a release-only ablation or unchanged-budget comparison.

The audit also found intermediate commands omitted `nav_policy`, bypassing the
server's freshness/progress/settling monitor. `db851c20` routes both ZMQ clients'
shared waypoint executor through the existing exploration policy. Its 7 cm /
0.15 rad arrival tolerances match the legacy native check; it adds measured
settling, stale/no-progress rejection and bounded correction. Ninety-eight
focused navigation, command, query and placement tests pass. No robot-name
conditional or automatic timeout extension is added.

Panel v4 `20260913_174607_4b2f47` is frozen on `6b020ecd` / set-down preset,
with order declared before launch: separated neighbor twice, mirrored clutter
twice, original twice. The six fixtures and scoring gates are unchanged; the
failed fixture runs first. It stops on failure and does not pool previous
panels or development controls. Root: `~/runs/emet/manipulation-panel-v4-20260913`.

V4 stops at **2/3 completed passes**, three unrun. Clearance repeats physically
pick/place at 73.622/175.188 s and 71.888/161.088 s. Mirrored repeat 1 picks at
74.030 s but loses contact near 122.2 s while wheel commands are zero. The
object migrates toward the fingertip edge before falling. Later placement
grounding correctly rejects the empty view; the red cylinder is on the floor,
not hidden behind the blue support. This is **not a VLM false negative**.

The last accepted grasp error is approximately `[3.60, -10.02, 3.07]` mm, inside
the previous 12 mm gate but appreciably shallower than the 3–4 mm residuals of
stronger grasps. The closure helper also reissues a supposedly zero approach
from measured joints, replacing a 0.1893 m arm target with 0.1780 m and resetting
wrist angles after visual verification. `b781c4d8` removes that redundant move
for geometry-servo grasps and gives the candidate a 5 mm closure gate. The
12 mm control remains the default, contact calibration and forces are unchanged,
and the existing bounded-correction/no-progress logic still applies. Unit tests
check the logged residual requests a correction, rather than closure, and that
verified geometry proceeds directly to closure before lift. This is a grasp
repair hypothesis requiring live retention evidence, not proof from tolerance
selection alone. Focused tests: 111 pass; full offline pack: 410 pass, four
simulation-gated skips (5.45 s).

Panel v5 `20260913_180405_4ed3fa` freezes `b781c4d8` and the set-down preset.
Predeclared order: mirrored twice, clearance twice, original twice, so the newly
failed fixture runs first. Same six-case scoring and stop-on-failure protocol;
no pooling of earlier source versions. Root:
`~/runs/emet/manipulation-panel-v5-20260913`.

The focused combined
suite passes **388 tests, 4 simulation-gated skips**; native carry physics and
live navigation remain separately reported above.

### Shared multi-step loop (offline validation)

CHAT previously stopped after action-only tools or selected observation tools,
and instructed every other follow-up to summarize without further tools. It
could not reliably execute an observation followed by two manipulation steps.
The repaired loop (`7a6e7541`) feeds every result back within three tool-bearing rounds,
refreshes the optional camera each round, and ends with a no-tool summary at
budget exhaustion. Structured failures/exceptions stop the remaining batch and
further tools for that turn without quitting the interactive session. Motion
and pick/place tools expose structured outcomes rather than relying on failure
wording; controller completion remains physically unverified. The prompt uses
the same contract, with no task-specific fast-reply whitelist.

Offline agent tests: **113 passed, 4 skipped** (simulation disabled). Mocked
full-loop tests cover observation→action→action, action-only continuation, batch
failure, fresh follow-up images and budget exhaustion, including a model that
still emits tools in the forced-final response. These changes were not in the
`e33bb5d1` trial and are **not learned TAMP acceptance**.

### Development history

`default_table_stretch_clearance.yaml` changes only the blue neighbor's x
position (-0.02 → -0.25 m). The original fixture is untouched; a model-equality
test checks geometry, inertia, limits and all other initial coordinates.

| Job | Source | Physical pick/place | Finding |
| --- | --- | --- | --- |
| `20260913_074731_6dfa15` | `336956a3` | true / false | Wrist reaches 0.163 m and lifts the cylinder, but loaded lift sags 2.33 cm, outside the unchanged 2 cm completion gate |
| `20260913_075853_3d9d40` | `63f7ed1b` | true / false | Lift completes; cylinder is lost during wrist folding; placement then crashes on missing shared `manipulation_radius` property |
| `20260913_081154_ea1dcd` | `4e1f7df2` | false / false | Fixed 30 cm pregrasp requests -5.8 cm extension after precision alignment; correctly rejects motion |
| `20260913_081801_d64db7` | `7e3d4d43` | true / false | Reachable 20 cm pregrasp, pickup, carry, receptacle navigation and place controller complete; released cylinder briefly touches cube then falls to table |
| `20260913_083212_06fb47` | `c6b6b761` | true / false | Cylinder slips during carrying; final visual check correctly rejects absent object and prevents empty release |
| `20260913_084739_a09908` | `424db036` | false / false | Calibrated 3D servo reaches ~1.4 cm error, but 12 mm base corrections are discarded inside the 2 cm arm-base deadband |
| `20260913_085624_0c7f62` | `a29d664d` | true / false | Fine base corrections reach ~5.7 mm grasp-center error and verified pickup; object slips during later transport; visual release check rejects the absent payload |
| `20260913_091308_642f8d` | `9714349f` | true / false | Preserved wrist/lift posture retains pickup and retraction; cylinder gradually moves toward pad edges and loses contact during base-turn braking |
| `20260913_092427_1e2e65` | `b91d761e` | true / false | First contact-offset pilot moves sideways because its calibration used the wrong diagnostic frame convention; still loses the payload |
| `20260913_093103_771bbf` | `69d8e478` | true / false | Corrected insertion direction improves initial depth but still slips during transport; closure tolerance and loaded arm displacement remain relevant |
| `20260913_093656_0b9dbd` | `292e84df` | true / false | World-frame handoff reaches the receptacle and starts placement; payload has already slipped |
| `20260913_094303_496e65` | `0a539b30` | true / false | 25 mm contact pilot retains payload through transport; loses it during a placement approach distorted by a height outlier and an unintended base turn |
| `20260913_095547_9ec2d2` | `c1468a11` | true / false | Payload remains held through all placement corrections; falsely low observed object bottom makes alignment raise the object and refuse release |
| `20260913_100708_0f6983` | `076fa458` | true / false | Registered depth fixes the camera contract, but minority mask-edge depth still drives oscillating placement corrections; payload remains held |
| `20260913_102057_336037` | `ea16a60f` | true / false (cancelled) | Receptacle reacquisition fails before visual placement; navigation repeatedly executes the same truncated prefix without progress; cancelled and independently scored |
| `20260913_103007_684ea5` | `89581c70` | **true / true** | Complete receptacle route, bounded visual placement corrections, released cylinder stably supported by cube at end; agent process also exits zero |
| `20260913_103442_018239` | `89581c70` | true / false | Unchanged repeat takes a longer route; payload slips during final turn, and empty-gripper placement is correctly rejected |
| `20260913_104630_1f3338` | `f7c024d3` | true / false | Payload retained, but Qwen accepts a mixed cube/cylinder receptacle mask; wrong support geometry and final nonconvergence prevent placement |
| `20260913_110132_edd836` | `cfb61c3d` | **true / true** | Coherent support geometry, longer transport retained, verified final release onto cube; process exit zero |
| `20260913_110659_839e2c` | `cfb61c3d` | true / false | Intermediate waypoint exceeds 10 s deadline; repeated approach/final-yaw reversals near the XY boundary coincide with payload loss |
| `20260913_112012_a863a5` | `ded0a3a3` | **true / true** | Stable phase handoff, shorter route, verified pickup and final placement; unchanged repeat required |
| `20260913_112432_810490` | `ded0a3a3` | true / false | Payload retained; clean cylinder mask, but placement corrections accumulate wrist sag and stall at 1.59 cm height error; release correctly refused |
| `20260913_113323_333844` | `cffc74d4` | **true / true** | Fixed placement reference, physical pickup and final placement; process zero |
| `20260913_113742_45c27b` | `cffc74d4` | **true / true** | Unchanged repeat; second visual correction reduces height error from 1.577 cm to 0.488 cm; stable physical placement |

Artifacts: `~/runs/emet/grasp-separated-neighbor-control` and
`~/runs/emet/grasp-loaded-carry-control`, including physical traces/results and
accepted/rejected wrist captures. These are diagnostics, **not acceptance passes**.
Both establish sustained physical pickup, not successful carrying or placement.

`63f7ed1b` preserves the gripper command through simulator posture changes,
matching the real bridge, and increases lift feedback/damping without increasing
the 70 N force limit. Static payload controls at 0, 0.5 and 0.75 kg pass the
existing position gate. This does not establish general grasp reliability.

Replay from the second run's held state (sim time 66.18 s), with identical
physics and arm retraction, reproduces payload ejection on a direct pitch target:
peak measured wrist speed is 20.90 rad/s. `bb8685e9` adds a physics-time
0.8 rad/s wrist reference limiter. The matched replay retains finger contact,
with peak measured speed 1.04 rad/s. These are conservative simulator settings,
not calibrated hardware limits or a guarantee on measured speed/acceleration.
The captured fixture and old-controller negative replay are regression tests.

`4e1f7df2` exposes reach on the common controller, shares precision arm-facing
alignment/reacquisition between pick and place, reads fresh copied joint state,
and prevents release after failed approach/IK. Release and retreat failures
remain failures; confirmed release clears held-object state even if retreat
fails. `7e3d4d43` replaces the fixed pregrasp with six bounded standoffs, largest
first, down to 20 cm; infeasible IK is rejected, not clipped at a negative joint.

The next live control retains the cylinder throughout carry/navigation. At
release, however, its center is offset about 2–3 cm from the nominal end-effector
origin. It briefly contacts the cube then falls onto the table. The tool reports
controller completion with an explicit physical-unverified disclaimer; the
independent scorer rejects placement. Artifacts are
`~/runs/emet/grasp-reachable-place-control`.

`c6b6b761` adds Qwen-verified current head RGB-D before release, without instance
creation or oracle state. It corrects observed payload XY alignment and visible
lower-surface height over the freshly grounded static support, with at most
three 5 cm moves, and abstains on missing freshness/identity/near-gripper support,
invalid IK or failed motion. Each alignment frame/mask/prompt is retained. This
is a local placement prototype: visible geometry may not reveal a hidden object
bottom, the support is assumed static, and it is **not real-robot acceptance**.
The first trial of that check correctly rejects a lost payload. Its saved head
frame shows only the receptacle; private tracing confirms the cylinder fell
before placement. No missing-object detection should be counted as a VLM error.

`424db036` adds a **separate** `query_geometry_manipulation_pilot.yaml` preset:
the same Qwen-associated wrist surfaces, but robust 5th/95th-percentile 3D bounds
and the measured grasp-center pose replace the fixed pixel/depth closure gate.
Corrections are bounded to 5 cm and require valid IK and completed motion.
The old preset remains the control. Placement also trims depth-edge outliers.
The geometry trial stabilizes at about 12 mm lateral / 7 mm vertical error;
actuator traces show no base response to the remaining 12 mm requests.

`a29d664d` targets 5 mm for the simulated arm's base-joint component, leaving
ordinary exploration/precision navigation policies and the 12 mm 3D grasp gate
unchanged. A new no-progress guard stops after three completed corrections
without measured end-effector movement. Job `20260913_085624_0c7f62` reaches
about 5.7 mm error and sustained pickup, but loses the cylinder during later
transport. The empty-hand navigation posture lowers the lift and folds the
wrist despite a potentially held payload.

`9714349f` captures lift/wrist posture after pickup and preserves it through
transport and placement preparation, retracting the arm using existing joint
commands. Confirmed opening clears the constraint; failed carry transition
retains conservative held-object state. This constraint is not holding proof.
Matched stationary replays (`20260913_090821_557461`) retain contact in **both**
postures: relative translation drift is 9.2 mm folded versus 5.1 mm preserved,
rotation drift 0.140 versus 0.093 rad. They do not establish the cause of loss
under navigation. The earlier scratch replay's absolute-height predicate was
invalid for comparing intentionally different lift heights; use contact and
relative drift, not that predicate. Live job `20260913_091308_642f8d` still fails
placement: verified pickup at 53.33 s, contact loss at 70.05 s during base-turn
braking. The last wrist frame correctly centers the intended red object.
Private contact replay places its center at pad-local x=-16 mm after lifting,
drifting to -27 mm before loss (pad half-width 20 mm). This suggests insufficient
contact depth rather than missing visual identity.

Matched recorded-command replay `20260913_091820_d1cead` loses the payload with
both unchanged wheel references and references limited to 8 transmission-rad/s².
Smoothing delays but does not eliminate loss; no braking workaround was added.
Grasp-depth counterfactual `20260913_092110_b42b80` tests this contact-geometry
hypothesis from the saved preclosure state, without changing physics. With the
same close/lift/retract and 35 s stationary hold, 0 mm insertion loses the object,
while 15/25 mm insertion retains it within 14.7/13.5 mm of the nominal EE link.
This is a diagnostic grasp-depth intervention, not an agent success count.

`b91d761e` provides a bounded grasp-frame `contact_offset_m` calibration and a
separate `query_geometry_contact_pilot.yaml` preset (15 mm toward the palm from
the nominal grasp link). The desired robot contact point, transformed by the
measured EE orientation, is aligned with observed object geometry. There is no
object-label rule, physics/friction change, or relaxed closure tolerance. This
is not a hardware calibration or proof of cross-object reliability. Full-task
job `20260913_092427_1e2e65` still fails: its executed grasp is shifted sideways,
not deeper. The simulator's massless `link_grasp_center` marker used intrinsic
MuJoCo Euler angles copied from fixed-axis URDF RPY, rotating its axes about
120 degrees relative to the published URDF frame. The diagnostic calibration
mistakenly used marker +Y instead of published-frame -X (toward the palm).

`69d8e478` expresses the fixed marker using the URDF-equivalent quaternion,
checks orientation at three wrist pitches (residual below 3 mrad), and corrects
the experimental offset to `[-0.015, 0, 0]`. It changes no collision geometry,
mass or inertia, and does not alter the already-URDF-based published pose.
Job `20260913_093103_771bbf` still loses the payload at 73.11 s. The commanded
arm extension at closure is 0.173 m versus 0.167 m in the nominal run: the
executed depth improvement is smaller than the 15 mm static intervention.
Keep the 12 mm closure gate, visible-surface bias and loaded arm displacement
in view; corrected axes alone do not establish robust holding. Original
geometry-only centering remains an explicit control. `ea26ace8` saves the
private preclosure fixture and matched retention tests for reproducibility.

The subsequent coordinate audit finds grasp/place absolute geometry using
episode-relative `get_base_pose()` with world-frame point clouds. `292e84df`
uses `get_base_pose_world()` and explicitly world-frame alignment goals;
nonzero-origin grasp/placement tests pass (62 focused tests). This is a contract
fix, not an offset tuned for the tabletop origin. Physical regression job
`20260913_093628_3c845e` passes all 34 tests (114 s), including matched retention,
frame agreement, loaded lift, wrist profiling, command atomicity and wheel
transmission. Live retry `20260913_093656_0b9dbd` starts afterward under the
exclusive experiment lock. It reaches placement, but the payload has slipped.

`0a539b30` advances only the experimental contact preset to the replay-tested
25 mm offset. Job `20260913_094303_496e65` retains the object through transport
and receptacle reacquisition (contact through 94.23 s), then loses it during
the placement approach. A mixed-depth point at z=0.749 m raises the requested
EE height to 0.849 m although the cube top is 0.56 m. Its short 4.3 cm arm-base
command also starts a general SE(2) correction: the base turns while the arm is
extended, and the client reports joint-position completion before base yaw
settles. The saved final image shows the dropped red object on the table beside
the blue cube; the near-gripper check correctly prevents an empty release.

Three separate fixes follow: `f5ac318e` uses robust support height and avoids
mutating sampled cloud points; `c508574d` executes the arm base component as a
signed translation with heading hold and requires 100 ms of quiet measured
joint/base velocities before completion; `c1468a11` reuses the existing
physics-time limiter for arm (0.1 m/s) and lift (0.15 m/s) references. Default
navigation stays pose-based. No force limits, object geometry, friction or
physical scoring gates change. Ninety focused tests pass, including motion
contracts and wrist/translation reference tests. Full-task retry
`20260913_095547_9ec2d2` runs before neighboring precision navigation control
`20260913_095730_685498`; the shared evaluation checkout remains frozen until
both finish. The manipulation retry keeps the object held to the end, but its
four alignment frames contain 10–15% support/table depth inside the correctly
selected red mask. The resulting z correction stays +0.10 m despite repeated
upward moves. The final object is still held: this failure is not a slip or
Qwen identity mistake. Navigation completes all ten route moves with no
failures; overall health remains `incomplete_telemetry`, not full acceptance.

The depth contamination has a source bug: the MJCF head depth eye is 15 mm
from RGB, while the bridge publishes color intrinsics/pose and advertises
registered pixels. `076fa458` moves only the virtual aligned-depth viewpoint to
the color optical center in both Stretch assets. It does not change physical
objects. Render control `20260913_100547_d6990c` passes four tests, reproducing
incorrect object-pixel depth with the old offset and verifying agreement with
registered rendering. Full-task retry `20260913_100708_0f6983` still refuses
release: the selected mask includes minority support-depth pixels, and its
four z corrections are +9.9, -5.5, +2.1 and +10.0 cm. Correct registration alone
does not guarantee clean semantic-mask depth. The object remains held.

`ea16a60f` trims minority depth tails in the already verified placement mask
using a median/MAD band, retaining at least 80% of valid pixels or abstaining.
It preserves raw masks/depth and records filtering statistics. This assumes
dominant coherent object support, not arbitrary multimodal geometry; no motion
or physical success tolerance changes. Twenty-eight focused tests pass.
The matched learned retry `20260913_102057_336037` never reaches this check:
after pickup it loses the receptacle view and repeatedly executes the same
short navigation prefix. It is cancelled after about five minutes; preserved
trace scoring confirms pickup but not placement. Do not count it as a depth
filter outcome or omit it from complete-system failures.

The navigation audit exposes search/execution disagreement: eight-connected
search allows diagonal edges through blocked corners that the execution LOS
check rejects. The safety filter then discards the unsafe suffix but retains
the original arrival marker and hides the rejection. `89581c70` shares
no-corner-cutting neighbors across both searches and reachability, rejects
unsafe routes without a false arrival marker, and stops chunk continuation
without measured translation. Forty-five focused tests pass, including
single/multi-goal detours and disconnected diagonals. The fresh learned retry
`20260913_103007_684ea5` uses the reproducible driver with explicit contact and
separated-neighbor presets. It passes independent pickup and final placement:
pickup at 59.134 s, final supported state at 117.584 s, no gripper/other contact.
Final cylinder center is approximately (-0.2632, -0.5612, 0.5996) m. Placement
corrections have z components -3.39, +1.64 and +1.46 cm; raw captures and depth
support statistics are retained. Agent process exits zero after 165 s. The
combined focused suite passes 168 tests. This is the **first separated-neighbor
diagnostic pass**, not the original-clutter or cross-task gate. An unchanged
repeat, `20260913_103442_018239`, fails: pickup at 55.834 s, last finger contact
at 95.960 s during the later turn, then object falls to floor. Final grounding
correctly finds no held target. Current repaired configuration is **1/2**, not
reliable. Do not advance to original clutter yet. Private matched replay
`20260913_104239_40cb0e` compares the recorded commands, wheel-reference slew
limits and the simulator's full-close endpoint; these are diagnostics, not
extra agent successes or changes to production grip limits.
All four replays retain the object, **including the recorded-command control**;
the 10 Hz snapshots with reconstructed velocity do not reproduce the live loss.
Do not use them to claim a causal benefit for smoothing or stronger closure.
The replay figures confirm both live endpoints (cube support versus dropped
object); they do not replace the original traces.

The wheel contract audit separately finds independent actuator saturation:
requested geared wheel targets exceed their limits and clip to equal speeds,
driving straight despite a turn request. The next fix scales both targets by
one feasible fraction, preserving the requested curvature without raising any
speed/force limit. Twenty-nine wheel/translation tests pass. No new grip force
or acceleration tuning is promoted from the inconclusive replay.
`7ee502f3` implements this wheel-target scaling. Learned retry
`20260913_104630_1f3338` and subsequent ten-move navigation control
`20260913_104659_c274cb` share frozen `f7c024d3` (the extra commit only hides
rangefinder debug rays in offline figures). Both finish. The navigation control
completes ten moves with no failures, but health remains `incomplete_telemetry`.
The learned run retains the payload but fails placement. Initially the residual
vertical correction suggests loaded-joint tracking bias; inspecting the raw
receptacle mask reveals a more important error: Qwen selects a combined cube
and held-cylinder proposal, producing a false support top at about 0.82 m
(real cube top 0.56 m). Do not tune release tolerance against that geometry.

`cfb61c3d` applies the existing 8 cm depth-adjacency boundary to external masks
as well as raw RGB-D proposals. The saved mixed mask separates into cube support
at roughly 0.92–0.99 m camera depth and held-object support at 0.58–0.63 m.
No color/category rule is added; smooth sloped and textured objects remain
connected. Coplanar/touching mixed objects still need semantic/multiview checks.
Thirty-four focused tests pass. Learned retry `20260913_110132_edd836` passes
pickup (57.634 s) and final placement (130.088 s), with process exit zero after
187 s. It retains the payload on the longer route and grounds the real cube
support. One visual correction precedes release; final z error is 1.48 cm,
within the unchanged 1.5 cm gate. Unchanged repeat `20260913_110659_839e2c`
fails before placement: a 10 s intermediate-waypoint deadline expires and the
payload slips during navigation. This version is again **1/2**, not reliable.
No accumulated-motion workaround or looser release gate was added.

The failed repeat stays upright (roll/pitch below about one degree around loss),
and reconstructed contacts show no robot/table collision. Wheel commands switch
between clockwise final-yaw rotation and counterclockwise XY approach near the
7 cm exploration boundary. The stateless phase switch has no positional buffer
for braking/turn drift. `ded0a3a3` acquires XY within half the acceptance radius,
then keeps final-yaw control until drift exceeds the original radius. Arrival
still requires the original XY/yaw tolerances; new goals reset the phase.
Thirty-four phase, translation and wheel tests pass. Live retry
`20260913_112012_a863a5` tests this hypothesis with the same contact preset,
scene and unchanged deadlines. It passes pickup at 57.734 s and final placement
at 118.808 s (process zero, 168 s), but takes the shorter route. Unchanged repeat
`20260913_112432_810490` retains the object through navigation but fails visual
placement: height error stalls at 1.587 cm, outside the unchanged 1.5 cm gate.
This version remains **1/2**. Boundary chatter is established in the earlier
failed trace; shorter successful transport is not proof of long-route retention.
The neighboring ten-move precision route `20260913_112148_6772e1` passes all
moves: maximum XY error 0.01937 m, yaw error 0.02895 rad, zero corrections.
Health remains `incomplete_telemetry`. The next route revision (`53d2f06a`)
adds four coupled translation/final-heading goals, for fourteen moves per
repetition; the previous isolated-turn controls missed this phase transition.
The combined focused suite passes 253 tests, including arrival/trajectory
contracts and bridge tests with the bridge package on `PYTHONPATH`.

The latest placement mask contains only the cylinder, unlike the earlier mixed
support failure. Across its four captures, measured wrist pitch moves from
-0.351 to -0.371, -0.393 and -0.418 rad. The controller reuses each loaded angle
as the next target despite intending to preserve orientation. After the first
correction, measured EE height stays at about 0.5908 m while lift commands rise.
`cffc74d4` preserves one orientation reference and accumulates current visual
position corrections into the commanded reference. It rejects reference-to-
measured displacement over 5 cm, keeps the three-motion limit, and still requires
fresh observed geometry inside the original release gate. Bias and no-motion
tests pass (23 placement tests). Retry `20260913_113323_333844` passes pickup
at 59.034 s and final placement at 116.972 s (process zero, 167 s). Unchanged
repeat `20260913_113742_45c27b` passes pickup at 58.034 s and final placement
at 118.298 s (process zero, 169 s). The repeat exercises the former stall:
height correction changes from -3.181 cm to +1.577 cm, then +0.488 cm after
the second move. The pre-fix method fails the new tracking-bias unit test;
the fixed method passes. The full focused suite passes 255 tests.

This is **2/2 separated-neighbor diagnostic placement**, not original-clutter
or room-task acceptance. Expanded route `20260913_113326_d91f20` passes all
fourteen precision moves, including coupled translations/final turns, with
zero corrections; health still reports `incomplete_telemetry`. It is an
empty-hand control, not long-route payload-retention evidence. The next learned
case `20260913_114456_d0f9e2` uses the unchanged original fixture and the
predeclared `query_geometry_contact_aperture_pilot.yaml` on frozen `cffc74d4`.

### Return to original clutter

These are distinct from the separated-neighbor results above. The aperture
preset is still experimental; neither scene nor physics is changed for retries.

| Job | Source | Physical pick/place | Finding |
| --- | --- | --- | --- |
| `20260913_114456_d0f9e2` | `cffc74d4` | false / false | Aperture narrows from 16.4 to 10.5 cm; diagonal approach contacts neighboring cube before lateral alignment and motion fails |
| `20260913_115105_65d8ba` | `458ed72b` | false / false | Full transverse-plane alignment requests negative arm extension at the retracted limit; rejected before approach |
| `20260913_115500_c04c71` | `34ba733a` | **true / true** | Finger-axis alignment reaches pickup; longer chunked transport retains the payload; final physical placement passes |

The first trace reconstructs right-finger/cube contact at about 43.43 s,
before the target is between the fingers. `458ed72b` separates alignment from
insertion, but the plane perpendicular to tilted grasp X couples lowering to
arm retraction. Captured-pose IK requests -4.4 mm extension from a +5.5 mm
current extension. `34ba733a` instead aligns only along grasp-frame Y, the
finger-opening axis, while separated; height and depth change together after
that lateral error is within the existing 12 mm gate. The same captured-pose
IK preserves +5.5 mm extension. Rotated-frame, aperture and placement tests
pass (53); original-scene retry `20260913_115500_c04c71` passes pickup at
63.534 s and final placement at 139.488 s (process zero, 196 s). It traverses
a longer chunked route before reacquiring the receptacle. Final visual height
error is 1.37 cm, inside the unchanged gate. Matched separated-neighbor control
`20260913_115609_934cb7` passes pickup at 65.360 s and final placement at
134.788 s (process zero, 191 s). It uses the same code and contact/aperture
preset; only the neighbor pose changes. The mirrored-neighbor fixture is predeclared in
[acceptance](manipulation_acceptance.md), not chosen after testing outcomes.

This sequence is **not a general collision-free approach planner**. No
object-label or fixture-location branch is added, and failed motion still stops
the grasp. Validate on the earlier neighbor fixture and varied clutter before
promoting the new sequence or making broader manipulation claims.

The independent IK audit also fixes a joint-layout contract: full eleven-joint
seeds must convert to nine solver joints once and return a full configuration,
preserving passive joints. Both IK entry points pass an actual zero-error FK/IK
round trip, alongside 41 focused tests. This later fix is not in the `cfb61c3d`
trials but is included in `ded0a3a3`. Private traces now retain measured velocity, activation and
solver warm-start; sampled traces are still not exact command replays.

Rendering caveat: the existing simulator hides robot geometry from head RGB-D
to avoid mapping self-obstacles; wrist images retain it. This removes real
head-camera self-occlusion during placement. Do not infer real perception
robustness from these passes. Validate robot-visible observations with proper
mapping self-filtering before hardware claims. Offline replay figures explicitly
show robot visual geometry and are labelled reconstructed views, not agent
observations.

`query_geometry_aperture_pilot.yaml` is an additional, not-yet-live-tested
clearance ablation. It measures both identified finger markers in calibrated
RGB-D and narrows at standoff toward the observed target extent plus a 6 cm
marker/thickness/uncertainty margin. Missing markers/depth/standoff stop the
grasp. This is not an inner-jaw calibration or collision-free guarantee. Keep
geometry-only and aperture variants separate in reporting. Original-clutter
clearance, room OVMM and learned TAMP gates remain pending.

`query_geometry_contact_aperture_pilot.yaml` combines the same contact setting
with that aperture margin for the next original-clutter diagnostic. The two
component controls remain unchanged. This tests a combined system, not an
isolated contact-depth effect.

## September 13: manipulation command delivery audit

The next battery and stop gates are in [bounded acceptance](manipulation_acceptance.md).
These diagnostics do **not** establish a passing manipulation gate.

| Job | Frozen source | Task outcome | Evidence |
| --- | --- | --- | --- |
| `20260913_012127_e3d28d` | `afc32857` | Pickup failed; place skipped | Two accepted wrist captures; stopped on failed servo motion |
| `20260913_013027_fc29a4` | `aa2739d4` | Pickup failed before wrist approach; place skipped | Private physical score false/false; zero gripper-contact samples; tool failure correctly retained |
| `20260913_013657_dc2c7d` | `1a9959bb` | Pregrasp and nine wrist steps pass; approach stalls; place skipped | Private physical score false/false; replay identifies right fingertip contacting neighboring cube |

Artifacts are `~/runs/emet/grasp-command-identity-retry/evidence` and
`~/runs/emet/grasp-precision-physical-retry/hybrid_learned_pick_place`. The latter
contains `physical_trace.jsonl`, `physical_result.json`, `process.log`, and
`evidence/manipulation_outcomes.jsonl`. The agent process exits zero, but the
driver exits one on physical failure. This is intentional, not an infrastructure
exception. Broken-pipe errors during teardown remain a separate cleanup issue.

The first run's short base correction requested 5.2 cm, while the dynamic motor
tolerance stopped at roughly 2.6 cm error, outside the arm client's 2 cm gate.
`60fb8652` applies the existing precision policy to manipulation base goals.
It does not relax client, geometry or grasp acceptance. Retransmissions now
retain command identity, and the grasp loop honors failed motion/invalid depth.

The second run exposed a separate lost-command bug: the server logs a lift goal
of 0.643286 m, but the private actuator trace stays at 0.600 m and the measured
lift stays near 0.593 m. This is not a failure to physically follow the requested
actuator setpoint: **the setpoint itself never changed**.

![Requested lift versus actual actuator target and measured joint position](lost_lift_command.png)

Figure source: that run's private trace, wall-time interval
`1789277543.5–1789277548.0`; `ctrl[2]` and `qpos[9]`, checked against the frozen
default-scene model (48 qpos). Dashed line is the request in the server log.

The physics consumer reads a serialized command snapshot, clears triggers, then
writes the entire snapshot back. Writers previously held a different lock, so
this acknowledgement could erase a concurrently submitted joint command.
`1a9959bb` shares the interprocess lock across read/modify/write on both sides,
and releases it before waiting for stop consumption. A deterministic concurrent
writer/consumer test and stop-wait test pass (25 focused simulation tests).
Matched live retry `20260913_013657_dc2c7d` on `1a9959bb` passes pregrasp and
nine wrist association steps, approaching from 0.368 to 0.220 m. Unlike the
previous run, lift and arm requests appear in the actuator trace. It then stops
on a genuinely stalled approach, reports pickup failure and skips place. Private
physical scoring is false/false with zero red-cylinder gripper-contact samples.
Artifacts: `~/runs/emet/grasp-atomic-command-retry/hybrid_learned_pick_place`.

Offline `mj_forward` replay of the frozen model with recorded `qpos` and `ctrl`
isolates a collision with the *neighbor*: `rubber_tip_right` contacts blue cube
`object1`, penetrating about 1.65 mm at wall time `1789277958.553609` and 2.10 mm
at `1789277961.4929755`. The red cylinder remains supported on the table. These
are replayed contact geometries, not saved live contact-force measurements.
The red-only live contact trace cannot by itself establish robot-to-neighbor
collisions; the replay supplies that additional evidence.

![Last accepted wrist view: red target between fingers, neighboring blue cube at the right finger](grasp_neighbor_clearance.png)

Unedited final accepted wrist capture `grounding-7d6f1abc90184d659eecc60b8109b14c`,
recorded at wall time `1789277959.3625605`. Contact attribution above comes from
the state replay, not solely from this image.

Next diagnosis: a predeclared separated-neighbor fixture control, then
geometry-aware grasp aperture/approach clearance. Do not silently remove the
blue cube from the original task, widen association thresholds, or call a
smaller graph a manipulation improvement. The tabletop gate still fails, so
room OVMM and learned TAMP have not advanced. A neighboring precision navigation
control `20260913_014326_489ec4` on `1a9959bb` succeeds on six route moves,
then fails move seven: XY error 0.02120 m, yaw error 1.08045 rad, stop confirmed,
reason `navigation stalled`. The remaining three moves are not run. Frozen
pre-lock control `20260913_014724_bf28d2` on `aa2739d4` passes all ten moves
(max XY 0.01723 m; max yaw 0.02901 rad). Its overall health status is still
`incomplete_telemetry`, because posture/actuator telemetry is missing.

This pair blocks a no-regression claim. One possible contributor is the added
manager-lock RPC overhead in the physics loop; a separate controller/progress
issue near the 2 cm XY boundary also needs consideration. `a244ac25` retains
atomicity with a native spawn-context lock, avoiding per-tick manager lock RPCs.
An actual spawned-process exclusion test passes. The same-route native-lock
control `20260913_015157_35365a` passes all ten moves, zero corrections, max XY
0.01958 m and max yaw 0.02874 rad. As with the baseline, the full health probe
is `incomplete_telemetry`, not full robot acceptance. This is one matched control,
not evidence that all navigation regressions are ruled out or that IPC overhead
alone caused the preceding failure. Keep that failed run in the record.
The final combined focused suite passes 161 tests (command lifecycle, navigation,
grasp handoff, query memory, physical scoring and simulator command atomicity).
This includes the native-lock spawned-process test.

## Earlier frozen pilot

Interpretation: [environment acceptance progression](../environments/README.md),
[Habitat search scope](../environments/habitat.md),
[simple-sim controls](../environments/simple_sim.md).

This freezes the September 10 candidate, not a new default. The earlier verified
red/blue table finds are integration evidence, not OVMM or manipulation success.

Launch: managed job `20260910_213038_d86e1a`, frozen source `b194395a` at
`/tmp/emet-cross-task-20260910-frozen`. Artifacts:
`/home/cpaxton/runs/emet/shared-grounding-cross-task-20260910`.
This first launch is invalid for comparing strategies: all four OVMM cases and
six EQA cases encountered SAM2's missing `iopath` dependency during model
construction. OVMM serialized the exception despite exit zero. The absent-object
find then reached its 360-second cap while exploring; learned pick/place never
started. Retain these artifacts as infrastructure failures and a search timeout,
not a model-quality score.

September 11 retry: `20260911_185541_56a4c4`, frozen `eed1d868`, artifacts
`/home/cpaxton/runs/emet/shared-grounding-habitat-retry-20260911`. Installed
`iopath==0.1.10` and `portalocker==3.2.0` in Habitat, without changing Torch or
model weights. The runner now constructs SAM2 and executes synthetic box
inference before episodes, and supports `PHASE=habitat|sim|all` (default all).
This retry selects Habitat only, with the original cases, budgets and presets.
`process_status.tsv` records exits, not task acceptance. All ten retry processes
completed without infrastructure exceptions:

| Strategy | EQA q15 / q16 / q25 | OVMM object + receptacle, two scenes |
| --- | --- | --- |
| Hybrid | correct / correct / wrong (2/3) | 0/4 |
| Qwen-box | wrong / correct / correct (2/3) | 0/4 |

These three questions do not establish no regression against the earlier 3/3
smoke. The strategies disagree on two questions despite equal aggregate scores.
Hybrid scene 00025 returned an object localization 3.615 m from the target;
Qwen-box returned a receptacle localization 5.184 m from its evaluator target.
Neither is a task success. Other phases returned no localization.

Manual review of hybrid scene 00025's accepted support
`grounding-965b2a585dd64549851ab3ce1af623dd-surface-0.png` shows bedding/fabric,
not an identifiable lamp. The proposal query is `nearest bed`, while the support
selector receives the full question about the lamp nearest a bed. Qwen accepts
the bedding while claiming a small object on it resembles a lamp. This is a
concrete target/anchor confusion and false semantic acceptance; metric distance
alone would not reveal it. Preserve identity versus relationship as separate
verification concerns in the next fix, without handing context pixels back the
authority to validate an incorrect mask.

The pending learned Stretch pick/place diagnostic was launched separately as
`20260911_194952_1415c5` on the same `eed1d868`, with a 600-second cap, velocity
navigation, hybrid preset and no oracle scene plans. It does not rerun the known
absent-query timeout first. The tool failed after 195.9 seconds with
`Command 101 failed: terminal command outcome is immutable`; the agent reported
the failure rather than claiming success. No successful learned pick/place is
established. The command lifecycle exception needs diagnosis; process completion
is not physical task acceptance.

Lifecycle follow-up `efba8d7f`: reproduced the exception when a goal had already
failed with stop unconfirmed, then a later cancellation confirmed stopping and
attempted to rewrite the terminal status to cancelled. The fix retains the failed
outcome/reason and records later stop confirmation separately, releasing motion
ownership only on confirmed stop. Both core/deploy copies match; 37 focused
command, trajectory and adapter tests pass. Same-preset bounded sim retry:
`20260911_200332_5b6840`, artifacts `~/runs/emet/lifecycle-manip-retry/evidence`.
The retry reached manipulation without the lifecycle exception, but failed
pregrasp with arm extension -0.180 m and subsequently lost current-frame target
support. No pick/place success. This run did not necessarily exercise stop recovery.

Manipulation handoff follow-up `40288111`: the verified find pose faced the
object with the camera, while Stretch's side-grasp expects the target along -Y.
The grasp adapter now owns a measured target-facing side rotation and fresh
reacquisition; find remains unchanged. Failed/nonfinite pregrasp IK now returns
failure instead of falling through into visual servo. 23 focused tests pass.
Bounded same-preset sim check: `20260911_202359_853d5b`, artifacts
`~/runs/emet/grasp-handoff-retry/evidence`. The retry completed orientation and
reacquisition, then reached pregrasp (IK arm -0.039 m, clamped by the legacy 5 cm
allowance). Wrist tracking rejected absent/ambiguous current-frame support;
pickup failed and place was skipped. This is not manipulation acceptance.

`3b2ce750` additionally propagates arm-motion failure rather than treating valid
IK as execution success (25 focused tests pass). `a7c45096` retains wrist RGB,
depth, calibration and grounded world points on tracking rejection under the
existing episode evidence opt-in. Serial diagnostic `20260911_202818_77c40c`
uses that frozen source; artifacts `~/runs/emet/wrist-tracking-audit/evidence`.
No tracking-tolerance relaxation or oracle fallback was introduced.

The wrist audit completed with failed pickup, not a task success. Manual review
of `wrist_tracking/grounding-054937277d1849de892afab78332be86.png` shows the red
cylinder clearly. Its stored world median projects to wrist pixel (160.3, 216.1),
with expected camera depth 0.380 m versus observed 0.370 m. This argues against
a gross camera-frame error in this observation. Replaying the current expanded
world bounding box gives two components of 4,726 and 2,005 pixels, triggering the
intended ambiguity rejection. Bounding-box-only association is insufficient in
this close view; object-specific wrist support is needed. Neither largest-mask
fallback nor looser margins is justified by this evidence.

September 12 candidate fix shares `ground_vlm_frame` between head and wrist:
same configured provider and semantic selector, followed by the original
grounded-target spatial mask association. Wrist verification does not ingest a
new instance. Positive wrist depth is accepted below the navigation-map minimum
because grasp views can be closer than mapping views; zero/invalid depth remains
excluded. The former box-only connected-component gate is not a semantic mask.
63 focused tests pass, including semantic abstention and wrong-world-target
rejection. Added per-frame VLM work may increase servo latency and must be
measured; no real-robot or learned-manipulation acceptance is claimed.

Offline red-cylinder, blue-cube and absent-banana queries on the exact saved
wrist view were submitted as `20260912_091644_fa11a1`, then cancelled before
inference: NVML reports a driver/library mismatch (loaded kernel 595.84,
userspace 595.91). GPU repair is required before cached-model and live validation.
The three inputs all reference the previously listed wrist capture; they are
diagnostic queries on one view, not an expanded independent test set.

Post-reboot replay `20260912_153104_dafcde`, frozen `dd29d217`, completed after
kernel/userspace driver alignment to 595.91.07 and successful full SAM2 inference
preflight. Artifacts are `~/runs/emet/wrist-postboot-{proposals,baseline,support}`.
YOLOE→SAM2 produced one red-cylinder proposal, one blue-cube proposal, and zero
absent-banana proposals. Support-only Qwen accepted red and blue (selector times
1.23 s and 0.99 s respectively); banana abstained without a model call.
Manual inspection of `wrist-postboot-support/support_only/0-panel-0.png` confirms
the red support depicts the cylinder. Reprojecting the saved calibrated depth
and applying unchanged original-red-target association accepts 1,397 red pixels
and rejects the blue mask. This validates the intended distinction on this
saved frame, not robustness during motion or physical pickup. The focused
63-test suite also passes post-reboot. Live bounded pick/place retry
`20260912_153738_fb4a22` uses the same frozen source and preset; evidence goes to
`~/runs/emet/wrist-live-postboot/evidence`. The tool failed after 40.3 s:
wrist semantic/world association completed, then a legacy shape check accessed
`servo.semantic.shape`, despite head semantic labels being absent on this path.
No pickup or place succeeded. Process exit zero is not task success; teardown
also emitted multiprocessing broken-pipe/reset errors. Fix `2db1b757` validates
the selected wrist mask against wrist world geometry instead, with tests for
absent head labels and mismatched wrist resolution (11 handoff tests pass).
Same-preset bounded retry `20260912_201500_bb5284` retains all model and control
settings; artifacts `~/runs/emet/wrist-shape-retry/evidence`. The focused suite
passes 65 tests. The retry completes the wrist approach from 0.369 to 0.166 m,
closes the gripper, and advances to finding the blue cube. It then fails
placement navigation at the first waypoint: command 186, `motion deadline
exceeded`, `stop_confirmed: true`. Tool duration is 182.3 s; overall pick/place
fails. There is no independent confirmation of a held object. Post-grasp head
image `grounding/grounding-6b9317d388ee4aa09d068ca40238fbdf.png` shows the blue
cube and a small red region beyond the table, not a verified in-gripper cylinder.
Do not score gripper closure as physical pickup. Next isolate physical grasp
verification and post-grasp navigation; do not loosen tracking tolerances or
launch broader acceptance pilots yet. Multiprocessing teardown errors remain.

### Placement-navigation timeout: wheel transmission units

The failing placement waypoint is approximately an in-place 175-degree turn,
not a long drive. Isolated baseline `20260912_232620_453a49` (source `933fa725`)
reproduces it without perception models or manipulation. The nominal 10 s
command budget is stretched by the existing sim-time ratio: failure occurs at
12.27 wall seconds, measured yaw -2.369 versus goal -3.05 rad (0.681 rad short).
Pose samples show continued slow rotation, not a stationary obstacle stall.
The earlier diagnostic `20260912_232447_3fc9ac` never launched simulation because
its temporary route YAML was malformed; exclude it from navigation results.

Stretch's MJCF wheel velocity actuators use `gear=3`. The bridge previously
sent wheel-joint rad/s directly to actuator controls, and interpreted actuator
velocity as wheel-joint velocity in feedback. MuJoCo's [transmission semantics](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-general-gear)
scale actuator velocity by gear. Fix `76ebf9a1` multiplies commands by the model's
actual gear and reads joint `qvel` for wheel-speed telemetry. No MJCF friction,
limits, navigation deadline, arrival tolerance, or speed-boost changes.
15 tests cover command/feedback units with unit, positive non-unit and negative
gears; the broader adapter/command suite passes 44 tests.

Matched fixed diagnostic `20260912_232858_58901d` succeeds: first positioning
command 1.35 s, half-turn 7.01 s, return turn 6.46 s. Terminal half-turn error
is 0.055 m / 0.142 rad, within unchanged exploration tolerances 0.07 m / 0.15 rad.
These are exploration-contract passes, not precision-navigation acceptance.
Both runs use the existing `probe_rby1_camera.py` on seed-0 default-table Stretch,
with a temporary wrapper selecting the unnamed contract and nominal 10 s timeout,
and logging pose samples. Route: `(0.13, 0.0085, 0.006)`,
`(0.13, 0.0085, -3.05)`, `(0.13, 0.0085, 0.006)` in episode coordinates.
Artifacts: `~/runs/emet/nav-turn-{trace-v2,geared}`, including observations and
images; managed job logs retain timestamped `NAV_SAMPLE` and `NAV_RESULT` lines.
Integrated same-preset learned retry `20260912_233137_f012c2` uses frozen
`76ebf9a1`; artifacts `~/runs/emet/manipulation-wheel-units/evidence`. Initial
navigation completes without timeout, but fresh target grounding rejects
`target absent or ambiguous` before pickup (tool 17.7 s); place is skipped.
This run does not exercise post-grasp navigation, and does not establish
manipulation success. Keep that integration gate open despite the isolated
navigation fix passing. Existing multiprocessing/EGL teardown errors persist.

Follow-up `20260912_233819_e45145`, source `94149560`: one repeat of the existing
`configs/benchmarks/navigation_acceptance.yaml` using `probe_rby1_camera.py`
(`--route-repeats 1`), default-table Stretch, no perception models. All ten
commands succeed under the named precision contract (2 cm / 0.03 rad), including
±10° and ±30° turns and 20 cm forward/return translation. Maximum terminal
errors: 0.01939 m and 0.02953 rad; zero corrective redispatches. Artifacts:
`~/runs/emet/nav-geared-precision`. The probe summary is `incomplete_telemetry`,
not full acceptance: Stretch omits the requested base-up/actuator-target fields.
This is one route repeat on one robot/environment, not cross-robot robustness.

Manual comparison of the final two frames from `manipulation-wheel-units`
shows the red cylinder on the extreme lower-right edge of accepted frame
`grounding-4e5d1b8424ba4aeda2f56cb0e358c07f.png`, and no visible cylinder in
rejected `grounding-8a445c194f364a029016ffcaa02cb195.png`. The rejection has no
supported proposals; Qwen is not rejecting a clearly visible cylinder there.
The sweep permits an early return within 0.35 rad of commanded head pose,
even during motion. Residual camera motion is a hypothesis to measure before
changing verification. Camera-only trace `20260912_234053_eb1405` confirms the
mechanism: after the soft-wait handoff, saved camera rotation changes another
8.9145 degrees; base displacement is under a micrometre and yaw change under
0.003 degrees. `~/runs/emet/head-handoff-trace/handoff/head_0.png` shows the
cylinder and cube, while `head_3.png` shows neither. First sample is already
0.20 s after soft-wait return; this is a lower bound on total post-return drift.
The full observations have `image_timing: null`; do not infer capture latency
from joint-state sampling alone.

Fix `d9bb2ce3` uses the robot adapter's blocking head move and existing newer-frame
wait for verification callbacks only; ordinary mapping retains its soft sweep.
91 focused head-sweep/query/grounding/manipulation tests pass. This does not
upgrade the underlying head-motion API to a verified success contract; clients
that silently time out remain a follow-up. Serial job `20260912_234501_ffeb03`
checks the fixed camera handoff, then the same bounded learned command. Artifacts
`~/runs/emet/head-handoff-settled` and `~/runs/emet/settled-gaze-retry/evidence`.
Fixed camera handoff rotates just 0.0037 degrees across the corresponding saved
views after a 1.05 s head/frame wait (versus 8.9145 degrees before). Integrated
retry passes arrival verification and immediate fresh grounding, then rejects
after the subsequent side-grasp rotation/head movement: final
`grounding-930a1f12ee68467e86625b9a83089794.png` shows floor and table edge.
Tool duration 30.1 s; no pickup or place. The grasp adapter did not wait for the
RGB-D stream to advance after that second movement. Candidate `4f946d1d` adds
the existing post-motion frame wait there, without changing mask acceptance;
46 targeted tests pass, including head → frame wait → grounding order.
Bounded retry `20260912_234917_6b6fdf`, artifacts
`~/runs/emet/grasp-fresh-frame-retry/evidence`, completes with failed pickup
(tool 62.1 s). Both head-camera handoffs pass and wrist servo approaches from
0.368 to 0.218 m. Final calibrated wrist capture
`wrist_tracking/grounding-006bc3656f6440b19ab073405e0573ec` shows the cylinder
partly behind a gripper finger and clipped at the lower image boundary. Qwen
selects candidate 4, but its resulting instance-0 mask fails original-target
world association: only 1,339 / 3,385 valid pixels (39.6%) lie in the original
bounds plus the unchanged 5 cm margin, versus the required 80%. Selected support
median `(0.118, -0.465, 0.476)` differs from original target median
`(0.079, -0.539, 0.523)`. Do not relax the threshold on this example. Distinguish
object displacement/contact, mixed mask support, and camera/pose timing before
changing tracking. Several servo-center depths are zero and manipulation base
goals are repeatedly reissued; inspect these control paths next. No gripper
closure or place success in this retry. The full 91-test focused suite still
passes; these are correctness repairs, not long-horizon manipulation acceptance.

## Fixed comparison

- Hybrid: `query_detector_segmented_pilot.yaml`, YOLOE boxes → SAM2.
- Reference: `query_segmented_support_pilot.yaml`, Qwen boxes → SAM2.
- Both: support-only Qwen3-VL-8B-Instruct int4 verification, lazy query memory,
  unchanged fusion and task budgets. This tests proposal strategies, not SAM2
  versus YOLOE segmentation, and not lazy memory versus DynaMem.
- Habitat OVMM: nearest-v2 scenes 00006 and 00025, seed 0, 12 rounds/8 nav steps.
- EQA: questions 15, 16, 25, 20 planning/10 movement steps, no semantic sensor
  labels or enriched GT hints. The CLI does not pin EQA RNG seeds: these are
  matched questions/settings, not a deterministic causal no-regression test.
- Hybrid-only Stretch table: one absent-object find and one learned pick/place.
  Velocity navigation; no teleport or oracle manipulation. This is a multistep
  integration diagnostic, **not the PR #160 TAMP battery**. The oracle TAMP
  control previously passed; learned TAMP acceptance remains outstanding.

All twelve processes run serially. Stop after a timeout to inspect cleanup.
Each has a wall cap, not an extended task budget. The runner records process
completion separately from benchmark correctness. Report abstentions, grounding
errors, navigation failures and manipulation failures separately; do not turn
tool `ok` or exit zero into physical success.

## Reproduction

From a frozen checkout with the SAM2 small, YOLOE-L and MobileCLIP checkpoints
available at the usual loader paths:

```bash
OUT=/absolute/new-artifact-directory \
HABITAT_BIN=/absolute/habitat-env/bin/emet-habitat \
AGENT_PY=/absolute/agent-env/bin/python \
SAM2_SOURCE=/absolute/pinned/segment-anything-2 \
emet jobs run --name shared-grounding-pilot --need-mib 16000 \
  --cpu-safe --gpu-exclusive -- bash scripts/run_shared_grounding_pilot.sh
```

`EMET_CONFIG` already feeds the unified preset to both Habitat runners through
`get_parameters`; no new benchmark-only config API is needed. Preflight verified
that the EQA and OVMM profiles retain the selected query options and model.
SAM2 source is pinned to `2b90b9f5ceec907a1c18123530e92e794ad901a4` for this run.
Using it through `PYTHONPATH` avoids changing the shared Habitat installation.
Habitat's installed executable sets its required C++ library path.

Review the saved grounding RGB/masks/poses, agentic traces, navigation receipts,
and EQA map/video bundles. Inspect final accepted support manually, especially
any positive absent-query response. A new contamination or wrong-object
acceptance blocks promotion even if aggregate correctness improves. Retain
zero-result failures and infrastructure exclusions. No full sweep is authorized
by this pilot; use its failures to select the next shared-mechanism fix.
