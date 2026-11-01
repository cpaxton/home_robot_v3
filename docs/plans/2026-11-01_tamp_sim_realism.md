# TAMP simulations with increasing realism

New experiments exclude Stretch. Start with the installed upstream Panda in
robosuite; keep Sourccey as a separately qualified robot-transfer track. This work
is on `feat/tamp-sim-realism`, stacked on the reviewed #226 foundation, rather
than expanding that PR. Machine-readable protocol:
[`configs/eval/tamp_sim_realism.json`](../../configs/eval/tamp_sim_realism.json).

The installed native public-tool adapter is currently Stretch/table-specific.
Neither Panda nor Sourccey can acquire native TAMP merely by changing its robot
name. Sourccey's corrected model/kinematic manipulation support does not qualify
native contact pick/place. The first gate is a truthful robot/controller binding
for the existing tool and task-session contracts.

## Progression

All executed tiers use native MuJoCo contact and ordinary controllers. Private
planning probes may change their own state; executing object attachment, pose
teleport or joint teleport never counts as physical execution. Change one input
or disturbance axis at a time and retain the matched control condition.

| Stage | Experiment | Pass gate |
| --- | --- | --- |
| R0: native reference | Upstream Panda Lift, then Stack pick/place/release, with exact-state scripted references and hold controls | Original success plus sustained grasp/retention and settled post-release support; hold controls fail the task. These are reference diagnostics, not TAMP/agent results. |
| R1: exact geometry TAMP | Same robot, full native environment and shared public primitive/multi-step tools, with privileged geometry explicitly declared | Collision-checked arm and payload paths, measured convergence, no forbidden assistance, two reference replays using the same motion tools. Bind the upstream controller to the shared planner; direct scripted OSC success alone is insufficient. |
| R2: clutter/composition | Add blocked straight paths, narrow placement spaces and buffer dependencies | Alternative collision-free paths, temporary relocation, protected goals and partial-progress reporting. Include intended failures; bounded search failure is not proof of impossibility. |
| R3: measured geometry, oracle masks | RGBD and known robot/declared object priors, with oracle instance segmentation isolated as assistance | Qualified spatial/timing alignment, complete-object bounds, support and persistent background evidence, actor/phase-specific contact admission. No hidden scene-object poses. This remains a privileged/assisted condition in public receipts, not fully observed success or a new public input mode. |
| R4: observed grounding | Same frozen cases with Caliban grounding and continuous measured tracking | Unknown-space and lost-track stops, fresh effects after release/retreat, explicit re-observation/replanning episodes through the same tools. No pose fallback. |
| R5: timing/physics sensitivity | Live asynchronous physics; test latency, stale/drop events, then declared mass/friction/actuation variations separately | Matched live-async baseline, honest possible-dispatch receipts, safe failure boundaries and measured completion. Parameter sweeps are sensitivity tests until sensor/contact ranges are independently calibrated. |

Task complexity is a separate axis: single transfer, buffered swap, clear-then-place,
then collection. Introduce multi-step plans in R2 with exact geometry first and
repeat the same cases under R3/R4. Keep the production CHAT agent, model/checkpoint,
prompts, JSON tool schemas and budgets matched. Scripted reference controls certify
construction/motion separately; they do not replace the agent condition.
Buffered swaps and clearing layouts are adapted composition tasks, not official
Lift/Stack benchmark scores; report their task goals separately from the original
environment's success predicate.

After the robosuite bindings pass, move the same applicable tests into full
RoboCasa tasks and fixed MolmoSpaces episodes. Preserve the upstream reset,
horizon/controller and original success check; scene XML export alone is not an
upstream benchmark integration. Articulation/navigation enter only after their
native execution/source gates pass. Reuse the existing benchmark runtime, registry,
single-use plans and task search rather than introducing another agent loop.

## Cases and measurement

Keep development seeds 7/13 separate from the scored population. Construct ten
candidate agent configurations spanning transfer, clutter and buffer dependencies;
freeze actual initial state, compiled model, robot/controller, goal and source
identity before agent evaluation. A seed alone is not a reset certificate. Require
two verified native reference replays per certified case through the same motion
tools. Report construction/reference/unsupported failures in the fixed population,
as well as the certified subset. Never remove an agent failure after looking at it.
Three agent repeats per condition are an initial pilot, not a precise success-rate
estimate. Match cases and execution mode when comparing input or planning conditions.

Retain upstream success and stricter goal/support/retention/integrity separately.
Report completed operations, failure phase/code, native control steps including
gripper waits, planning/IK/LLM budgets, wall time and forbidden-command audit.
Generate videos with task, active skill, input tier and stop reason in the border.
Store raw traces/captures in managed artifacts and commit compact JSON manifests.
No force/contact sensors are implied on hardware; simulator contact is an
independent scorer/debug channel unless explicitly named as privileged input.

## First implementation

`scripts/eval_tamp_panda_reference.py` runs full upstream Lift/Stack environments
through `RobosuiteAdapter` and `EpisodeRuntime`. Normal native OSC and gripper
actions perform all motion; the upstream horizon is 1000, while the diagnostic
has a separately declared 400-action cap. Controller frame/configuration, model
XML, initial simulator state, actions and original scores are recorded.
The final diagnostic also archives compiled models, initial actuator controls,
the runner source and original upstream task source. Reference/hold pairs require
matching state, controls, model XML and controller configuration. These captures
do not by themselves satisfy the complete benchmark reset certificate.

These OSC references qualify the upstream task/actuation path only. R1 must bind
an action interface that follows the motion planner's chosen joint paths and
requalify its references under that declared controller profile before freezing.
An OSC end-effector target does not certify tracking of a redundant joint-space
RRT path. Do not transfer the present reference result to a changed controller
or count it as a public motion-tool certificate.

The reference requires 30 consecutive retained-lift samples, bilateral grasp
contact, at least 8 cm object lift and at most 1 cm relative-position spread.
Stack additionally requires 30 post-retreat samples with released support contact,
at least 10 cm EE separation, stable object position, and expected center/height
over the support. These are ideal-simulator diagnostic checks. Global collision
integrity and reset certification remain unqualified. Hold controls are negative
task controls; their expected failure must never be counted as a successful task.

Launch through `emet jobs run --cpu-safe --no-gpu-exclusive -- ...` with
`MUJOCO_GL=disable`, `CUDA_VISIBLE_DEVICES=` and single OpenMP/BLAS threads:

```bash
python scripts/eval_tamp_panda_reference.py --output RUN_DIR --tasks Lift Stack --seeds 7 13
```

This uses no renderer or VLM. GPU use is unnecessary for R0; Caliban remains the
planned inference endpoint for agent conditions. Start with at most two CPU
simulation jobs. Do not infer an observed task success or an agent score from
this reference pilot. See the [first results](../experiments/tamp_sim_realism_20261101.md).
