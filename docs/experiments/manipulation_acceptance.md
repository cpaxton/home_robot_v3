# Bounded shared-agent acceptance

This is the next acceptance battery for the review branch, not a report of
passing results. The goal remains **one learned agent across EQA, OVMM and
multistep manipulation**, with robot adapters rather than task-specific oracle
policies. See [current evidence](shared_grounding_pilot.md) and the
[environment progression](../environments/README.md).
The [PR review map](../plans/PR167_review.md) separates the dependency layers
and lists the evidence needed before landing this mixed change set.

## Frozen settings and reporting

Use Qwen3-VL int4, the separate `query_detector_segmented_pilot.yaml` preset
(YOLOE proposals → SAM2 support → Qwen verification), and `lazy_graph` for the
initial manipulation gates. Preserve geometry, freshness, collision and motion
completion checks. Ground-truth names, masks and trajectories are evaluator-only;
do not expose them through the agent's observations or use scripted grasp/teleport
actions. Record source SHA, effective configuration, seeds, model/endpoint,
commands and budgets for each case. Do not edit running source.

Contact-depth and geometry-servo development presets are separate diagnostic
rows, not replacements for that frozen control. In particular,
`query_geometry_contact_pilot.yaml` specifies a robot-frame contact-point
offset; its value is not a universal object-localization correction. Published
RGB-D geometry and absolute manipulation targets use world coordinates, while
`get_base_pose()` is episode-relative on ZMQ clients. Use `get_base_pose_world()`
for those transforms and explicitly world-frame navigation goals. Robot tool
offsets must use the published URDF grasp axes, not a visually inferred axis
from a simulator marker. Validate nonzero episode origins and wrist rotations.

Native Stretch snapshot timing is transported as per-camera `image_timing`
through full and servo observations. `timestamp_ns` is simulation time from
the rendered state, with `clock_domain: mujoco_sim` and
`source: render_state_snapshot`; it is **not Unix time or hardware exposure
time**. Compare it only to the same simulator episode's physical trace, not
directly to host clocks. Republishing a frame retains its stamp. Legacy peers
without timing remain unknown. Separately sampled joint feedback and the
state-only FK stream are not thereby acquisition-synchronized.

Run heavy cases **serially**, through `emet jobs --cpu-safe --gpu-exclusive`.
Use fresh artifact directories and bounded subprocess deadlines. After a timeout,
check simulator/model cleanup before starting the next case. No real robot is
needed for this battery. Never push to main or merge on a diagnostic smoke.

The original tabletop diagnostic is runnable through the existing driver:

```bash
OUT=/absolute/fresh/output PHASE=manipulation AGENT_PY=/absolute/venv/bin/python \
  bash scripts/run_shared_grounding_pilot.sh
```

Submit that command through the managed job launcher above, from the frozen
checkout. `PHASE=manipulation` skips the absent-query exploration and Habitat
tasks. `PHASE=eqa` runs the existing paired hybrid/Qwen-box questions without
Habitat-OVMM; it is not yet the four-row Stage E panel. Avoid `PHASE=all` for this
staged battery. The driver fails if independent tabletop physical scoring fails,
even when the agent process exits zero.

For explicit simulator-only ablations, set `SIM_AGENT_CONFIG` and `SIM_CONFIG`
to the selected agent and scene YAML paths. Defaults remain the detector preset
and original tabletop; these overrides do not change Habitat/EQA rows. The
driver saves both selected files, hashes and the exact command alongside agent
process status and independent physical scoring. For example, the current
separated-neighbor contact diagnostic uses
`SIM_AGENT_CONFIG=configs/emet/query_geometry_contact_pilot.yaml` and
`SIM_CONFIG=configs/sim/default_table_stretch_clearance.yaml`. It is not an
original-clutter acceptance result.

`EMET_SIM_EVAL_CONFIG` selects explicit evaluator-only object/support/EE/gripper
body names, and `EMET_SIM_EVAL_TRACE` selects a fresh JSONL destination. Neither
is part of observations or tools. The Stretch simulator records at 10 Hz in
simulation time, with wall time for matching wrist captures, poses, contacts,
qpos and actuator targets. Score offline with `python -m
emet.eval.manipulation_trace TRACE --output RESULT`. This opt-in has no effect
on ordinary runs. Agent-side `manipulation_outcomes.jsonl` remains a separate
tool report with `physical_success_verified: false`.

Report process completion, tool outcome and independently measured physical
outcome separately. The existing OVMM object-displacement and receptacle-distance
scores are proxies: a knocked object is not a successful pickup. Missing evidence
is unverified, not success. Retain negative cases and infrastructure failures.

## Stages and stop gates

| Stage | Bounded cases | Advance only when |
| --- | --- | --- |
| A: causal diagnostic | Original red-cylinder/blue-cube task; stationary and known-motion controls as needed | Accepted/rejected wrist sequence and measured commands explain the failure; independent pickup evidence is available |
| B: basic manipulation | Original fixture twice, plus two predeclared pose/clutter variations twice (6 total) | Original 2/2; at least 5/6 overall; each variation passes; no falsely reported physical successes |
| C: single-room OVMM | Stretch in RoboCasa and MolmoSpaces; visible and search starts; seeds 0/1 (8) | At least 6/8 and 3/4 in each environment, independently verified pick and place |
| D: learned TAMP | Two open-receptacle pick/place steps in each environment, seeds 0/1 (4) | At least 3/4, at least one pass per environment, all ordered subgoals verified |
| E: paired regression | EQA q15/16/25 and one find per room, each on four frozen rows; two absent-object candidate controls; one rby1 MolmoSpaces navigation/find control | Report paired differences and false acceptances; investigate baseline-correct/candidate-wrong cases before promotion |

Stage E rows: frozen `a2021391`, repaired candidate, candidate `dynamem_voxel`,
and candidate `static_no_instance` (names from `configs/benchmarks/paper_eval.yaml`).
These are **system comparisons**, not isolated ingestion/fusion ablations. Hold
model, exploration and budgets fixed. Repeat discordant EQA cases with seeds 1/2;
do not claim statistical equivalence from three questions. Specify the two varied
fixtures, room task bodies/supports and effective row configurations *before*
launching those stages. Unspecified cases are pending, not implicit passes.

Stage B fixtures are now predeclared: original
`default_table_stretch.yaml` (blue neighbor x=-0.02 m), separated-neighbor
`default_table_stretch_clearance.yaml` (x=-0.25 m), and mirrored-clutter
`default_table_stretch_right_neighbor.yaml` (x=+0.18 m). The target remains at
x=+0.08 m; the mirrored fixture retains the original 10 cm center separation
on the opposite side. Model-equality tests check that only the neighbor pose
changes. Run each twice on one frozen candidate/configuration. Earlier
development passes on different source or aperture settings are not pooled
into that six-case panel. Keep the detector preset as the frozen control;
the current development candidate uses the separate
`query_geometry_recovery_pilot.yaml` preset, which inherits contact/aperture
settings and opts into one bounded alternative proposal pass. Record that
difference explicitly; do not pool its results with the contact/aperture control.

The next **v6 numerical-physics panel** freezes
`query_geometry_tracked_narrow_pilot.yaml` across all six cases and uses
`default_table_stretch_noslip.yaml`, `default_table_stretch_clearance_noslip.yaml`
and `default_table_stretch_right_neighbor_noslip.yaml`, in that order, twice
each. These wrappers change only `noslip_iterations` from 0 to 10; compiled-model
tests check every fixture against its original. The source SHA is recorded by
each driver invocation. This is an explicit solver ablation, not promotion of
new production physics. Do not pool the earlier mirrored development pass or
default-solver panel results. All cases retain seed 0 (execution repeats, not
independent environment seeds), Qwen int4, original task and physical scorer.
Stop at the first failed case for diagnosis; remaining cases are unrun, not
failures or passes. Render each completed case using its matching wrapper XML.

V6 stopped with original 2/2 and clearance repeat 1 failed (three unrun).
V7 repeats this exact order/configuration on `ef533ed3` after fixing the
rendered-camera versus published-grasp calibration mismatch. A separate
clearance retry passed; it is excluded from the new six-case panel.

For Stage D, submit the two-step request through the learned CHAT loop, not two
externally scripted tool invocations. The shared loop permits three tool-bearing
rounds, including observation and action results, followed by a no-tool summary.
Failure stops the remaining batch and turn; completed actions must not be
replayed. Preserve model tool-call traces and score both ordered physical
subgoals independently. Offline loop tests are not multistep physical evidence.

The private recorder also accepts an ordered `steps` list for **distinct target
objects**, with shared `ee_body` and `gripper_bodies`. Each step supplies
`object_body` and `support_body`; these are evaluator-only simulator body names,
not handles supplied to the agent. `EMET_SIM_EVAL_TRACE` then names a schema-2
manifest and sibling `.step-N.jsonl` traces recorded on the same simulation
timeline. The usual `python -m emet.eval.manipulation_trace MANIFEST --output
RESULT` scores every step with the existing physical thresholds, checks that
the next verified pickup starts after the previous verified release, and
requires earlier placements to remain intact at episode end. Repeated use of
the same target object is rejected rather than implicitly counted twice.
Missing, mismatched, out-of-order or incomplete evidence does not pass. Render
the individual schema-1 step traces, not the manifest, with the existing replay
renderer. This connects evaluator plumbing, **not** room-task generation or a
passing learned-TAMP experiment; room bodies/supports and instructions must
still be frozen before execution.

Physical pickup acceptance requires at least 5 cm lift, gripper contact and stable
object-to-gripper pose for one simulated second, without support contact. Placement
requires release onto the designated support and a stable pose for one simulated
second after pickup. Save pose/contact traces privately for offline scoring and
visual audit; the agent must never read them. Calibrate these evaluator checks on
known positive and knocked/dropped/wrong-support negative controls before using
them as gates. These are simulator acceptance criteria, not real-robot guarantees.

When a case fails: preserve RGB-D, masks, raw VLM prompts/responses, calibration,
timing, command receipts and measured motion; write one causal hypothesis; add a
focused test; rerun that exact case and a neighboring control. Two unchanged
failures stop that stage for diagnosis. Do not advance to expensive room tasks
while the simple physical manipulation gate fails.

## Articulation is deferred

The separate `configs/sim/robocasa_counter_to_sink_stretch.yaml` selects the
installed `PickPlaceCounterToSink` task with Stretch, layout/style 1 and seed 0;
`--sim-seed 1` supplies the second predeclared seed through the existing CLI.
The original closed-cabinet fixture is unchanged. This is a **fixture candidate**,
not room acceptance: preflight generated assets, task-object/support bodies,
visible/search starts and effective contact solver settings before freezing the
room manifest. Do not silently transfer the tabletop NoSlip ablation to a new
environment or claim it used identical physics without checking.

Door/drawer opening and closing are **unsupported**, not successful no-ops.
Use open or explicitly pre-opened receptacles for this PR's TAMP cases, and label
that fixture intervention in results. `run_tamp_agent_tools_gate.sh` is an oracle
control, not learned TAMP acceptance. Later articulation work needs handle/axis
grounding, constrained contact-aware motion, bounded force/travel, recovery, and
independent joint-state scoring. See [TODO](../../TODO.md).

## Evidence and paper handoff

Archive accepted as well as rejected wrist frames, before/after views, top-down
routes and chase-camera views where available. Include a failed case, not only a
successful montage. Link artifacts and exact commands from the experiment report.
For private physical traces, `scripts/render_manipulation_trace.py TRACE --scene
SCENE.xml --output FRESH_DIR` produces overview, top-down and object views at
initial, pickup, last-contact and final states. Run in the trace's frozen
checkout under the exclusive GPU lock. These are explicitly labelled qpos
reconstructions, not agent images or new executions; retain the manifest.
Update environment notes and paper claims only after scoring: distinguish view
evidence from localized instances, system rows from causal ablations, and bounded
single-room manipulation from unresolved long-range Habitat-OVMM. Sourccey,
Galaxea, real-robot manipulation and learned rby1 manipulation remain follow-ups.
