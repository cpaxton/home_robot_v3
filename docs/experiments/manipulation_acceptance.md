# Bounded shared-agent acceptance

This is the next acceptance battery for the review branch, not a report of
passing results. The goal remains **one learned agent across EQA, OVMM and
multistep manipulation**, with robot adapters rather than task-specific oracle
policies. See [current evidence](shared_grounding_pilot.md) and the
[environment progression](../environments/README.md).

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
Update environment notes and paper claims only after scoring: distinguish view
evidence from localized instances, system rows from causal ablations, and bounded
single-room manipulation from unresolved long-range Habitat-OVMM. Sourccey,
Galaxea, real-robot manipulation and learned rby1 manipulation remain follow-ups.
