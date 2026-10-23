# Household multi-step agent TAMP experiment

Status: design and integration plan; no household benchmark scores yet.
Date: October 18, 2026.

Expanded implementation: [shared-agent MuJoCo benchmarks](2026-10-22_mujoco_tamp_benchmarks.md)
adds six-suite coverage and matched planning methods. The historical design and
results below remain unchanged.

## Decision

Use **RoboCasa365 composite tasks as the primary public benchmark** and
**MolmoSpaces fixed episode specifications as the secondary scene-diversity
track**. Keep the latest packaged-table run as an integration regression:
2/6 complete tasks, 6/12 satisfied goals, 6/6 attempted transfers. It is not a
household benchmark result. Machine-readable design:
`configs/eval/household_agent_tamp_protocol.json` (not an executable registry).

Primary sources checked October 18:

- [RoboCasa365](https://robocasa.ai/): household kitchen task benchmark.
- [Composite tasks](https://robocasa.ai/docs/build/html/tasks/composite_tasks.html):
  published composite task collection.
- [Task-length curriculum](https://robocasa.ai/docs/build/html/benchmarking/lifelong_learning.html):
  official 2–3, 4–5, and 6+ stage groupings; use these to stratify tasks, without
  claiming that this experiment is the official lifelong-learning protocol.
- [MolmoSpaces evaluation](https://github.com/allenai/molmospaces/blob/main/molmo_spaces/evaluation/README.md):
  fixed JSON episodes and external policy integration.
- [MS benchmarks](https://allenai.github.io/molmospaces/ms-bench/):
  released pick, pick/place, and open/close task sets.
- [MB benchmarks](https://allenai.github.io/molmospaces/mb-bench/): harder
  manipulation and relational-placement component tests. Some use Filament;
  do not silently substitute renderers in official comparisons.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): useful optional
  robosuite-based tabletop comparison; requires its own robot/controller adapter.
- [BEHAVIOR-1K](https://behavior.stanford.edu/): useful long-horizon household
  comparator but uses OmniGibson, so defer its additional simulator integration.

MolmoSpaces's released component tasks must not be advertised as our new public
long-horizon benchmark. Composing them or changing the robot produces an
**Emet household composition suite**, with upstream provenance, not an official
leaderboard score. RoboCasa official scores likewise require preserving its
robot, task, resets, horizon, observations, and success protocol. A Stretch or
Sourccey adaptation gets a separate result table.

## Current capability gap

`NativeTableAdapter` defaults to the packaged Stretch model and validates model
signatures. The current actual-CLI evaluator scores exactly `object1/object2`
on that fixture. Its read-only connection explicitly requires camera-free mode.
It cannot be pointed at an arbitrary RoboCasa/MolmoSpaces scene by changing YAML.
The existing `eval_tamp_clutter.py` explicitly describes latch/teleport controls;
its deterministic chain is not a native-contact LLM benchmark.

Before household scoring:

1. Provide an explicit scene-model binding, robot profile, joint/EE/gripper
   topology, and boot/model identity check for the live server. Match the full
   scene without pretending a table-scene signature validates it. Keep native
   private-MjData planning, dense arm/payload checks, and ordinary actuators.
2. Replace body-name-specific discovery/goals with episode-bound object and
   receptacle roles, world frames, support/containment geometry and access state.
   Handles remain opaque; object categories and benchmark names are test data.
3. Preserve upstream task reset metadata and success checks where the upstream
   environment is retained. A merged XML alone is not the upstream task runtime.
   Otherwise bind independent measured predicates and label the task adapted.
4. Add generic full-episode scoring: conjunction of final goals, protected prior
   goals, causal/ordering predicates, sustained support/containment, no illegal
   contacts, and explicit infrastructure-failure accounting.
5. Admit native articulation and measured base navigation only after their own
   live tests. Simulator-assisted articulation gets a separate named condition;
   never count joint qpos setting or attachment as native physical execution.
6. Extend the production `run agent` route rather than scripting the chosen
   sequence. Keep complete user instructions in context, rediscover after changes,
   and preserve failure-stop behavior. Fix shape guidance for `placement_region`;
   any safe pre-execution validation retry needs a separately reviewed policy.

Do not remove the model gate, enable observed execution without providers, or
relax collision/freshness thresholds to make the new suite launch.

## Task families: dependencies rather than longer pick lists

| Family | Requested final goal | Required dependency | Initial tier |
| --- | --- | --- | --- |
| Clear and serve | Serving object occupies an initially obstructed destination; blocker ends in safe storage | Relocate blocker to a free buffer before placing the serving object | Native pick/place, one work area |
| Swap occupied destinations | Two objects exchange occupied regions; buffer is empty at completion | Temporary placement breaks the occupancy cycle; usually three transfers | Native pick/place, one work area |
| Retrieve and close | Requested object is outside an initially closed cabinet; cabinet ends closed | Open, retrieve, place, then close; preserve object goal | Articulation-gated |
| Collect and stow | Objects from two work areas end in designated storage | Acquire/ground observations, navigate, transport and place, preserving completed goals | Navigation/observed-input gated |

These are proposed composition templates, not claimed upstream task IDs.
Resolve actual upstream IDs, object instances, layouts and fixture affordances
before registry freeze. Local RoboCasa source includes `ClearSinkArea` and
`PlaceFoodInBowls`; inspect their reset and `_check_success` implementations as
initial candidates. The latter starts with cabinet doors open in the inspected
implementation, so it must not be described as an open-door benchmark.
Containment, articulated access and multi-room tasks require additional support;
do not flatten them into XY rectangles and call the original task solved.

The agent gets the natural-language final goals and allowed scene observations,
not an expert tool list, the reference ordering, or private simulator body IDs.
The evaluator checks dependencies from measured transitions, not LLM text.

## Frozen sampling and admission

Start with an **8-case integration pilot**: four families by two simulators,
one scene/reset each. This is a capability audit, not a paper success estimate;
unsupported families must be reported as blocked rather than substituted.

The proposed full adapted suite has four families, three held-out layouts per
simulator, and two object-placement seeds: **48 distinct configurations**.
Run three independent agent trials per configuration: **144 episodes per
robot/input/assistance condition**, not 144 independent problems. Development
uses two different layouts per simulator and disjoint seeds. Actual layout IDs
and upstream releases remain unresolved until compatibility inspection; the
protocol JSON contains no fabricated episode IDs or executable fixtures.

Before the agent sweep, freeze the entire candidate population, sampling rules,
asset/source hashes, budgets, and capability admission decisions. Replay complete
reference solutions through normal actuators for a predeclared physically
solvable subset. A high-level symbolic plan or simulator set-pose is not a
reference certificate. Keep reference traces private from the agent.

Publish both intent-to-evaluate counts over the fixed candidate population and
success on the predeclared supported/reference-certified subset. List unsupported,
construction and reference failures separately. Never remove an agent planning,
grasp, API or infrastructure failure after seeing its outcome. No post-hoc
resampling of a difficult seed. Hold-outs stay frozen during hill climbing.

## Conditions and baselines

1. **Scripted/reference sequence + same motion tools:** isolates execution and
   geometry limits. It may read privileged state and is labeled accordingly.
2. **LLM + privileged grounding + native motion:** primary orchestration test;
   Caliban inference, local simulation only. No simulator attachments/teleports.
3. **LLM + observed grounding + native motion:** separate later end-to-end gate;
   requires working camera/tracking, navigation and observation provenance.
4. **Ablations:** remove rediscovery or omit the task-level dependency representation
   on the same frozen episodes. Do not disable physical checks to define an ablation.

Start with Stretch; add the corrected Sourccey model as a separate robot condition
once the shared profile/scene binding works. Retargeted robot results are not
pooled with upstream native-robot benchmark results. Any privileged articulation
assistance is a further separate condition, never mixed into condition 2.

Pin simulator and task-library revisions, MuJoCo, robot/asset hashes, remote model
revision and decoding parameters where the server exposes them; record what could
not be pinned. Fix agent/tool, wall/sim-time and motion-search budgets beforehand.
Use jobs-managed runs, unique ports, two simulations in parallel initially, and
Caliban for model inference. GPU rendering requires a separately scheduled slot;
do not assume CPU-only kitchen rendering is fast enough for observed freshness.

## Metrics and paper artifacts

Report full task success first; also final-goal fraction, stage/dependency success,
protected-goal regression, success conditional on attempted execution, API shape
errors, planning/IK/collision/tracking/retention failure taxonomy, stop compliance,
audit violations, calls, runtime and assistance. Show confidence intervals by
configuration/layout rather than treating repeated trials as independent scenes.

Figures: a multi-step video/keyframe strip with task/active-skill borders; success
versus dependency length; failure breakdown; and matched task success by input
condition. Keep planned experiments and preliminary table results explicitly
separate from completed household measurements in `paper/sections/04_experiments.tex`.

## PR sequence

- Finish #213 with the reviewed orchestration boundary, measured pregrasp fix,
  exact baseline evidence and overview docs. Preserve its draft status; no
  household success claim is a merge condition for that integration milestone.
- One focused follow-up for scene/profile binding, generic episode/scoring
  contracts and the 8-case household pilot. Add selected upstream task adapters
  there; do not bolt kitchen bodies onto `NativeTableAdapter`.
- Freeze the certified registry, run the larger agent sweep, then add measured
  household results and figures to the paper. Camera/exploration and Sourccey
  conditions follow their own admission gates.
