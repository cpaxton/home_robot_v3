# Native multi-object goals and grasp/placement search

Implementation validation revision: `0841c997` (stacked on #211).
Scope: stationary Stretch, packaged table, privileged simulator geometry,
ordinary arm/gripper actuation. No native navigation, observed perception,
articulation, or hardware claim.

## Why the old sequence failed

Managed diagnostic `20261013_195605_947f9a` recorded every candidate for the
cylinder at the back of the two-object fixture. Its first tilted grasp reached
pregrasp but the grasp endpoint collided with the box through the gripper body
(about 4.6 cm penetration in the rejected hypothetical state). Later candidates
failed IK. The previous response exposed only the last IK error, hiding the
initial collision. No rejected path was executed.

Steeper center grasps cleared the blocker but put the fingertips through the
table. Bounded height candidates derived from object extent permitted a native
cylinder transfer without changing collision tolerances, grip commands, gains,
or retention criteria. The subsequent box task exposed the missing goal-location
contract: a locally free placement is not necessarily a useful placement for the
next task.

## Implementation

- `plan_pick_place` accepts optional world-XY `placement_region` bounds. Only the
  native privileged adapter currently supports this constraint. Other adapters
  explicitly reject it. The planner clips support patches and preserves its
  existing 2 cm clearance margins; execution verifies the full released footprint
  in the region (5 mm tolerance), support, stillness, and the existing pose checks.
- Bounded 45/60-degree grasps and geometry-scaled pinch-height alternatives follow
  the original 15-degree candidate. These are robot/object geometry rules, not
  object-name features.
- A grasp without a feasible downstream placement now advances to another grasp.
  A shared 256 pose-IK-call budget spans the complete search. Failure is reported
  as `no_plan_within_budget`, with sanitized per-stage rejection counts. This is
  not a proof of infeasibility. Raw contacts stay in privileged diagnostic events.
- The two-object fixture has explicit final predicates. Both are false initially:
  the cylinder must move into the back region and the box into the middle region.
  Region coordinates are checked in at
  `configs/eval/native_table_two_object_regions.json`, cylinder then box.

## Development diagnostics

All directories below are under `/home/cpaxton/runs/emet/`. Early runs recorded
HEAD `9aace83e` while planner/harness edits were uncommitted; they are diagnostics,
not acceptance evidence for that SHA.

| Directory | Finding |
| --- | --- |
| `native-steep-r1` | Steeper center grasps rejected for fingertip/table collision. |
| `native-steep-r2` | Raised, geometry-derived grasp completes cylinder transfer; subsequent unconstrained box placement search fails. |
| `native-regions-r1` | Both transfers pass, but the box's broad region was already satisfied initially; not a valid two-goal benchmark. |
| `native-regions-r2` | Tightened box region is too narrow for the unchanged clearance margins; no complete sequence. |
| `native-regions-r3` | Both initially unmet region goals pass, and the earlier placement remains stable. Freeze this fixture before comparing selectors. |

## Evaluation boundary

The scripted reference and LLM selector use the public registry and CHAT
dispatcher. The LLM harness additionally uses the same `AgentPromptBuilder`,
response parser, and model-call helper as `run agent`. It is not the interactive
CLI or live ZMQ transport. Physics advances through the in-process actuator
facade and pauses during model inference. No visual perception is evaluated.

The selector receives both region goals and may choose the order. It must discover
real task handles and execute real returned plans. Independent scoring requires
both full footprints in their regions, table contact without fingertip contact,
and low linear/angular speed for two seconds. Language claims of success cannot
pass this evaluator. Partial execution stops the episode because no held-object
recovery skill is available. The prompt, seed, raw model responses, tool outputs,
commands, and final predicates are archived. There is a 12-turn limit.

These are development fixtures, not held-out generalization rates. Keep the
scripted native reference, LLM selection, repeated-object regressions, and earlier
live single-transfer results as distinct denominators. A dedicated task-order
search/MCTS comparison and live agent acceptance remain separate work.

## Frozen native validation

Managed job `20261013_200940_892d7d`, root
`/home/cpaxton/runs/emet/native-sequence-final/`, at `0841c997`:

- `reference/`: **2/2 complete two-object episodes**, four transfers. Both goal
  regions are initially unmet. Each episode uses fresh physics, reacquires task
  handles between actions, and verifies earlier released placements after later
  transfers.
- `regression/`: **2/2 repeated-object episodes**, four transfers, retaining the
  previous box/cylinder controls. Separate denominator from the reference.
- **98 targeted tests passed**, no failures/skips; five relevant cases were rerun
  after the last scorer/diagnostic refinements. Ruff and diff checks passed.

The native planner, controller tolerances, and goal fixture are unchanged in
`d267fbe4`; that commit only adds an explicit model-device option for the agent
evaluation. The CPU model attempt `20261013_200941_1b2eab` was cancelled before any
action once the GPU became available; it is unscored infrastructure work. The
exclusive managed GPU comparison is `20261013_201226_25071d`, under
`/home/cpaxton/runs/emet/native-sequence-llm-gpu/`.

## LLM-selected native execution

The first Qwen2.5-3B-Instruct trial at `d267fbe4` failed: **0/1**, 12 turns,
zero robot commands. It confused the `robot` argument with input mode, flattened
or stringified the region array, and invented task references. Inspection showed
that CHAT's local-model prompt listed argument names but dropped their JSON
Schemas and descriptions. The API-style schema existed but was not delivered to
this text-only client.

At `e1a76582`, CHAT includes those schemas, the argument descriptions explicitly
distinguish robot identity and input mode, and unknown native task references
request rediscovery. The same model, seed 0, fixture, goal regions, and 12-turn
budget then passed: **1/1 complete two-object episode**, six model turns,
186 ordinary arm/gripper commands. The model recovered from an invalid robot
argument and a plan-handle-as-task-handle error. It used the supported semantic
object/receptacle-name planning path; no successful standalone discovery call
was required, and it did not invent a successful execution result.

Managed job `20261013_201646_751247` completed successfully; artifacts are under
`/home/cpaxton/runs/emet/native-sequence-llm-schema/`. Both final footprints were
inside the requested regions, supported without fingertip contact, and still
for two seconds in the independent evaluator. Model-call time summed to 7.55 s;
this excludes model loading, physics, and orchestration and is not episode time.
Model cache revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

This comparison establishes integration on one development fixture, not general
LLM reliability. The prompt/schema/handle-guidance changes were evaluated together;
there is no isolated ablation attributing the result to any single change.
The final code passed **124 targeted tests with zero failures/skips**, including
prompt schemas, native planning/execution, independent goal scoring, JSON API,
CHAT dispatch, and CHAT/EQA skill-pack separation. Full XML/logs are archived in
`native-sequence-final/validation/`.

Remaining gates: repeated/held-out agent trials, live ZMQ agent timing and scene
changes, task-order search comparison, observed geometry and exploration. The
interactive `run agent` CLI and a full navigation/manipulation episode have not
been validated by this in-process test.

The schema change applies to all local CHAT tool descriptions, not only TAMP.
For the default empty-context registry (30 tools), the tools block grows from
9,012 to 12,524 characters. Structured API tool schemas and EQA tool membership
are unchanged. Full interactive-session model behavior still needs its own check.
