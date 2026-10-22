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

## Caliban remote agent repeatability trials

Source `3e61f34e`, managed job `20261013_215016_50a36b`, artifacts:
`/home/cpaxton/runs/emet/native-caliban-agent-20261013/`. Caliban was reachable
through the configured SSH alias/LAN despite its stale supplied Tailscale entry.
The existing server supplied `Qwen/Qwen3-VL-8B-Instruct` in float16 on Orin;
no local model weights or local GPU reservation were needed. Model revision:
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Remote serving source SHA:
`d07af78ac75d844ebcede886eada7e3b3b231828`; server file SHA256 and runner are
archived under `validation/`. The server uses `do_sample=False`. Local seed 0
does not seed the remote model. These are fresh fixture repetitions with new
opaque handles, not independent random samples or held-out scenes.

The same two initially unmet goals, 12-turn budget, CHAT prompt/parser/dispatcher,
and independent whole-footprint/support/stillness scorer were used throughout.
No controller, planner, prompt, or scoring changes were made during the batch.

| Trial | Both goals | Turns | Commanded transfers passing physical checks | Model-call seconds |
| --- | --- | --- | --- | --- |
| 1 | Fail | 12 | 2 (cylinder twice) | 98.09 |
| 2 | Pass | 6 | 2 | 66.73 |
| 3 | Pass | 5 | 2 | 59.17 |

**2/3 episodes passed; 6/6 commanded transfers passed their physical checks.**
These denominators measure different things. Each episode issued 186 ordinary
arm/gripper commands. There were no request timeouts or infrastructure failures.
Model-call seconds exclude physics and orchestration. Five relevant tests passed
(three independent region-scoring tests and two existing client-history tests),
and Ruff passed for the remote harness change.

Trial 1 discovered tasks correctly, then passed the cylinder's `task_ref` with
`object_name="blue cube"` when requesting the second move. The native API gives
the handle precedence and ignored the contradictory name, so it moved the
cylinder into the cube's region. Both transfers individually succeeded, but
neither final object goal was met. The model subsequently claimed completion;
the independent evaluator correctly failed the episode. Trial 2 recovered from
one malformed JSON response and completed both goals. Trial 3 completed in five
valid tool turns. The successful trials did not rediscover between transfers,
despite the instruction to do so; existing task handles remain valid.

Next interface fix: reject conflicting handle/name selectors before planning,
with an explicit rediscovery hint; return the selected semantic object and
receptacle in plan/execution results so the agent can inspect what actually
happened. Add contract regressions, then rerun this frozen batch separately.
Do not treat per-operation `ok` as proof that the whole requested task is done.

This remains stationary, privileged, in-process native-contact execution.
Physics pauses during inference. The result does not establish live ZMQ timing,
observed perception, exploration, navigation, or general agent reliability, and
must not be pooled with the earlier local Qwen2.5-3B result.

## Selector validation and repeat trials

Implemented at `57232014` on the same integration branch. Native planning now
rejects contradictory handle/name selectors with `task_selector_mismatch` and
`recovery="rediscover"` before constructing the motion executor. Redundant names
must match the full published semantic label, ignoring case and outer whitespace.
The rejection reports the handle's selected object/receptacle. Successful plans
save these semantic identities, and execution results retain them on success,
stale-plan rejection, or partial failure. Private simulator body IDs stay private.

The targeted native adapter, API contract, and CHAT dispatch regression suite
passed **92 tests, zero failures/errors/skips** using the prescribed CPU affinity
(`taskset -c 0-7,12-31`, one BLAS/OpenMP thread). New regressions check object and
receptacle conflicts before planning/actuation, matching selectors, saved identity
after task-cache removal, and identity on stale-plan failure. Ruff passed. The
initial run without that affinity restriction crashed inside MuJoCo IK during
the existing grasp-search backtracking test; it was incomplete, not a test pass.
The completed suite's XML/logs are archived with the rerun below. The crash's
root cause has not been established by this change.

Managed job `20261013_223600_e935eb`, artifacts:
`/home/cpaxton/runs/emet/native-caliban-selector-fixed-20261013/`.
Model revision and serving-file hash matched the earlier Caliban batch. Prompt,
goals, initial fixture, 12-turn budget, planner/controller, and physical scoring
were unchanged; the API validation and response identity fields changed.

| Trial | Both goals | Turns | Successful transfers | Model-call seconds |
| --- | --- | --- | --- | --- |
| 1 | Pass | 5 | 2 | 56.20 |
| 2 | Pass | 5 | 2 | 55.74 |
| 3 | Pass | 6 | 2 | 67.44 |

**3/3 complete episodes, 6/6 transfers**, with 186 ordinary commands per episode
and no infrastructure failures. Trial 3 recovered from one malformed JSON turn;
the other episodes used five valid tool calls. Both goals remained independently
verified for two seconds. No conflicting selectors occurred in this rerun, so
live model recovery from `task_selector_mismatch` is not demonstrated; refusal
before motion is covered by the targeted regressions. The small 2/3 → 3/3
comparison is useful integration evidence, not a statistical reliability claim
or an ablation separating identity reporting from selector validation.

Remaining gates are unchanged: held-out tasks/scenes, live physics during model
inference through ZMQ, observed geometry, and exploration. No new PR was opened
for this follow-up.

## October 14 stack integration and IK hardening

Reviews approved #209–#212 with non-blocking notes. The #208 prerequisite was
also approved; its duplicate gripper-drift condition was combined, the endpoint
stillness threshold named, and the path-error expression parenthesized. Its 82
targeted tests passed. Integration of #209 passed 98 targeted tests, and #211
passed 63 contract/dispatch tests. The articulation membership concern was already
addressed in `TAMP_TOOLS`; explicit batch-stop coverage now includes
`set_receptacle_state`. Outcome accounting documents that `ToolOutcome.status`
stores the reason code, while the envelope status determines success.

Pose IK now resolves native joint names once before the iterative loop, uses
cached coordinate indices/bounds, and rejects non-finite seed state and solver
parameters, repeated/empty controlled joints, and unsupported joint types before
native kinematics. These changes remove repeated native named-view access at the
previous crash site and guard invalid inputs; they do **not** establish the
original segfault's root cause or prove unrestricted-CPU stability. Keep the
documented CPU affinity and managed-job settings. No acceptance tolerance or
collision check was weakened. Scorer extraction from the script remains an
optional organizational follow-up, with the existing independent scoring tests.

Managed final agent job `20261014_013312_b6bba3`, source `26dd4e06`, completed
**3/3 full two-object episodes and 6/6 transfers**. Results are under
`/home/cpaxton/runs/emet/native-tamp-merge-validation-20261014/`; validation logs
and source-specific protocol/trace files are retained there. Turns were 5, 6, 6;
model-call times were 57.29, 67.55, 67.91 seconds. Each episode issued 186
ordinary actuator commands. Trials 2 and 3 recovered from one malformed response
each. There were no IK, physical execution, or infrastructure failures. Same
Caliban model, fixture, prompt, budgets, scoring, and privileged in-process scope
as before; this remains a development repeatability result.

The combined suite passed **226 tests with no failures/errors/skips**, both
before and after final main integration.
The final main integration at `1d4b7216` preserves the paper changes and has
identical `src/`, `scripts/`, and `configs/` content to the passing agent source.
The remaining milestone is live ZMQ agent execution with physics continuing
during inference, then held-out tasks and observed exploration. These gates
remain separate from this merged native-tool foundation.
