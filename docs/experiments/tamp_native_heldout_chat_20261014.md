# Held-out placements and actual CHAT integration (2026-10-14)

This gate follows the [two successful fixed-fixture live trials](tamp_native_live_agent_20261014.md).
It deliberately separates task success from correctly stopping on a failure.
Caliban serves `Qwen/Qwen3-VL-8B-Instruct` at `192.168.1.55:8000/v1`;
local simulation uses CPU affinity `0-7,12-31`, `CUDA_VISIBLE_DEVICES=` and
`no_cameras: true`. No local GPU reservation or image inference is used.

## Frozen held-out placement results

Both new initializations were fixed before running either case. Goals, budgets,
collision checks, and retention thresholds were unchanged from the earlier
regional fixture. Initial positions are world XYZ meters; the original starts
were cube `[-.02, -.55, .6]`, cylinder `[-.02, -.70, .6]`.
The source was `0e1e2e37` (same runtime as the preceding successful batch).

| Case | Cube start | Cylinder start | Task result | Evidence |
| --- | --- | --- | --- | --- |
| Left | `[-.04, -.53, .6]` | `[-.04, -.68, .6]` | Partial at preplace; grasp completed, payload retention failed | `20261014_081312_ce78d5`, `native-heldout-left-20261014` |
| Right | `[0, -.57, .6]` | `[0, -.72, .6]` | Partial at pregrasp; fingertip/table contact | `20261014_081312_c35825`, `native-heldout-right-20261014` |

Artifacts are under `/home/cpaxton/runs/emet/`. **0/2 held-out episodes completed**.
The left run issued 190 audited commands, the right 23; both had zero forbidden
actions and physics progressed during every model request. Both stopped after
partial execution. Neither is a task success, even though the failure boundary
worked. Do not combine this denominator with repeated familiar-fixture successes.

Left: payload translation residual was 0.005275 m, but rotation residual reached
0.100136 rad against the unchanged 0.1 rad limit. The final cylinder was not
independently supported/settled. This is not justification to relax retention.
Right: measured `rubber_tip_right`/`table` penetration was 0.001267 m;
the next linear edge was rejected before grasp completion. Nominal planning
success is insufficient when measured tracking reaches contact. No native crash
was observed in either episode.

The next motion investigation should reproduce the right contact with the
measured-state trace and test clearance under tracking error, then evaluate
transport/grasp stability on the left. Preserve failed starts and compare paired
reruns without changing goals or tolerances. Physical held-object recovery is
still unavailable; do not retry automatically after a partial result.

## Actual run-agent session

`93dd3d1a` adds a bounded TAMP continuation policy to the production CHAT loop.
Successful structured TAMP results may continue the original authorized task;
the follow-up explicitly reminds the model that plan-only requests do not
authorize execution. Any failed, partial, unknown, or malformed
TAMP result forces a terminal summary, and tools returned by that summary are
not executed. A later explicit user turn can request recovery. Non-TAMP tools
retain their prior summary behavior. The default budget remains three rounds;
`--max-tool-rounds 12` gives multi-object requests a bounded larger budget.

`--state-only-sim --memory-backend dynamem` explicitly selects a camera-free
Stretch diagnostic session. It disables perception model initialization and
exposes only TAMP tools plus `quit`; the client still requires simulation metadata
advertising zero cameras. This exercises the actual CHAT executor and loop, but
cannot establish visual grounding, mapping, navigation, or exploration.

`scripts/eval_native_chat_session.py` launches the real
`python -m emet.app.run_agent` entry point in a separate process, then independently
scores released support, settled whole-footprint goals, and the physical audit.
`--reject-first` asks the model to attempt one invalid plan handle in a diagnostic
turn, then submits a separate authorized two-object task. A rejection alone does
not count as task recovery: both physical goals must subsequently pass.

The first CLI attempts (`20261014_081822_fc2fa1`, `20261014_081923_444461`)
were cancelled during startup: the omitted robot address selected the machine's
active connection profile. `53a1a9c0` explicitly sets `--robot-ip 127.0.0.1`.
Two immediate replacement launches (`20261014_082022_8e436f`,
`20261014_082022_ff63e0`) failed before execution due to a temporary runner cwd
substitution error. These infrastructure attempts are retained separately.

The next attempts (`20261014_082045_47aeff`, `20261014_082045_20a8f5`)
reached the local simulator but failed before the first model turn: the scoring
client had issued startup commands and owned the command session. The bridge
correctly refused a second controller. `e82dee57` makes the observer read-only
and adds a test that it issues no startup commands.

At source `e82dee57`, corrected CLI trials:

| Case | Job | Artifact directory | Result |
| --- | --- | --- | --- |
| Familiar fixture, invalid-plan turn then task | `20261014_082257_1476eb` | `native-chat-cli-recovery-observer-20261014` | Rejection verified; cylinder transferred; cube execution refused stale observation |
| Frozen right held-out task | `20261014_082257_ea28d3` | `native-chat-cli-heldout-right-observer-20261014` | Task failed: no accepted plan; correct terminal stop |
| Frozen left held-out task (logging-only source `ebd66db6`) | `20261014_082549_34eadd` | `native-chat-cli-heldout-left-observer-20261014` | Task failed: retention lost during place; correct partial stop |

The actual right-hand CLI session returned `no_plan_within_budget` after 76 IK
calls (cap 256): candidates were rejected for IK/collision before execution.
It summarized and ended the session; no `execute_pick_place_plan` call followed.
Its seven audited commands were session startup/cleanup, with zero forbidden
actions. CLI exit code 0 only means a clean session exit, **not** task success.
This differs from the earlier harness reaching pregrasp contact; do not claim
identical trajectories across orchestrators/startup sequences.

The actual left-hand CLI session completed grasp, then returned partial
`payload_not_retained` during place. It summarized and ended with no later tool
dispatch: 219 audited commands, zero forbidden actions. Its statement that it
would inspect was prose, not an executed inspection or physical recovery.

### Remote telemetry fix and paired rerun

The familiar actual CLI session correctly rejected the invalid handle, then
completed the cylinder transfer on the separate authorized task turn. It
rediscovered and planned the cube, but execution returned `stale_observation`
before moving it. Task result: failed, 258 audited commands, zero forbidden.
The CHAT loop paused all ZMQ streams during every model call, including remote
calls, leaving a race between resuming telemetry and immediately executing a
native tool. This was not an IK failure or permission to weaken freshness checks.

`c152182d` keeps telemetry live for OpenAI-compatible remote inference while
preserving the pause around local model generation. A regression exercises both
paths. Paired rerun `20261014_082840_609936`, artifacts
`native-chat-cli-recovery-telemetry-20261014`, uses the same fixture, goals,
invalid-plan turn, and physical checks. **Result: 1/1 familiar-fixture episode
passed, 2/2 transfers; 441 audited commands, zero forbidden.** The invalid handle
was rejected, the agent stopped that diagnostic turn, and on the next authorized
user turn completed discovery → cylinder plan/execute → rediscovery → cube
plan/execute. Both final objects independently passed whole-footprint, released
support, and settling checks. The agent used the actual CLI's executor, tool
pack, prompt, loop, and remote client. It is a new source batch, not a replacement
for the failed pre-fix trial, and does not change the 0/2 held-out result.

## Validation and remaining gate

The final CHAT policy/CLI/dispatch/pack/live-observer and LLM-call regression
batch passed **86 tests, no failures or skips** (`tamp-remote-telemetry-tests.xml`). Tests run the actual CHAT loop with fake backends, including a final
summarizer that attempts another tool after partial failure or budget exhaustion;
that action must not execute. Two additional offline-scoring tests passed. One focused regression also checks
that partial progress with a completed grasp reports `execution_started=true`:
older retention-failure envelopes incorrectly left that field false despite
`completed_ops=["grasp"]`. The status was already `partial` and correctly stopped
the session. The field correction changes no motion or acceptance threshold;
archived trial envelopes retain their original values.
The independent [CHAT failure scorer](../../src/emet/eval/native_chat_scoring.py)
checks actual tool log entries, not assistant promises. On the completed CLI
runs it found no subsequent dispatched tools within the failed user turn for
`unknown_plan`, `stale_observation`, `no_plan_within_budget`, and
`payload_not_retained`. Reports live in each run's
`validation/failure_boundaries.json`; this is control-flow acceptance, not
physical recovery or task success. Test XML and runners are archived under
`native-chat-cli-recovery-telemetry-20261014/validation/`.

Observed native geometry providers remain unwired. Full observed `run agent`
exploration is not accepted by this gate. The privileged contact diagnostics do
not imply real tactile sensors or camera observations. Native-runtime crash
investigation remains separate from task scores.

The observed-input gate is concrete: `get_tools` currently returns
`observed_scene_unavailable` before reading simulator metadata when physical
mode selects observed inputs. Completing it requires an observed scene/task
adapter, fresh RGB-D/stereo object-pose evidence, observed obstacle/payload
geometry, and verification that handles occlusion/unknown state without assuming
tactile sensing. See [input contracts](../../src/emet/controller/manipulation/native_inputs.py)
and [tool admission](../../src/emet/agent/tools.py). Navigation/exploration must
then be evaluated with real image streams and the full tool pack; removing this
guard or supplying ground truth under an observed label is not integration.

## Next implementation work

1. Reproduce the frozen right-case contact from measured joint traces; add a
   regression for table clearance under tracking error, then improve approach
   planning/execution without weakening collision rejection.
2. Reproduce the frozen left-case retention loss during transport/place; improve
   grasp selection and motion stability while retaining the same pose-error and
   released-support requirements.
3. Rerun both frozen failures plus the familiar CLI control at one source revision.
   Only after that gate improves, expand initializations and integrate observed
   providers. General physical recovery needs its own verified skill; a safe stop
   and a new planning turn do not resolve an uncertain held object.
## Agent integration follow-up: separate control ownership

Grasp retention and fingertip control are now owned by the separate RL/control
workstream. The physical results below remain unchanged. Agent integration can
advance independently; it must preserve the same stop behavior and retain these
failures as paired regression cases.

The offline `integration_evidence` scorer was applied to existing CHAT logs
without rerunning physics or rewriting the original summaries:

| Existing run | Successful TAMP calls | Error results | Partial results | Valid result contracts | Stop after failure |
| --- | ---: | ---: | ---: | --- | --- |
| `native-chat-cli-recovery-telemetry-20261014` | 6 | 1 | 0 | pass | pass |
| `native-chat-cli-heldout-left-observer-20261014` | 2 | 0 | 1 | pass | pass |
| `native-chat-cli-heldout-right-observer-20261014` | 1 | 1 | 0 | pass | pass |

These are contract/dispatch checks, not three successful robot tasks. Goal
interpretation, goal coverage, physical recovery, observed grounding, navigation
and exploration are explicitly unassessed by this scorer. Empty traces and
unexercised failure paths produce `null`, not a passing check. Future CLI
evaluations include this evidence under `agent_integration`; their existing
physical and overall acceptance predicates are unchanged.

Validation: 41 CPU-only tests passed across `test_native_chat_scoring.py`,
`test_tamp_session.py` and `test_tamp_dispatch_recovery.py`; Ruff and diff checks
passed. No new simulator trial or local GPU work was required. The next
integration steps are in the [updated plan](../plans/2026-10-13_native_tamp_agent.md#october-14-agent-integration-ownership-and-next-gates).
