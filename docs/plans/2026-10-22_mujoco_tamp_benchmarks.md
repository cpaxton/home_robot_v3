# Shared-agent MuJoCo TAMP benchmark implementation

Approved direction: six MuJoCo suites, one production agent and tool executor,
matched planning conditions, separate upstream and adapted robots. This extends
[the household design](2026-10-18_household_agent_tamp.md); it does not replace its
fixed-population, reference-certification, or observed-input gates.

Machine-readable program: `configs/eval/mujoco_tamp_program.json`. It remains
non-executable until real certified episodes and provenance are available.

## Implementation status

The first implementation branch provides immutable episode/model/reset/profile
contracts, native control-step accounting, full-environment adapter interfaces,
JSON registry validation, fixed-population result accounting, and single-use
multi-step plans in the production CHAT tool registry. Greedy, beam and the
existing UCT/MCTS share operator/precondition/feasibility interfaces. Each step
must be verified against a fresh measured snapshot. Unsupported algorithms are
rejected rather than relabeled as another search method.

This is the runtime foundation, **not six qualified benchmark integrations**.
The new `plan_task` / `execute_task_plan` tools require an explicitly injected
`TaskSession`; the normal agent does not automatically acquire a household task
provider. An opt-in stationary Stretch diagnostic provider now binds measured
region occupancy through `--native-task-config`, probes native first moves and
refines motion per step. This is not upstream suite qualification.
Grounded operators, instruction compilation, native controllers and
observed providers still need qualification per suite. The shared budget ledger
exists, but whole-agent/inner-ranking/IK reservations are not all wired yet;
there are no matched-budget benchmark claims.

### Remaining integration gates

1. Bind complete upstream environments and certified resets to robot command
   profiles and fresh grounded task providers. Preserve upstream controllers,
   task success checks, reset states and horizons. Exported XML alone does not
   qualify a suite. Count every native control action inside skills.
2. Qualify transfer, temporary storage and stacking/containment separately.
   Articulation, press/turn and navigation remain unsupported until native
   actuator execution and measured verification pass. No attachments or object
   pose writes during execution. Private MuJoCo data is for planning only.
3. Reuse production CHAT orchestration for all LLM conditions. `react` selects
   one operation; `llm_plan` proposes a complete sequence; `llm_affordance` ranks
   candidates with binary geometric feasibility. Label these adaptations
   accurately: affordance ranking is SayCan-inspired, not learned skill success.
   Bind actual PDDLStream symbolic/optimistic streams to our MuJoCo checks;
   do not import its PyBullet execution examples or emulate it with beam.
4. Wire a single budget ledger through CHAT, nested ranking and IK. Defaults:
   32 tool rounds, 64 total LLM calls, depth 16, 500 expansions, beam width 8,
   500 MCTS iterations, 2048 episode IK calls, 120 s planning, 1800 s wall time.
   Preserve existing per-primitive bounds. Upstream native horizon is authoritative.
   Keep live asynchronous episodes in a separately reported condition.
5. Qualify two tasks × two resets × six suites (24 cases), with genuine success
   and controlled failure through the production tools. First frozen subset:
   six task types × two resets × six suites = 72 configurations, three repeats.
   Apply all planners only to multi-operation cases; component tasks use the
   production strategy. Do not freeze unsupported placeholder episodes.
6. Preserve the separate adapted household protocol: four causal families ×
   three held-out layouts × two seeds × two simulators = 48 configurations,
   three repeats = 144 episodes per condition. Require two actuated reference
   replays before certification; reference order is unavailable to the agent.
7. Run observed-input RoboCasa/MolmoSpaces/LIBERO conditions through the same
   applicable planners. Declare camera/depth inputs, freshness and tracking;
   privileged pose/contact inputs cannot masquerade as hardware observations.
8. Publish raw JSON/provenance, fixed-population and certified-subset metrics,
   configuration-grouped uncertainty, task-length/failure plots and annotated
   videos. Run two CPU simulations initially, Caliban-hosted VLM, no local VLM.
   Paper household claims wait for observed end-to-end evidence.

## Suite boundaries and environments

| Suite | Runtime family | Required integration beyond the bridge |
| --- | --- | --- |
| robosuite | Native full robosuite env | Fixed reset installer, robot/controller/profile, grounded skills |
| RoboCasa365 | Full kitchen task env | Task metadata, original success, reset/layout certificates |
| MolmoSpaces | Upstream episode/task runtime | Dedicated adapter preserving fixed JSON episode and evaluator |
| LIBERO | Full BDDL robosuite env | Official init-state arrays, BDDL goals and task success |
| Franka Kitchen | Gymnasium task env | Requested task set, completion semantics, native control |
| Meta-World | Gymnasium task env | Task instance/reset binding, original success info and horizon |

Use isolated, pinned environments when versions conflict. This checkout reports
MuJoCo 3.5.0, robosuite 1.5.2 and RoboCasa 1.0.0 distributions. LIBERO,
gymnasium-robotics, Meta-World and PDDLStream are not installed as distributions;
MolmoSpaces also has no installed distribution record. Distribution presence
alone does not establish asset availability or compatibility. Never silently
install a different upstream version into the shared simulator environment.

## Validation requirements

Test reset identity, horizon accounting inside skills, planning purity, stale
plans/restarts, moving obstacles, blocked pregrasp, slip, protected goal
regression, temporary-buffer dependencies, malformed arguments, timeouts and
cleanup. Strategies must solve known causal domains with the same operators and
respect shared budgets. Audit native commands for forbidden assistance.

Current CPU tests exercise contracts, causal swaps, measured-effect rejection,
consumed/stale handles, budget reservations, fixed-denominator accounting and
production CHAT registry/dispatch. The renderer-free robosuite Lift smoke is a
native runtime/control/horizon check, not agent success or suite qualification.

Validation evidence: [October 22 runtime checks](../experiments/tamp_benchmark_runtime_20261022.md),
258 tests passed with no skips; no household task-success claim.

## Review organization

Keep #213's existing native table evidence separate. First follow-up:
`feat/tamp-benchmark-runtime` (runtime boundary and task-tool contracts).
Second follow-up: suite-qualified providers, matched planning bindings and runner.
Do not remove draft status or claim readiness for the second gate from interface
unit tests alone. Existing 2/6 table task success and 6/6 attempted transfers stay
historical diagnostics, not household results.
