# Main integration validation, October 22, 2026

## Source and scope

Merge `eb74c624` brings main `80818ac4` into the TAMP benchmark/runtime branch.
The common ancestor is `b2b71216`. The only textual merge conflict was SAM device
selection: preserve explicit device selection and main's shared-model behavior.
No motion tolerances, acceptance thresholds or task regions were changed.

Main adds acquisition-time propagation, measured head/navigation arrival,
fresh observation/map checks, bounded floor inspection/recovery, canonical
command-runtime synchronization and Stretch wheel-contact priority.

## Regression checks

Managed CPU jobs, `MUJOCO_GL=disable`, no local GPU:

| Job | Checks | Result |
| --- | --- | --- |
| `20261022_220202_7cbad3` | TAMP provider/runtime/CHAT, contact tracking, measured head/nav, map and floor recovery, SAM | 556 passed; 13 simulation checks initially skipped |
| `20261022_220342_e41693` | Command runtime/tracker, deployed parity, arrival, image timing, wheels | 42 passed; 5 wheel checks initially skipped |
| `20261022_220438_43ffb3` | All 18 skipped acquisition/frame/wheel checks, after linking existing RoboCasa assets | 18 passed, zero skipped |

Combined selected coverage: **616 passed, zero failures/errors; all initial skips
resolved**. These are contract/regression tests, not full observed-task acceptance.
Raw JUnit and job logs: `.tamp-runs/main-integration/`.

## Live agent controls

Two independent CPU-only live Stretch simulators use the actual `run agent` CLI
and Caliban `Qwen/Qwen3-VL-8B-Instruct` at the same remote endpoint as earlier
controls. No cameras, local VLM or observed grounding are enabled. The preserved
fixture retains its prior task geometry, initial placements and controller
parameters; only the wheel collision class/priority is updated to match main.
This is an integration/physics regression comparison, not a single-factor
state-update ablation or a household benchmark.

- Two-goal transfer: job `20261022_220310_70a4af`, port offset 1720.
- Wider-destination buffered swap: job `20261022_220311_863e3f`, port offset 1730.
- Output: `.tamp-runs/main-integration/{two-goal,wide-destination-swap}/`.

Both trials completed successfully:

| Case | Completed transfers | Final goals | Audit commands | Forbidden |
| --- | --- | --- | --- | --- |
| Two-goal transfer | 2/2 | Both footprints contained, supported and settled | 529 | 0 |
| Buffered swap | 3/3 | Both exchanged footprints contained, supported and settled | 750 | 0 |

Both CLI processes exited zero. Independent final scoring passed, and neither
trace contains a failure-boundary violation. The agent used discovery, beam task
planning and one-shot task execution through the production CLI. This preserves
the earlier successful control outcomes after integrating main; one run per
configuration is not a broad reliability estimate. The earlier narrow-destination
failure remains a distinct failed case and has not been rescored.

[Machine-readable results](tamp_main_integration_20261022.json) retain source,
job IDs, final measured predicates, test counts and raw protocol/summary hashes.
Remote checkpoint/sampling were not independently pinned.

## Remaining gate

Even passing stationary controls do not demonstrate navigation-to-manipulation,
observed multi-step grounding, exploration, hardware sensors or native open/close.
The next vertical slice must bind fresh observed task state after navigation and
invalidate any plan made before the world/robot state changed.

## PR review: failure contracts

Follow-up review identified uncaught discovery/planning provider exceptions and
loss of actionable execution codes. These are addressed through the shared
public failure translator, typed observation handling, and receipt validation.
Scene changes during planning return `stale_task_plan` with no stored handle;
private provider details remain redacted. The native transfer wrapper also
preserves the adapter's public code. Tests cover JSON tool boundaries, unknown
exceptions, precommand rejection and partial progress followed by failure.

The low-priority `TaskBudget` finding is confirmed: it has no production caller.
Its module and the task API now label it as unwired infrastructure. Whole-agent
budget wiring remains an explicit acceptance gate, not a claim of this PR.

Validation: managed job `20261022_224214_a0ce72` passed 636 broad regression
checks with zero failures/errors/skips. After the final added partial-progress
case, the focused provider suite passed 38 checks (overlapping the broad suite;
do not add these counts). Ruff and `git diff --check` are clean. These changes
alter failure reporting, not trajectories; the live controls above predate this
review follow-up and are not a new post-follow-up live result.

## Independent review follow-up, October 23

Native runtime stop reasons now distinguish upstream truncation
(`upstream_truncated`) from the local configured horizon (`horizon_exhausted`);
upstream termination takes precedence when both flags are set. Original info and
step flags remain intact, and every terminal case still rejects further actions.

The offline CHAT scorer now audits discovery-only stop boundaries as well as
failed/partial outcomes. Boundary entries include `boundary_kind`; the existing
compliance field covers both kinds, while `failure_cases` counts actual failures
and `discovery_only_cases` counts policy-only stops. `stop_boundary_cases` is their
total. Discovery remains a successful read-only tool result. Explicit user turns
reset either boundary; assistant prose and dispatcher bookkeeping do not.
Historical experiment aggregates have not been rescored.

Validation: 81 targeted runtime, scorer, dispatch and live-evaluator checks passed;
Ruff and whitespace checks pass. No motion or controller behavior changed.

## API contract and precommand budget review correction, October 23

The earlier 636-check result was a selected regression set, not the full TAMP
suite: it omitted `test_api_contract.py`. Reproducing the review command on that
head fails with `KeyError: plan_task`. The fixture now supplies arguments for
both new task tools and asserts that its keys exactly match `TAMP_TOOLS`, so
future tool additions cannot silently escape this contract check.

Native execution refinement now catches planning-only `BudgetExhausted` before
calling the adapter executor and returns an explicit no-actuation receipt. A
successful feasibility probe can consume the final IK allowance; the subsequent
execution refinement must then return an error, not claim physical progress.
The regression reproduces that path and verifies zero adapter execution calls.
A second case checks that an earlier completed transfer remains partial progress.
Unknown provider exceptions remain conservative; this fix does not infer that
arbitrary exceptions imply no actuation.

Validation job `20261023_124947_5a6a9b`: **760 passed, zero failures/errors/skips**
in 93.844 seconds. This includes all 161 tests in
`src/test/controller/task/tamp/` plus the listed integrated agent, native motion,
observation, navigation, runtime and scoring regression set. It is not a run of
the entire repository suite. The focused API/provider run passed 89 overlapping
checks. Ruff and whitespace checks pass. JUnit: `.tamp-runs/review-precommand/all-tests.xml`.

## Second preflight/protected-goal review follow-up, October 23

The previous budget fix covered native refinement only. Review correctly found
that region-provider snapshot/feasibility can also fail after the session enters
its execution phase but before transfer. The provider now contains an explicit
read-only preflight exception boundary and returns a no-actuation receipt from
that boundary. Transfer itself remains outside the catch, so unknown physical
outcomes remain partial. Regressions inject both support-state rejection and
budget exhaustion inside provider snapshot and feasibility, and separately
verify conservative handling of an unknown transfer exception.

Goal compilation now protects unmentioned objects' current locations. An explicit
move into another object's occupied region cannot silently relocate that object;
a multi-object goal or swap authorizes those moves. Native task configuration is
rejected with EQA evaluation before JSON loading or runtime startup. Both remote
OpenAI client classes retain live telemetry; the local-inference pause guard is
unchanged and tested.

Validation job `20261023_131049_682a56`: **775 passed, zero failures/errors/skips**,
93.856 seconds. This covers the entire TAMP test directory and the integrated
regression set, extended with text/vision inference guard tests. It is not the
entire repository suite. Ruff and whitespace checks pass. No live-task aggregate
is rescored by these contract fixes.

## Native planning boundary and phase attribution, October 23

Re-review found the native execution-refinement catch still covered only budget
exhaustion. The entire planning-only refinement and handle-validation region is
now contained before `adapter.execute`: planning timeouts, malformed responses,
missing handles and invalid handles return typed no-actuation failures. The actual
adapter executor is deliberately outside this catch; an executor timeout remains
an uncertain physical outcome.

The task session labels a failed typed no-actuation receipt as `preflight` for
the failed step. Earlier physical progress remains partial without relabeling
that earlier work. Tests assert the phase as well as status, completed steps,
execution-started flag, private-data redaction and absence of executor calls.
The focused provider suite passes 55 checks.

Validation job `20261023_154613_e3af8e`: **784 passed, zero failures/errors/skips**,
93.868 seconds, covering the entire TAMP test directory plus the same integrated
regression set. Ruff and whitespace checks pass. This is not the entire repository
suite or a new live task-success trial.
