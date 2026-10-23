# Native task-provider integration and crash recovery — October 22, 2026

This extends [the runtime foundation](tamp_benchmark_runtime_20261022.md).
Production CHAT now discovers measured region occupancy, plans an operation
skeleton, probes native first moves, executes ordinary pick/place commands and
checks measured effects after each transfer. No scripted expert sequence selects
ordering. Inputs are privileged, the Stretch base is stationary, and native
execution is live asynchronous. These are diagnostics, not upstream benchmark
scores or observed-input acceptance.

## Recovery and provenance

The compute crash cleared `/tmp`, including the temporary worktree and the first
trial artifacts. The pushed foundation `f33be8d5` and managed-job registry
survived. Uncommitted provider work was reconstructed and checked into
`feat/tamp-benchmark-runtime`. The active checkout and all new raw artifacts now
live under `/home/cpaxton/src/home_robot_v2/.worktrees/tamp-benchmark-runtime`.
The original shared checkout's unrelated edits were left untouched.

Do not cite the pre-crash success as reproducible scored evidence: its raw data
was lost. The results below come only from new persistent artifacts.
Machine-readable data with source commits and raw-summary hashes:
[agent_task_provider_20261022.json](../../paper/data/native_tamp/agent_task_provider_20261022.json).

## Recovered production-agent diagnostics

All use remote Caliban `Qwen/Qwen3-VL-8B-Instruct` and CPU physics with
`MUJOCO_GL=disable`, no renderer/cameras/local VLM. Checkpoint/sampling settings
are not independently pinned. Two simulator episodes ran concurrently for the
expanded swap and two-goal cases. XML copies replace obsolete temporary asset
paths; the physical fixture is the familiar table variant.

| Case | Result | Completed transfers | Audit commands | Forbidden commands |
| --- | --- | --- | --- | --- |
| Narrow buffers | Search rejected all native first moves; no execution call | 0 | 7 startup commands | 0 |
| Expanded forward buffer | Partial: cylinder moved to buffer; next cube refinement rejected | 1 of planned 3 | 232 | 0 |
| Wider-destination swap | Both exchanged goals independently supported/settled/contained | 3 of planned 3 | 700 | 0 |
| Two-goal causal transfer | Both final goals independently supported/settled/contained | 2 of planned 2 | 477 | 0 |

These are different task configurations, not a common statistical denominator.
The two-goal case moves the cylinder out of the cube's destination first.
The wider-destination swap completes all three transfers. It changes the allowed
destination geometry and remains separate from the failed narrow case.
The swap needs temporary storage. Expanding the forward buffer made a native
cylinder-first move feasible; it did not establish a complete physical reference
for the swap. The code must not label the latter certified solvable from this
partial run, nor infer impossibility from bounded search failure.

Raw artifact directories under the active checkout:

- `.tamp-runs/live-recovery`: job `20261022_164003_c2b5b9`, source `579dc9eb`.
- `.tamp-runs/large-buffer`: job `20261022_164318_21c555`, source `f00bdb0b`.
- `.tamp-runs/two-goal`: job `20261022_164318_faf73c`, source `f00bdb0b`.
- `.tamp-runs/wide-destination-swap`: job `20261022_170253_1d62e3`, source `0adaec26`.

Each contains configuration, source hashes, full CHAT logs, independent summary,
command audit, server/agent logs and managed-job metadata. Failure-stop checks
pass for both failures; a successful-only episode leaves failure-stop unassessed.
Startup commands are not task execution evidence.

## Failure diagnosis and changes

The narrow-buffer run rejected four candidate first moves. Cube proposals
included grasp collisions, pregrasp IK failures and insufficient placement
space; cylinder proposals included preplace IK failure. These are counted
rejections, not a proof that the task is impossible.

In the expanded-buffer run, the first transfer physically succeeded. The next
cube move was rejected before its motion, using 73 planning IK calls with 54
pregrasp IK, 4 grasp-collision, 3 surface-space and 2 pregrasp-collision
rejections. No retention loss caused this stop. Rejection counts alone initially left IK initialization versus geometric access
ambiguous. The recorded-state controls below isolate destination geometry;
additional IK seeds do not resolve the original case.

The agent's prose incorrectly described the first step as failed even though
`completed_steps` was 1. Structured outcomes now include `failed_step_1based`
and `failed_phase`, and the forced final-summary prompt explicitly preserves
completed progress. The stop policy still prohibits automatic retries.
Opt-in `--debug-tools` for this privileged native provider adds recorded private
state and IK rejection details to JSON debug logs; no state is added to public
tool results or the public motion-evidence list. A diagnostic replay is managed
as job `20261022_165209_0e1a2f`, source `0adaec26`, with persistent artifacts in
`.tamp-runs/swap-diagnostics`.

## Verification

After reconstruction, job `20261022_163803_724d6d`: 275 passed, no failures/skips.
After CLI and budget updates, job `20261022_164558_e27ff1`: **280 passed, no
failures/skips**, 82.68 s. The subsequent failed-step/JSON diagnostics change
passes 97 focused provider/runtime/CHAT tests. Ruff and the paper build pass.
Tests exercise causal buffer dependencies, source gates, settled native
occupancy, private planning, first-move pruning, shared probe allowance including
failed calls, consumed/stale plans, explicit partial progress and CLI refusal.

![Measured initial/final footprints, not trajectories](../../paper/figures/native_task_integration_20261022.png)

## Remaining gate

Preserve dense collision checks, measured convergence and contact verification.
Do not promote extra yaw/IK restarts from unsuccessful offline controls. The
wider swap now has one successful physical episode; require repeated complete
reference replays and held-out resets before freezing benchmark instances. Qualify complete references before a frozen task registry;
then expand suite providers and observed-input execution. The new graph/skeleton
integration does not complete the six-suite or seven-planner program.

## Recorded-state controls

The debug replay captured the same post-buffer failure. The agent now correctly
reports one completed step and a step-2 preflight failure. The resulting record
is preserved in `src/test/fixtures/tamp/recorded_post_buffer_20261022.json` with
source/model binding. `RecordedNativeTablePlanner` restores it only to private
data, labels results offline, rejects model mismatches/nonfinite state and
unconditionally blocks execution. No synthetic live observation timestamps are
created.

Controls on this exact state, preserving the 256-call cap and every pose,
collision and contact tolerance:

- Existing search: no accepted plan, 73 IK calls.
- Additional nearby yaw samples: no accepted plan before 256 calls.
- Home/midrange IK restarts: no accepted plan, 181 calls.
- Original destination `[[−.08,−.76],[.08,−.64]]`: no accepted plan.
- Wider/shifted destination `[[−.08,−.75],[.08,−.62]]`: accepted collision-checked
  transfer in 14 calls, with the same robot/object state and solver.

Thus aggregate pregrasp IK counts alone were misleading. Several grasp/lift
branches were accepted but lacked an accepted support placement in the original
region. The region control provides a planning witness; it does not establish
that the narrow physical task is impossible, or certify physical execution of
the wider task. No IK restart/yaw changes were promoted to production from these
negative controls. The original failures remain failures.

Jobs: yaw control `20261022_165718_006d65`, seed control
`20261022_165907_31948b`, region control `20261022_170123_06c887`.
A deterministic native regression verifies the original rejection, the wider
accepted path, unchanged private state and blocked actuation (2 tests passed,
job `20261022_170415_5d64e8`). Complete JSON control results accompany the
paper data. A new wider-destination physical swap is separately labeled in
`.tamp-runs/wide-destination-swap`, job `20261022_170253_1d62e3`.


## Wider-region physical outcome and final validation

The separately labeled wider-destination trial completes all three transfers:
cylinder to buffer, cube to the cylinder's original region, cylinder to the
cube's original region. The actual production agent discovers, requests a beam
plan and executes its opaque handle; it does not supply an expert ordering.
Independent final checks find both footprints contained, supported and settled.
The audit contains 700 ordinary commands and zero forbidden commands. The
original narrower task remains partial in the dataset. This single successful
variant does not certify general swap solvability or either benchmark population.

Final unified validation job `20261022_170543_7b0f3a`: **284 passed, 0 failed,
0 errors, 0 skipped**, 84.04 s. This includes the new recorded-state regressions,
failed-step contract, debug JSON/private-state boundary and CLI tests. Ruff is
clean. The paper figure and data are reproducible using
`scripts/plot_native_task_integration.py`; the paper build succeeds.
