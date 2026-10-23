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
