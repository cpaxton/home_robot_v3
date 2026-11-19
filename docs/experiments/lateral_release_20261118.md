# Bounded lateral release alternatives — November 18

Starting checkpoint: `0cea6246`, branch `fix/tamp-observed-admission`.
The prior [source-finalization follow-up](source_finalization_20261117.md)
completes cylinder and shifted-cube tasks, while clutter completes grasp/carry
and rejects release on unknown opening volume near a distractor. This record
preserves that result. The original seven-case 5/7 development cohort is unchanged.

This is synchronous CPU MuJoCo with simulated Stretch, native robot/object state,
native full-scene geometry and a provided external RGBD scanner. Physics advances
during controller sleeps. It does not qualify observed-only perception, hardware,
household OVMM or LLM-agent success. No new model or GPU allocation is needed.

## Change and controls

An existing uncommitted draft supplied sideways release search and three geometry
tests. The completed implementation keeps vertical-first selection, adding a
bounded horizontal fallback only when vertical candidates cannot certify opening.
Default lateral targets are 1/2/3 cm along four directions at the existing height
increments. The original measured XY anchor bounds the entire sequence. Full
arm/payload paths, commanded/lowered opening, released fall and retreat must pass;
the actual executed retreat follows the selected lateral shift. Original region,
raise/drop, contact and retention limits are preserved, with at most two total
physical corrections.

Release IK now consumes the retained primitive's call budget. Review identified
that timing only IK would omit dense geometry work; complete planning passes now
charge elapsed time and check their deadline before motion authorization. Loaded
refinement also carries its consumed calls/time forward. A public-tool boundary
test preserves completed grasp and reports partial release-budget failure without
an opening or another arm send. Tight-region and cumulative-radius tests reject
escaping alternatives; the positive region test checks the full payload footprint.

## Managed runs

Artifacts live under `.tamp-runs/lateral-release-20261118/` in the owning worktree.
Runtime patches are retained before launches; each task lane records source and
fixture SHA256 values. These local paths are not a public archive.
The [compact JSON record](lateral_release_20261118.json) keeps versions, receipts,
hashes and resource measurements separate from the dense traces.

| Job | Scope | Result |
|---|---|---|
| `20261118_000136_3985c8` | Initial release/protocol checks | 69 passed |
| `20261118_000416_544617` | Initial broad regression | 348 passed, two old counter assertions failed |
| `20261118_000739_289f4a` | Region/radius/retreat, full-pass timing and partial-feedback checks | 77 passed |
| `20261118_001020_dfec3d` | Final native/contact/API/release regression | 353 passed, zero failures/errors/skips |
| `20261118_002245_e25874` | Multi-step provider and retained-release budgets | 74 passed |
| `20261118_002644_5be4c8` | Lateral fall-volume clear/unknown controls | 2 passed, no commands |
| `20261118_000258_e1858e` | Clutter, CPUs 0–3 | Failed safely during initial search; no manipulation |
| `20261118_000259_8b4baa` | Cylinder control, CPUs 4–7 | Full task passed |
| `20261118_001151_540dd1` | Separate clutter repeat, CPUs 12–13 | Full task passed |
| `20261118_001322_79b9f8` | Saved-state geometry inspection with a newly rendered provided scan | Found a checked 1 cm lateral target; no commands/task-success score |

The two initial unit failures assumed `plan.planning_ik_calls` would remain at
its pre-refinement value. The corrected test saves that starting value and
requires both adapter and retained plan to include all consumed calls. Native
state, payload reference and no-command assertions remain intact.

The first clutter task spends 274.12 s in initial planning after a 24.92 s source
map build and returns `support_geometry_unknown` before source finalization.
It does not exercise lateral release. Capture time/lifetime is not renewed to
rescue that run. The source lifetime stays 299 s; a separately retained CPU repeat
tests the reviewed code without replacing this failure.

The first cylinder/clutter launches predate the review's complete-pass timing
correction. The clutter repeat includes that correction. Exact source patches
and per-case hashes distinguish these versions; no earlier run is relabeled as
having used later code.

## Measured task result

The complete clutter repeat uses one vertical correction, followed by a 1 cm
negative-X destination correction. It then certifies the fresh measured pose
with adjustment disabled after reaching the unchanged two-correction limit.
Opening verifies supported/settled release; the public tool additionally checks
two seconds of settled support and the original region. The shifted retreat,
released-object remapping and departure all complete. There is no task-level
obstacle relocation and no increased contact/drop/retention allowance.

`plan_transfer` takes 484.24 s and `pick_carry_place` 469.63 s on CPUs 12–13;
these include nested map/sweep work and are not real-time performance results.
Five maps are loaded/unloaded, peak counted storage is 24,114,174 bytes, and
cleanup leaves zero resident bytes. Peak full-process RSS is 2,019,155,968 bytes.
The released object occupies 84 cells in the next scan and drifts 0 m during
departure. The cylinder control also completes, retaining vertical-only release;
its measured departure drift is approximately 0.65 micrometres.

Thus two full task executions complete among three launches in this follow-up:
one initial-search expiry, one successful cylinder and one successful clutter
repeat. The repeat is not substituted for the expiry or the historical failed
release. These selected diagnostics do not constitute a new seven-case success
rate, held-out evaluation or an agent score.

The saved-state inspection reconstructs the prior failed physical state as
initial geometry and renders a genuinely new provided scan. It finds five
feasible lateral proposals among the default schedule and selects negative X
by 1 cm using support slack, after 25 planning IK calls. No command is sent.
This isolates geometric feasibility; the separate complete trial establishes
measured release and continued task execution.

Independent read-only review confirmed geometry, retreat propagation and retained
budget accounting after the full-pass timing correction. There are no remaining
concrete review blockers for this change; this does not qualify broader sensor
or hardware assumptions.

The updated paper appendix builds successfully through managed job
`20261118_003214_208be3`; pre-existing font/float warnings remain.

## Remaining gates

Initial search can still outlast source evidence; its safe rejection is retained.
Expand clutter/layout tests and reduce planning cost before claiming general
robustness. Saved-state planning alone cannot establish physical task success.
Real observed source assembly, asynchronous control and broader task qualification
remain separate. Unknown-space rejection and all failed/expired attempts remain.
