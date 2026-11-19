# Reacquire source evidence after native search — November 17

Starting checkpoint: `0934bad3` on `fix/tamp-observed-admission`.
This is an assisted synchronous MuJoCo/Stretch development control with native
robot/object state, native full-scene geometry and a provided external RGBD
scanner. It is not an observed-only, LLM, household benchmark or hardware result.
Physical Stretch is excluded. Physics advances during controller sleeps, not
during expensive planner wall time. No runtime poses or capture times are rewritten.

## Problem and implementation

The [prior declared matrix](mobile_release_20261117.md) remains five of seven
task passes and two of two expected negative controls. Its slow-CPU shifted cube
and long-search clutter case expire source evidence. A separate faster-CPU
shifted-cube success does not replace that primary failure.

The opt-in handoff separates an inert search proposal from an executable handle.
It retires old leases and consumes old handles, acquires a new post-barrier scan,
then recertifies selected nominal edges/FK endpoints and the loaded base sweep
on private data, without IK search. New obstacles and changed original depth rays
can reject even when candidate endpoints or coarse grids match. Model, policy,
intent, source identity and strict full-scene stationary gates remain enforced.
Runtime measured refinement, checked closing, retention and release bounds are
unchanged. This source provider is trusted to query its declared geometry;
arbitrary callbacks are not evidence certificates.

The final implementation also binds the immutable collision callback tuple to
its lease and reports the actual possible-command boundary. Source loss during
planning before any send is no execution. A possible send with lost acknowledgement
remains partial/unknown. Old handles cannot be retried.

## Retained pilots and checks

Artifacts: `.tamp-runs/source-finalization-20261117/` in the owning worktree.
Dense traces stay there; the [companion JSON](source_finalization_20261117.json)
records compact hashes and results.

| Managed job | Scope | Result |
|---|---|---|
| `20261117_211014_3f8e55` | Initial pure boundary/store/default native tests | 77 passed |
| `20261117_211605_a0c918` | First public-boundary test collection | Import error in test helper; no product trial |
| `20261117_211753_7efdbe` | Public handle finalization, changed obstacle and acquisition timeout | 3 passed after helper import repair |
| `20261117_213507_575890` | Final binding, stale-before-send, transport timeout, source/store controls | 25 passed |
| `20261117_214234_de0f7d` | Additional cleared-admission and mutable-guard lease controls | 27 passed |
| `20261117_214033_c06ebf` | First broad native/API/contact/release regression | 323 passed |
| `20261117_214444_645b74` | Association guards, including legacy-fallback protection | 326 passed, zero failures/errors/skips |
| `20261117_215810_42778c` | Adapter-owned close/hold/base transport guards | 329 passed, zero failures/errors/skips |
| `20261117_220136_edc070` | Final source/feedback/transport and adapter-cache boundaries | 32 passed, zero failures/errors/skips |
| `20261117_211304_552d5a` | Initial source-refresh mobile cylinder | Full task passed; predates final binding/dispatch hardening |
| `20261117_211303_fc3bce` | Initial source-refresh clutter, 180-second validation budget | Safe rejection before manipulation |

The clutter pilot spends 469.96 seconds in `plan_transfer`, including a real
24.13-second fresh source scan and the 180-second fixed-path validation budget.
It returns `no_plan_within_budget`, `execution_started=false`, no completed
operations and no arm/gripper sends. Its three base entries belong to the earlier
separately checked unloaded approach/stop and final cleanup stop. It is not a
task success. The dedicated validation budget is now 240 seconds; the original
299-second source lifetime, collision tolerances and IK/search budgets remain
unchanged. Final runs are separate from this retained pilot.

The initial refreshed cylinder completes navigation, pick/carry/place, retreat
and departure. `plan_transfer` takes 338.06 seconds including the new scan and
recertification; map lifecycle records five loads/unloads and zero resident bytes
after cleanup. This is a single development success with assistance, not an
agent success rate. The later trials below are recorded separately.

## Follow-up simulator results

| Managed job | Case / safe CPUs | Full task | `plan_transfer` / physical transfer |
|---|---|---|---|
| `20261117_214001_29a0a4` | Cylinder / 4–7 | Passed through departure | 344.61 s / 299.60 s |
| `20261117_214002_ee4792` | Cluttered cube / 0–3 | Partial: stopped at release | 478.15 s / 277.98 s |
| `20261117_214530_350d17` | Shifted cube / 12–13 | Passed through departure | 375.29 s / 332.68 s |

These are **two full task passes among three selected development diagnostics**,
not a replacement seven-case matrix or benchmark success rate. All three
complete fresh-source revalidation of eight nominal segments and a loaded base
sweep with zero additional IK calls. No trial stops from expired source evidence.
Successful cylinder/shifted-cube runs unload all five maps and finish with zero
counted resident bytes; map peaks remain about 24.1 MB. Full process peaks remain
about 2 GB, independent of that retained-map count.

The clutter case now physically grasps, lifts, carries and reaches the placement
pose. Its measured release search accepts one bounded height correction, then
rejects `gripper_open_path_failed:invalid_goal:collision`. The final original-map
witness reports `unknown_swept_volume` for a **non-payload** moving-robot volume
around x=0.197–0.234 m, y=-0.678–-0.634 m, z=0.521–0.565 m near the distractor.
Earlier candidate queries also report occupied environment. Native contact lists
on those opening rejections are empty: this does not establish a real physical
penetration or justify treating the unknown volume as free.

The public receipt correctly says `status=partial`, `code=collision`,
`completed_ops=["grasp"]`, `failed_stage=release`, `execution_started=true`,
`recovery=inspect`. The object is not scored placed, no departure is attempted,
and cleanup unloads all four maps. Its 90 command entries include two gripper
presets, a grip-hold target and one release-adjustment arm send; no failed release
opening is sent.
This exposes a **remaining placement-selection gate**, rather than a source
lifetime regression: select/retry a destination with evidence for the full hand
opening, within the original goal region and planning budgets. Do not relax
unknown-space rejection or assert that a tabletop is impossible from one failed
release candidate. Task rearrangement remains an option after bounded placement
alternatives are exhausted.

The cylinder/clutter reruns started before the last review fixes for missing
admission metadata and mutable guard lease ownership. Their precise runtime
hashes are retained; matching source bytes are SHA-verified and archived under
`runtime-snapshots/`. These guards narrow authorization and do not change motion
on their valid paths. The shifted-cube trial includes these association guards.
All three simulator trials predate the last adapter-owned transport proxy/cache
hardening, which is covered by subsequent regression and boundary tests. Proxy
resolution preserves the original adapter and namespace during tool-registry
cache lookup inside execution. No simulation is silently relabeled as having
run later code. Independent read-only review identified these issues before
commit and confirmed the final cache/transport corrections with no remaining
concrete blocker. The updated paper builds to 71 pages
(`20261117_220430_b50831`; initial build `20261117_214843_5e9579`);
pre-existing font/float warnings remain.

Each task uses `run_mobile_map_regression.py --refresh-source --case ID`, the
unchanged declared fixture/suite and a 1,200-second per-case watchdog. The OSMesa
renderer is CPU-only. Per-case `runtime_sha256` and fixture hashes are captured
before execution; failures/timeouts remain in the lane manifests. No local GPU
or VLM is used. Original development cohorts are not pooled with these reruns.

## Remaining qualification

Full observed camera/motor/refiner ownership, asynchronous scene changes,
hardware stopping/localization bounds, onboard-only sensing, household scenes
and agent exploration remain separate gates. This strict stationary finalization
does not repair all slow planning, nor establish a continuous collision certificate.
