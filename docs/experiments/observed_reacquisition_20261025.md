# Fresh RGBD surface reacquisition — October 25, 2026

The local reassociation path now refreshes **visible surface positions** without
repeating semantic search. It does not estimate full object pose, unseen payload
volume, grasp retention or contact, and does not enable observed manipulation.
The actual full `run agent` control and the observed sensor diagnostics are
separately scored below. All attempts, including failures and offline replays,
are retained in [the result record](observed_reacquisition_20261025.json).

These are development diagnostics on evolving code based on `6c0d1a20`, not a
frozen paper benchmark. Raw RGBD arrays, masks, per-frame associations and logs
remain under `.tamp-runs/reacquisition-20261025/`. New harness runs record source
and code hashes; the development attempts here preceded that provenance addition.

## What ran

| Condition | Outcome | Remote initialization calls | Local updates |
| --- | --- | ---: | --- |
| Airborne reset reference, live physics during inference | rejected after object fell about 8 cm | 2 | 0/10 accepted |
| Same reset condition with detailed diagnostics | rejected | 2 | 0/3 accepted |
| Settled home pose, old color/depth components | VLM selected mixed object/table surface; reassociation rejected | 2 | 0/10 accepted |
| Settled home pose, normals split, cylinder | passed | 2 | 10/10 accepted |
| Settled home pose, normals split, cube obscured by arm | candidate overflow; rejected | 2 | 0/10 accepted |
| Settled stowed arm, normals split, cube | passed | 2 | 10/10 accepted |

An earlier harness wrapper error made zero remote calls; it is retained as a
harness failure. An additional one-call **offline** support-only presentation
comparison still selected the mixed region. A CPU-only scaled-context replay did
not resolve cube rejection and was not adopted. These conditions are not pooled
into a task success rate or silently replaced by the later successful conditions.

For the successful cylinder run, both initial HTTP calls together took 13.53 s;
local reacquisition stages had median 9.26 ms. For the unobstructed cube they took
15.28 s initially and median 8.60 ms per update. These update timers include the
diagnostic selected-mask PNG write. Current-frame evidence ages, measured from
capture receive time rather than inference completion, stayed below 80 ms in
both successful runs. No further VLM calls occurred for the ten updates. Each
run independently rejected synthetic missing depth.

HTTP timings include transport, server queueing, image prefill, decoding and any
SDK retry. They do not isolate server compute or time to first token. Local JPEG
encoding took milliseconds, while individual remote requests took seconds.

## Root causes and fixes

1. The fixture reset puts objects above the table. Taking a semantic reference
   immediately after reset makes it obsolete while physics settles. The harness
   now advances physics before its reference capture and keeps physics advancing
   during inference. The association distance remains 2.5 cm; an 8 cm fall is
   correctly rejected.
2. Local RGB/depth adjacency can transitively join an object to touching support
   furniture. Optional calibrated world-point normals split measured components
   at 30-degree normal changes. In the saved settled view this separated the
   tabletop from the cylinder side: the new selected reference contained 176
   red-dominant pixels out of 176, compared with a previous mixed region of 2,540
   pixels. Color counts are fixture-specific diagnostic scoring, not runtime
   identity logic. The same split is available to the existing grounding API.
3. Stretch's home arm obstructs the cube in this camera view. The failed attempt
   exceeded the unchanged eight-candidate budget. Physically commanding the
   existing stow control targets before observing gave a clear view; the cube
   reference then contained 242 blue-dominant pixels out of 242. No arm qpos
   teleport, larger candidate budget, target-specific color rule, or hidden retry
   was used to obtain that success. Acquisition posture is a real precondition.

![Actual camera images and measured selected surfaces before and after the geometry fix](figures/observed_surface_reacquisition_20261025.png)

## Implementation and integration boundary

- `surface_candidates(..., world_xyz=...)` optionally uses aligned measured
  normals in addition to RGB/depth continuity. Invalid/degenerate normals cannot
  connect pixels. Coplanar touching objects can still remain unresolved.
- Existing `ground_vlm_region`/`select_supported_region` accept this geometry.
  `mapping.query_memory.split_surface_normals: true` forwards current world depth
  through `LazyGraphController`; defaults remain unchanged. It requires aligned
  world geometry and `region_strategy: depth_candidates`.
- `reacquire_surface` associates whole current measured components to the fixed
  reference using appearance, bidirectional point support and bounded geometry.
  It rejects ambiguity, clipped regions, large motion, size changes and missing
  support. It never grows a drifting reference or chooses the nearest survivor.
- `RgbdTaskProvider.update_reacquired` publishes only read-only discovery
  positions. The current capture must be identified, fresh within 2 s, and in
  the reference motion epoch; the semantic reference expires after 30 s. A failed
  surface clears the complete proposal snapshot. Replayed captures cannot renew
  age, and invalidation during computation cannot publish a result.
- `OpenaiVLLMClient.last_call_metrics` separates local encoding and HTTP request
  wall time, counts images and records success even when a request raises. It
  stores no request/image payloads and does not change error propagation.
- `scripts/eval_observed_reacquisition.py` is a repeatable CPU-rendered in-process
  MuJoCo sensor diagnostic with remote Caliban initialization. Object poses never
  enter association; camera calibration comes from the simulated sensor model.

The provider method is a tested integration seam, **not an automatically wired
continuous camera worker in `run agent`**. The registry test verifies that fresh
proposals are exposed and observed planning remains blocked without an execution
provider. This is bounded local reassociation, not continuous tracking during
inference, long-term re-identification, post-navigation acceptance, or proof that
an indistinguishable object was not substituted during the gap.

## Actual full-agent regression

Job `20261025_091237_dc8365` ran the actual `run agent` CLI against a live ZMQ
Stretch simulator using Caliban inference. The agent discovered roles, generated
a two-step task plan and executed it. Independent final checks passed **both
objects' region containment, support and settling**. The receipt completed two
steps with `goal_verified=true`; the audit counted **507 commands, zero forbidden
commands**. There were no failed-tool recovery cases to score.

This is a stationary **privileged-geometry** control, not the observed diagnostic
wired into native execution. It verifies that the working native task path still
passes; it does not certify observed navigation/exploration or hardware sensors.

## Remaining gate

Connect the acquisition worker to this provider seam, including explicit
post-navigation/view-motion admission and a clear-view action when occluded.
Then supply calibrated object pose/uncertainty, conservative payload/arm geometry
and visual effect verification to native observed execution. Until then,
`discovery_only` remains correct: fresh visible points alone cannot authorize a
full manipulation task.

Validation: **122 selected checks passed, zero failures/errors/skips**, including
real-registry discovery/planning rejection, replayed capture age, motion during
association, all-or-nothing snapshot publication, missing support, ambiguity,
normal segmentation and request-error telemetry. Ruff and `git diff --check`
are clean. The unit-test cases are separate from the sensor/agent experiments.
