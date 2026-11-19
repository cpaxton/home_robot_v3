# Agent camera-stream binding — October 26, 2026

Observed CHAT now binds the concrete RGBD task provider to the robot client's
existing camera stream. `scene_tasks` obtains the latest capture and runs local
surface reassociation, preserving capture identity and its original receive time.
It does not start another inference worker or repeat semantic search. The normal
perception processor seeds the immutable reference masks/world geometry. Negative
segmentation, expired references and external navigation/view-motion discard it.

The current feature branch also incorporates main `61da55cb` in merge `0f6cb3fd`.
Conflicts preserved both agent/control appendices and both branches' timing/view
metadata; a duplicated `image_timing` parameter introduced by automatic merging
was removed. Post-merge syntax/lint checks and 179 targeted tests passed with
zero failures/errors/skips. The paper builds to 52 pages with no unresolved
citations/references in the final LaTeX log; existing font/float warnings remain.

## Live sensor finding

The initial live ZMQ probe failed all five updates despite the in-process renderer
control passing. Raw snapshots exposed a simulation calibration problem: head
RGB and depth cameras have distinct optical origins, but the stream paired their
pixels without registration and published spec intrinsics instead of the actual
render projection. Thus RGB object pixels could receive support-table depth.

The camera manager now reprojects **measured depth samples** into the RGB frame
using calibrated rigid-camera extrinsics and a nearest-depth z buffer. Unsampled
pixels stay zero/unknown; no scene object geometry, synthesized depth, interpolation
or hole filling is used. Intrinsics now describe render FOV, image dimensions and
pixel centers, including the clockwise head-camera rotation. Both camera-manager
publication paths apply the registration. The wrist path is unchanged.

Normal proposals also preserve the boundary between near-horizontal and upright
bands, preventing gradual local normal chains from joining an object's side to
its support. This still does not prove semantic identity or split coplanar touching
objects. Registration and the normal-boundary refinement were tested together;
the development runs are not an isolated causal ablation of either change.

This is spatial registration, not exposure synchronization. It does not certify
RGB/depth/joint-pose timing on moving hardware. The inherited head renderer also
suppresses robot geom groups for mapping; these sensor smokes therefore do not
qualify hardware self-occlusion. Simulator/evaluator truth remains separate from
provider evidence.

## What was tested

The live smoke uses real simulator ZMQ RGBD frames, a Caliban-selected object
surface, the production binding helper and the actual tool registry. Subsequent
requests make no model calls. It deliberately seeds only one object and **no
support**, so successful reassociation returns an empty discovery task list.
This tests the camera/provider/tool path, not successful task discovery, the LLM
policy, native manipulation, navigation or exploration. Registry/processor unit
checks separately exercise complete object/support proposals and execution refusal.

| Run | Outcome | Meaning |
| --- | --- | --- |
| `20261026_090542_05a69c` | 0/5 updates | initial unregistered sensor pair |
| `20261026_090934_7988ff` | harness stopped before inference | missing required physical audit path; zero VLM calls |
| `20261026_091130_d07ad4` | 0/5 updates | instrumented raw pair, mixed table/object geometry |
| `20261026_103424_011249` | 5/5 updates | corrected calibration plus normal boundaries, before main merge |

The passing pre-merge requests took 0.198–0.219 seconds each, including the
diagnostic compressed-RGBD write. No inference occurred during these updates.
The semantic reference remained bounded at 30 seconds; current geometry remained
bounded at two seconds. The older failed conditions remain in the records.

The repeatable harness is
[`scripts/eval_observed_capture_binding.py`](../../scripts/eval_observed_capture_binding.py).
It records source/script hashes, both initialization model calls, raw reference
and current RGBD arrays, matching diagnostics and the tool JSON. It exits nonzero
on rejected reassociation. Earlier scratch runs recorded the raw failure correctly
but exited zero; their job completion status is not their sensor success score.
Raw data is under `.tamp-runs/capture-binding-20261026/`.

## Remaining observed execution contract

The production native table adapter is explicitly privileged. The observed
physical executor requires an observed object-pose provider and observed 3D
collision geometry; a current visible-surface centroid does not meet either.
Do not construct an identity rotation or a complete payload box from partial
surface points and relabel it observed pose/volume.

The remaining provider must supply a tracked reference frame with uncertainty,
conservative attached-object/robot volume checks against observed occupancy and
unknown space, and visual grasp-retention/release/support evidence. Robot-only
kinematics and synchronized measured feedback must replace scene-truth queries.
Until those are connected and qualified, discovery remains `discovery_only` and
planning/execution stays blocked. Assisted articulation and privileged native
controls are separate capabilities, not replacements for those observations.

## Post-merge verification

Source `6eb7c242` includes merged main `61da55cb`. The repeatable sensor smoke
(`20261026_104104_f3d9e6`) passed **5/5** refreshes with exactly two initialization
VLM calls and no later inference. Requests took 0.195–0.235 seconds, including
compressed-frame diagnostic writes. Its task list is intentionally empty because
there is no support seed; this is not five successful tasks.

The separate full native CHAT control (`20261026_104106_015947`) discovered roles,
planned and completed **two transfers**. Both final goals passed containment,
support and settling checks. The receipt reports `goal_verified=true`; the audit
contains **529 commands, zero forbidden commands**. This control uses privileged
geometry and camera-free execution, so it does not qualify observed manipulation.

The broader regression job (`20261026_104721_86d491`) passed **369 checks, zero
failures/errors/skips**, including TAMP API contracts, live-source binding,
processor ingestion, fresh/replayed/stale captures, motion invalidation,
real-dispatch stopping, depth registration and incoming-main evidence-view
selection. The earlier 179-check selection overlaps this suite; counts are not
summed. Syntax/lint and the 52-page paper build also pass. No unresolved references
or citations remain in the final paper log; existing font/float warnings persist.

[All recorded attempts and independent control results](observed_capture_binding_20261026.json)
retain the initial failures. None were rescored after changing calibration.
