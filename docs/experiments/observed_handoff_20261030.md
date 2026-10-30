# Stretch head/wrist handoff during native contact transfer — October 30

The final development repeat completes native contact pick/place and retains one
RGBD track for **81/81 captures**, through approach, lift, placement, release and
retreat. This fixes the earlier head-only visibility failure for this fixture.
It is **not** a full observed TAMP success: task planning/control and task scoring
remain privileged, while the independent tracker consumes RGBD and declared
priors. No training loop, learned controller, local GPU or VLM query is involved.

[Video: head/wrist views and current measured masks](https://media.githubusercontent.com/media/cpaxton/home_robot_v3/feat/tamp-observed-integration/docs/experiments/media/stretch_head_wrist_contact_20261030.mp4).
The head panel is cropped only for presentation. Full images enter tracking.
The [JSON record](observed_handoff_20261030.json) retains attempts, original failure
captures, source hashes, profile declarations and validation.

## Changes and root causes

- A wrist-only track cannot initialize because the target is initially outside
  its view. `MultiViewRgbdTracker` retains the original head reference and permits
  wrist association while the global measured track remains live. Every view
  belongs to one shared source capture, not matching receive timestamps. Global
  loss stays latched; no reacquisition silently restores an old identity.
- Near contact, visible side and cap patches separate into several measured
  components. The explicit `complete_prior_union` profile fits **all** reachable
  components, including singleton fragments, against one declared primitive.
  It cannot select a fitting subset or trim foreign points. Default tracking
  retains single-component rejection. Optional clipped-view admission still
  requires informative measured geometry and a complete primitive prior.
- Neutral highlights change normalized RGB enough to exclude the clearly visible
  cap in the older colour profile. With only side points, sliding along the
  cylinder axis is unobservable. The opt-in `neutral_specular` profile subtracts
  the neutral colour component before intensity normalization, and rejects
  low-saturation patches. The original chromaticity distance and geometry/motion
  gates remain; no commanded or predicted attachment pose supplies missing data.
  This lighting model is not invariant to arbitrary illuminants/material changes.
- Wrist intrinsics now match the rendered 480×270 pinhole camera. Previously,
  published `ee_K` copied physical focal calibration, while the camera manager
  used 1280×720 sensor dimensions. Actual Stretch model tests compare head/wrist
  K with MuJoCo GL frusta, including both head resolutions. Physical camera
  calibration settings are unchanged; historical metric results are not rescored.

## Retained attempts

| Attempt / managed job | Native control | Tracking / outcome |
| --- | --- | --- |
| Wrist v1, `20261030_174948_752dd3` | Pass | 0/80; no target in the initial wrist view, loss remains latched |
| Paired v1, `20261030_180051_471f16` | Setup failure | Camera startup aged the privileged debug snapshot; corrected by reading a genuinely fresh state |
| Paired v2, `20261030_180938_2886e7` | Pass | 187/749; conservative wrist association stops. Excessive sampling during telemetry polling prompted a scheduler fix |
| Paired v3, `20261030_183224_d2cbdf` | Pass | 17/81; four disconnected target components trigger ambiguity |
| Paired v4, `20261030_183643_9b7083` | Pass | 56/81; fragment union works, but cap colour rejection leaves unobservable side-only geometry |
| Paired v5, `20261030_184129_cc941e` | Pass | 81/81 with neutral-highlight profile; one reference, no reset or pose fallback |

Offline replay of v3 reaches 56/81 with whole-fragment union and 81/81 after the
lighting change, then the fresh v5 contact repeat confirms the latter. Replays
evaluate original capture times on their recorded clock; they do not publish
retimed evidence to a live controller. Failed runs are preserved. Frame counts
are correlated samples, not independent tasks; v2's polling cadence also differs.

V5 uses head only for 7 captures, both cameras for 8, and wrist only for 66.
Maximum center error against independent simulator truth is **1.251 mm**; one
capture touches an image boundary. This is an ideal rendered fixture with an
operator colour seed, complete cylinder prior and shared frozen physics sample.
It does not establish agent grounding, hardware range/noise, cross-host clocks or
calibrated execution uncertainty. The tracker does not receive scorer masks or
object poses. Control uses public native tools via scripted CHAT dispatch and
in-process normal actuator commands, not the full ZMQ/LLM agent session.

## Historical map evidence and remaining execution gate

`ObservedBackground` preserves a real earlier free-space measurement when a
robot/target endpoint later occludes it. It never carves free space from a robot
AABB or fills unseen cells. Each cell retains its acquisition time; a new image
cannot refresh an old hidden cell. Static-scene extensions require an explicit
bounded assumption and unchanged epoch. Mixed ownership retains background
occupancy; contradictions reject the update and require source invalidation.

A raycast regression now drives the actual native collision checker from a prior
workspace scan: a short arm path passes, expires with the hidden scan evidence,
passes under a declared static-workspace assumption, and still fails when entering
unseen space. This synthetic geometry case establishes the query semantics, not
a live Stretch scan or the robot-interior/contact policy. Its oracle endpoint
ownership labels exercise exclusion; no live robot labeler is supplied. The default source is
still read-only. Full-arm uncertainty, support evidence and actor/phase-specific
contact admission must be assembled before a scored observed transfer is valid.
The first complete observed transfer through the public tools remains pending.

## Reproduction and validation

Launch `scripts/eval_stretch_multiview_masks.py --output DIR` through `emet jobs
run --cpu-safe --no-gpu-exclusive -- ...`, with `MUJOCO_GL=osmesa`,
`PYOPENGL_PLATFORM=osmesa`, `CUDA_VISIBLE_DEVICES=` and the project's OSMesa library
path. It declares the development cylinder/lighting profile in source and records
raw RGBD, intrinsics, optical poses, capture intervals, masks, independent scoring,
control receipts and video. `scripts/eval_stretch_moving_masks.py --camera wrist
--output DIR` reproduces the strict wrist-only initialization diagnostic. Help
for both commands was verified. No local model allocation is needed.

The final managed affected regression (`20261030_184624_3ef777`) passes **853
checks**, with zero failures/errors/skips. The overlapping 49-check focused run
preceded the final grid-boundary regression, which is covered by the broad run.
Tracker/map/actual-camera checks include fragmented and
highlighted surfaces, same-colour distractors, singleton foreign returns, cap-only
ambiguity, clipping, mismatched/replayed captures, original-cell-age expiry and
unknown-volume rejection. The affected regression and paper-build results are
recorded in the JSON evidence. Ruff and whitespace checks pass; the paper remains
53 pages without undefined references or citations. Self-review also adds a grid
boundary regression so an empty floating-point query cannot be treated as free.

Raw data, offline replay scripts and logs live under
`.tamp-runs/observed-transfer-20261030/`. Jobs ran from a development working tree;
final source hashes do not stand in for frozen historical attempt commits.
