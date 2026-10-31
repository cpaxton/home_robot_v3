# Moving RGBD masks and whole-gripper geometry: October 29

The moving-stage diagnostics pass all four cases, and conservative gripper bounds
contain every sampled collision volume in two native contact controls. The actual
Stretch head-camera trial still stops during approach: the gripper obscures most
of the cylinder and the remaining shape is no longer observable. This is
**component progress, not observed pick/place acceptance**.

![Moving geometry and containment diagnostics](figures/observed_moving_geometry_20261029.png)

Videos: [moving/lifting mask](media/moving_mask_lift_20261029.mp4),
[distractor and latched stop](media/moving_mask_stop_20261029.mp4),
[Stretch contact transfer with observed tracking loss](media/stretch_head_mask_contact_20261029.mp4).
The last video's controller uses privileged state; its observed mask does not
control or certify the transfer. Borders identify those roles throughout.

## Runtime changes

- `MovingRgbdTracker` associates whole measured colour/depth components within an
  explicit motion envelope. It fits fresh geometry rather than cropping around a
  commanded pose or the old stationary reference. Multiple candidates, mixed
  geometry, insufficient observability, time/budget errors, scene changes and
  replay stop tracking. Loss clears and latches the track; appearance reappearance
  cannot revive identity. Interior reference colours avoid antialiased background
  contamination; current components are not trimmed by model fit quality.
- `ShapePoseEvidence` distinguishes a complete primitive's geometric pose class
  from unique material pose. The tools, payload retention and visual effects can
  consume declared sphere/cylinder/box symmetry with independently supplied error
  bounds. Generic unique-pose readers still reject it. Public plan/execute
  receipts distinguish `declared_shape_equivalence` and `unique_rigid_pose` and
  identify the payload construction tier. Changing shape/calibration/bounds
  invalidates stored evidence. Shape equivalence cannot certify handle or material
  orientation.
- `bound_stretch_gripper` anchors the entire tool housing, D405, fingers and rubber
  geometry at motor-derived rigid EE FK. Analytic joint intervals enclose passive
  motion; unmeasured fingertip qpos is never consumed. Default intervals use full
  model limits, including a full turn for unlimited hinges. Narrowing/overtravel
  needs a declared calibration ID. The result binds capture, all eight motor
  positions, robot geometry and EE pose before supplying visual-effect brackets.
  Pose-error margins grow the bounds. These are conditional model bounds, not a
  hardware certificate or a new full-arm interval collision provider.

## Experiments and retained failures

All jobs used CPU OSMesa or disabled rendering. No local model, local GPU or VLM
query was needed. Initial masks use explicitly operator-declared red appearance;
this does **not** measure agent semantic grounding. The head diagnostic uses in-process stepped physics and public native tools,
not a full ZMQ/CHAT agent run. Simulator poses, segmentation
and contacts are separate evaluator/scoring inputs. The cylinder prior is an
explicit complete-shape assumption, not an observed enclosure.

| Run | Outcome |
| --- | --- |
| Stage v1, `20261029_123142_6b9bcb` | Failed early: mixed silhouette colours contaminated the appearance reference and admitted tiny floor components. |
| Stage v2, `20261029_123339_887f73` | 4/4 cases after learning colours from interior reference pixels. |
| Stage v3, `20261029_124204_c6225b` | Final stricter chromaticity profile: 4/4 cases, same motion, age, geometry and rejection gates. |
| Contact envelope, `20261029_123144_4fa419` | 2/2 native controls; 175/175 sampled gripper envelopes contain actual collision-geom corners, including 75 target-contact samples. No sampled interval overruns. |
| Head camera v1, `20261029_123636_6ad76b` | Privileged control passes; 0/80 tracking candidates. Initial silhouette RGB paired with unrelated table depth prevents fitting. |
| Head camera v2, `20261029_124203_74681c` | Privileged control passes; 12/80 candidates, then `moving_mask_ambiguous` at 5.774 s. V1/v2 use ideal co-located RGBD renders. |
| Head camera v3, `20261029_125129_2715ce` | Separate RGB/depth optical frames, production FOV calibration, measured depth registration with unfilled holes. Privileged control passes; 11/80 candidates, then `moving_geometry_ambiguous` at 5.522 s. |

The stage suite contains two 13-capture translation/lift views and two negative
episodes. Positive paths translate 24 cm and lift 8 cm. Maximum final-profile
center error against the independent scorer is **0.1214 mm**; this is an ideal
rendered fixture without hardware noise. The distractor/occlusion episodes each
retain eight candidates, reject at frame eight, then require explicit new
initialization. They are development diagnostics, not four robot task successes.

The final head repeat uses an explicitly interior operator seed and tighter colour
matching (0.03 normalized-RGB distance). Initial mask overlap is intentionally
smaller than the silhouette; it supplies measured surfaces, not an object-volume
estimate. Accepted center error is at most **0.274 mm**. Visible target pixels
fall from 1349 initially to 513 at loss, mostly the cap above the gripper housing.
The final stop is insufficient geometric observability. The earlier co-located
render repeat stops on competing appearance inside its reachable motion volume.
Do not call this a solved retention or recovery episode. The privileged controller
continues only to keep the independent contact control outcome visible.

The contact samples come from two development episodes, not 175 independent
tasks. Analytic intervals are conditional on mechanical/model validity; MuJoCo
joint limits are soft and may overrun elsewhere. Separate generic tests exercise
nonzero pivots, joint references and rotated ancestors against MuJoCo FK, plus 200
random passive states/telescope splits for the complete Stretch tool subtree.

## Validation and reproduction

Managed regression `20261029_123638_26e81c`: **810 passed**, no failures/errors/skips.
After the final colour profile, motor-record binding and shape-drop test variants,
the focused motion/effects/tool integration suite passed **65 tests**, overlapping the broad
run. Tests include shape-symmetric public tool execution, axis-slip/calibration
rejection, volume enclosure, unobserved-pose rejection and lost-track boundaries.
Ruff and whitespace checks pass. The paper builds at 53 pages without undefined
references or citations.

Raw captures, unmodified failure runs, control receipts and job logs are under
`.tamp-runs/moving-gripper-20261029/`. The [JSON record](observed_moving_geometry_20261029.json)
retains attempts, per-frame score summaries, original times, validation and final
source hashes. Jobs ran from a development working tree, not frozen per-attempt
commits. Final source hashes must not be interpreted as historical attempt hashes.

Launch `scripts/eval_moving_rgbd_geometry.py --output DIR` and
`scripts/eval_stretch_moving_masks.py --output DIR` through `emet jobs run
--cpu-safe --no-gpu-exclusive -- ...` with the project's OSMesa environment.
`scripts/eval_stretch_encoder_geometry.py --output DIR --gripper-envelope` uses
`MUJOCO_GL=disable`. All scripts' help was verified. Rebuild the figure with
`scripts/plot_moving_geometry.py --input .tamp-runs/moving-gripper-20261029
--output docs/experiments/figures/observed_moving_geometry_20261029.png
--paper-output paper/figures/observed_moving_geometry_20261029.pdf`.

## Next integration gate

Qualify a view that remains informative near the gripper: test wrist RGBD and/or
planned head views, with independent identity continuity and capture alignment.
Do not infer an occluded object's pose from commands or silently reset its track.
Then bind that source to persistent support/background evidence, explicit static
scene assumptions, full-arm uncertainty and phase-specific contact/unknown-space
policy. Current-image timestamps cannot renew hidden map cells. The first full
stationary observed transfer through the public tools remains the next task gate;
held-out transfers, buffered multi-step tasks and exploration follow it.
