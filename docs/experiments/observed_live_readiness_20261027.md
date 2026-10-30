# Live observed assembly readiness — October 27

**Outcome: the live camera path works; full observed execution is not qualified.**
The queue was clear. Managed job `20261027_001043_cb29cc` completed using CPU
OSMesa rendering and Caliban `Qwen/Qwen3-VL-8B-Instruct`, without local GPU/model
use. Source base was `8ae7819d` plus the diagnostic/helper changes hashed in the
[JSON evidence](observed_live_readiness_20261027.json).

## Measurements

- Fresh Stretch simulator ZMQ RGBD: 5/5 production provider/tool surface refreshes
  passed, taking 0.190–0.209 s each including compressed capture writes.
- Two semantic-initialization model calls took 8.86 s and 4.90 s; no later model
  calls. Caliban was reachable and advertised the expected model.
- The latest frame had a 10.4 ms local receive age and 86.2% in-range depth pixels.
  Receive age is not capture/encoder alignment qualification.
- Depth carving produced 8,343 free, 35 occupied and 5,446 unknown cells in the
  bounded diagnostic volume. This is not separated execution collision geometry.
- Actual production reference-feature admission returned
  `rigid_tracking_insufficient`: the selected visible object surface contained
  only 217 pixels, 15 after the tracker's 9x9 interior guard, and **zero detected
  material landmarks**. This establishes a view/feature admission failure; it
  does not isolate texture from resolution as a causal factor.
- Default `plan_pick_place` and `execute_pick_place_plan` both returned
  `observed_scene_unavailable`. The actuation audit was unchanged across these
  admission checks. No pick/place action was attempted.

Successful surface refreshes return an empty task list here: the diagnostic seeds
one object and no support. These are not five successful discovered tasks or
executed goals. No task-success denominator is reported.

## Assembly gates

| Gate | Result |
| --- | --- |
| Live calibrated RGBD/provider/tool refresh | Passed this stationary sensor smoke |
| Rigid material-feature pose tracking | Unknown / rejected at reference admission |
| Capture interval and encoder/alignment bounds | Not configured; renderer timestamps alone do not supply them |
| Complete conservative payload geometry | Not configured; partial visible points are insufficient |
| Robot/target occupancy provenance | Not configured; carving alone does not separate them |
| Observed support grounding | Not configured in this single-object diagnostic |
| Visual manipulation effects | Not assessed; no manipulation attempted |
| Default execution refusal | Passed; no fallback or extra actuation |

The current simulator head renderer also suppresses robot geom groups; this run
does not qualify hardware self-occlusion.

## Implementation and follow-up

`eval_observed_capture_binding.py --execution-readiness` now records this
prerequisite audit and preserves original image timing/receive metadata in saved
captures. The feature admission helper is shared with `RigidRgbdTracker`, so the
live diagnostic does not approximate or loosen production thresholds. All 25
selected readiness/tracker/observed-adapter regressions passed; Ruff and
`git diff --check` pass. No default execution capability changed.

Next assemble a source in dependency order:

1. Acquire a closer or more informative measured view and run pose admission
   before planning a grasp. Do not loosen guards or infer an orientation to force
   this small surface to pass. Textureless/symmetric shapes still need explicit
   pose ambiguity handling rather than fabricated unique yaw.
2. Supply complete conservative geometry (qualified observed multi-view bounds,
   or an explicitly separate known-model prior) and provenance-preserving
   robot/target/background occupancy.
3. Establish camera/encoder clock and inter-sample motion bounds and observed
   support geometry, then wire the concrete source into the existing adapter.
4. Run real stationary transfers and failure cases before the same-agent swap
   and navigation/exploration gates. Keep privileged controls separate.

Raw artifacts: `.tamp-runs/live-qualification-20261027/live/` (reference image/mask,
calibrated captures, timing, model-call metrics, JSON receipts and audit);
job log: `.tamp-runs/live-qualification-20261027/job/job.log`.
