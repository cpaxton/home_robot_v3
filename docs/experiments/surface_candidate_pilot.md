# Detector-free surface candidate pilot

This is an experimental perception component, not OVMM or grasp acceptance.
The shared controller enables it through `query_memory.region_strategy:
depth_candidates` in `configs/emet/query_surface_pilot.yaml`. Production defaults
remain unchanged. The same grounding handoff serves EQA, OVMM and TAMP; task
success and action readiness remain separate from selecting a visible surface.

## Contract

1. Qwen supplies a search box and may reject an absent/ambiguous referent.
2. Aligned RGB-D produces observation-local, class-agnostic connected surfaces.
3. Qwen sees the original image followed by separate labeled candidate images,
   and selects one ID or abstains. Invalid IDs and ambiguous selections fail.
4. Only measured pixels in that mask enter calibrated geometry and the existing
   memory admission path. A selected patch is not a complete object or grasp.

Local connectivity uses adjacent depth differences at most 8 cm and maximum
RGB-channel differences at most 35/255. Components need 25 valid pixels; more
than eight proposals causes abstention, not arbitrary truncation. These are
prototype constants, not calibrated sensor confidence. The low-level API can
accept external class-agnostic masks, but no SAM service is wired in this pilot.

Touching same-appearance objects can remain joined. Texture can fragment a
surface beyond the proposal limit. Occlusion, bad calibration, stale frames,
reflective depth and incorrect VLM selection are not solved by connectivity.
Another view is required when evidence fails; this change does not establish
a new autonomous retry policy or permit unsafe motion.

## Reproduction and review

Use `scripts/audit_vlm_regions.py --manifest MANIFEST --output-dir NEW_DIR
--strategy depth_candidates`. Manifest entries specify `arrays` (NPZ containing
depth and RGB, optionally camera_K/camera_pose), optional RGB image path, query,
and optional description. Simulator segmentation/evaluator labels are never
sent to the model. Missing calibration means no XYZ result.

Repeat with `--depth-noise-std-m 0.005 --depth-dropout 0.1 --seed 0` for synthetic
depth stress. This is not a model of all real sensor errors. The audit retains
perturbed depth, exact bit-packed masks, individual candidate images, human
contact sheets, prompts and raw responses. Review mask purity and localization,
not merely whether a depth-valid point or plausible centroid was returned.

Run one GPU-exclusive CPU-safe job at a time, with numerical threads set to one.
Freeze source, model and input manifest; keep failed revisions separately.

## Development evidence

- `6d38acb6`: tinted overlays recovered the stationary red cylinder but selected
  table for blue; overlays obscured small objects.
- `1e4a9236`: enlarged contact sheets gave six of six clean red/blue centroids
  within the 15 cm diagnostic tolerance and four of four noisy-route centroids.
  Manual masks still included support-table pixels: **not acceptance**.
- `135433c1`: local RGB-D boundaries produced pure target patches in the static
  fixture, but Qwen confused the first reference tile with candidate zero and
  selected background. Textured sofa exceeded the proposal limit and abstained.
- `34b98af9`: separate candidate images, no reference tile among the options.
  Managed job `20260909_160725_a5963e` tests eight static/relation/absence cases,
  four post-turn queries and four identical queries with noisy depth. These are
  correlated diagnostics, not sixteen independent benchmark episodes.

The lamp miss and learned end-to-end OVMM/TAMP failures remain explicit blockers.
No robot-specific prompts, evaluator labels, relaxed motion tolerances, or
production-default promotion are justified by this pilot.

### Final frozen audit (`34b98af9`)

| Diagnostic | Outcome |
| --- | --- |
| Stationary red cylinder / blue cube | 2/2 selected target surfaces |
| Same objects after two previously accepted turns | 4/4 localized |
| Same post-turn frames, 5 mm noise + 10% depth dropout | 4/4 localized |
| Absent object or unverifiable relation | 4/4 abstained |
| Visible lamp | Missed: semantic localization abstained |
| Textured sofa | Abstained: more than eight proposals |

All ten red/blue surface centroids are within 15 cm of known object centers
(maximum 10.2 cm). This metric includes the surface-to-center offset; it is not
grasp pose accuracy. In the clean stationary fixture, simulator geometry IDs
used **only after inference** show 100% selected-mask purity for both objects;
the patches cover 34.3% of visible cylinder pixels and 22.3% of visible cube
pixels. Post-turn masks were visually inspected on representative clean/noisy
panels, not scored with unavailable per-frame geometry IDs.

Implementation checks: 242 focused tests passed; one pre-existing CLI test that
can load a real model was deselected. No new live OVMM/TAMP success is claimed.
The next gate is clutter/occlusion and textured-room localization, followed by
bounded end-to-end tasks; do not promote this simple-scene result as real-world
robustness or a paired benchmark improvement.

Host artifact bundle (inputs, exact model evidence, and failed revisions):
`/home/cpaxton/runs/emet/jobs_runs/separate-surface-cached-pilot/evidence.tar.gz`.
The managed job directory also retains the launch command and complete log.
This host-local archive is not yet a public paper dataset.
