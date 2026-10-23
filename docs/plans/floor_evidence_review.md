# Floor-evidence review slice

Status: opt-in prototype for review, not navigation or hardware acceptance.
Independent of the frozen movement/EQA acceptance source in PR #220.

## Changes

- Fix DynaMem ignoring `use_negative_obstacles`: classify measured below-floor
  points before negative Z is clamped. Preserve those obstacles even when the
  legacy visited/self mask would erase them. The enabled-flag regression failed
  against the old implementation and now passes.
- Add optional calibrated floor layers: measured support, bounded inferred
  single-cell holes, lower-surface evidence, and raised-surface evidence.
- Inference requires four measured, height-consistent neighboring floor cells;
  no iteration, frontier growth, filling measured conflicts or bridging known
  discontinuities. A no-return hole is not classified as a drop by itself.
- Inferred cells remain hypotheses only: they do not enter physical known-floor
  coverage or sensor-observed coverage. A narrow unseen pit must not become
  traversable solely because neighboring cells look level. No automatic
  inspection-candidate creation is implemented in this slice.
- Opt-in physical navigation uses measured floor support rather than the visited
  disk or generic image-coverage morphology. Height gates use the explicitly
  calibrated floor reference; semantic points keep their original world frame.
- Drop/raised-surface hints reach rejected-footprint feedback. Positive and
  negative hazards remain hard rejections, independent of model confidence.

## Configuration and limitations

Existing defaults are unchanged. A calibrated simulation prototype can use:

```yaml
floor_evidence:
  enabled: true
  reference_z_m: 0.0
  tolerance_m: 0.03
  hazard_height_m: 0.08
```

The reference must be correct in the mapping frame. These thresholds are test
parameters, not robot-specific certified step/drop limits. This is not a slope
or stair-traversal model, nor a guarantee that every cliff is detected. Retained
voxel points can lose thin/sparse structure; dense-depth hazard qualification
and sensor-specific tests remain required before enabling on hardware.

This does not yet implement measured footprint traversal history, qualified
spawn-state support, bounded exploratory motion, or a complete resolution of
the seven blind rear cells. Missing returns remain unknown to navigation.
Do not claim Molmo recovery or EQA non-regression from this prototype.

## Reproducible tests

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 12 .venv/bin/python -m pytest -q \
  src/test/mapping/test_floor_evidence.py \
  src/test/mapping/test_navigation_map_contract.py \
  src/test/motion/test_navigation_sweep.py \
  src/test/controller/test_validated_relative_motion.py \
  src/test/eval/test_agent_observation_cases.py \
  src/test/eval/test_observation_scene_geometry.py
```

Cases include measured flat floor, raised step, drop, no return, conflicting
heights, bounded hole interpolation, shifted floor reference, visited-mask
hazard preservation, and footprint rejection. MuJoCo downward-ray controls use
actual surface intersections, not a model or full robot-motion acceptance.

Review/acceptance order: negative-obstacle preservation; floor-layer semantics;
saved sensor replay/visual checks; qualified simulated robot/drop fixtures;
only then an opt-in live comparison. Keep the default #220 acceptance separate.
