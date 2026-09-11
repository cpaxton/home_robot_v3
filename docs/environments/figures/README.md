# Environment figures and evidence

[Environment index](../README.md)

`acceptance-progression.svg` is a conceptual diagram, not experiment evidence.
Do not substitute generated illustrations for simulator captures or robot data.

For each representative episode, retain:

- Top-down map with scale, coordinate convention, spawn, measured path and
  explored region; distinguish commanded waypoints from achieved poses.
- Full egocentric RGB with timestamp/observation ID, proposal box, selected
  support mask and depth-derived location. Keep the unannotated source.
- Third-person/chase-camera views when available to expose posture, occlusion,
  approach and contact. They are diagnostic views, not necessarily agent inputs.
- Before/after manipulation views and an independent success check; a controller
  completion receipt is not proof of grasp or placement.
- At least one informative failure alongside successes. State how examples were
  selected; do not present a hand-picked scene as a representative score.

Captions must identify environment/scene/seed, robot asset, source commit, preset,
model/quantization, task budget, motion mode and outcome. Link the experiment
record and raw artifact manifest. Mark evaluator-only GT overlays and oracle
controls explicitly. Record missing views instead of inventing them.

Keep large RGB-D caches/videos in artifact storage. Commit selected small figures
with stable relative links only after checking dataset redistribution terms and
privacy. Machine-local run paths in experiment notes are provenance, not public
download links. Follow [evaluation exports](../../evaluation.md) for capture tools.
