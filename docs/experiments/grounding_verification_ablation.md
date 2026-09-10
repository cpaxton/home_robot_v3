# Cached grounding verification diagnostic

Purpose: test localization and final surface verification before another OVMM
pilot. This is not robot navigation, a grasp test, or an end-to-end benchmark.
Production defaults are unchanged. Detector proposals remain a separate possible
search mechanism; these particular replays use Qwen boxes and RGB-D components.

`scripts/build_grounding_dataset.py` captures original Molmo/RoboCasa scenes from
oracle-aimed free cameras, without physics stepping. Views are fixed in config;
occluded targets and cameras blocked by walls/robots are retained. Camera placement
uses simulator geometry, so these images cannot support exploration claims.
Inference NPZ files contain only RGB, depth, intrinsics and camera pose. Separate
truth files contain descendant-object geometry masks and source XML hashes.
Do not redistribute third-party scene assets without checking their licenses.

Initial cache: 16 Molmo queries (bottle, bowl, sponge, paper towel) and 24 RoboCasa
queries (sugar cube, broccoli, bell pepper; turmeric, pickle slice, lime), four
views each. RoboCasa layout/style 2, seed 1 is reserved as held-out. This is a
small scene-level holdout, not a generalization claim. Inspect RGB and visibility
counts before interpreting scores; zero visibility measures rejection only.

Run capture serially using the managed CPU-safe/GPU-exclusive job mechanism:

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/build_grounding_dataset.py \
  --config configs/ovmm/grounding_views.yaml --output-dir NEW_MOLMO_CACHE
MUJOCO_GL=egl PYTHONPATH=src python scripts/build_grounding_dataset.py \
  --config configs/ovmm/grounding_robocasa_views.yaml --output-dir NEW_ROBOCASA_CACHE
PYTHONPATH=src python scripts/run_grounding_ablation.py \
  --datasets NEW_MOLMO_CACHE NEW_ROBOCASA_CACHE --output-dir NEW_RESULTS
```

The Molmo config currently names a local cached scene XML; override that path in
a copied config on other machines. RoboCasa uses the existing seeded generator.
The first RoboCasa capture attempt failed on reserved spawn metadata; v2 excludes
that metadata from object enumeration. Failed artifacts are retained.

Five predeclared variants:

- Baseline: existing box plus measured-surface selector.
- Expand: 25% of box width/height padding on each side, clipped to image bounds.
- Whole-object prompt: explicitly request all visible object edges.
- Verify: original image, annotated box and unmarked crop; reject wrong or
  incomplete boxes. No previous model reasoning is provided.
- Repair: same verification with at most one corrected original-image box,
  followed by the existing surface selector. No recursive retries.

Expansion/verification/repair replay baseline box responses exactly; whole-object
prompting requires fresh localization. Every actual prompt, response, latency,
box overlay, verification crop, candidate panel and selected mask is saved.
Replay latency excludes the cached first model call and must not be confused
with full online latency. Same-model agreement is not independent verification.

Scoring reports box/mask overlap, visible target recall, selected-mask purity,
false acceptance and latency separately by scene split. A >=95% pure selected
surface is a diagnostic only, not complete object geometry or a safe grasp.
No minimum-recall gate is hidden in that count: inspect recall alongside it.
Never choose a winner only by acceptance rate or condition on successful frames.
Keep development and held-out outcomes separate; do not tune on held-out results.
