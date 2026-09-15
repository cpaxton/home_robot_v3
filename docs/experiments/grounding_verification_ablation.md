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

## Optional detector-proposal comparison

`scripts/cache_grounding_proposals.py` caches query-conditioned YOLOE-L masks at
the existing 0.05 proposal threshold. It reads RGB only, not simulator labels.
`audit_vlm_regions.py --proposal-cache CACHE --strategy depth_candidates` then
uses those full-image masks instead of a Qwen box. Qwen receives candidate RGB
panels, not detector scores; measured-depth validity and Qwen acceptance remain
mandatory. Missing masks mean abstention, not an implicit detector-free fallback.
This compares proposal pipelines, not just a verifier. Cache and inference run
serially; include proposal latency when estimating online cost. Production
controllers do not enable this experimental path automatically.

## Completed pilot (2026-09-10)

Capture caches:

- `/home/cpaxton/runs/emet/grounding-views-20260910`
- `/home/cpaxton/runs/emet/grounding-robocasa-v2-20260910`

Five-variant source `c6b37fc1`, job `20260910_020936_cb13d5`, outputs
`/home/cpaxton/runs/emet/grounding-ablation-local-20260910`.
Detector comparison source `de3a58ac`, job `20260910_021539_5feba4`, outputs
`/home/cpaxton/runs/emet/grounding-proposal-qwen-20260910`, cached masks in
`/home/cpaxton/runs/emet/grounding-proposal-cache-20260910`.
All inference used local Qwen3-VL-8B-Instruct int4, not Caliban FP16.
Jobs ran serially with the CPU-safe/GPU-exclusive runner and completed.

Of 40 views, 21 have any target pixels: 14 development and 7 held-out.
All methods reject all 19 zero-visibility views. A few visible targets have only
14–134 pixels; the held-out layout is particularly robot-occluded. These are
not 40 clear acquisition opportunities. Additional clear-view controls are needed
before making broad cross-scene claims, and category labels are not grasp labels.

| Variant | Pure surface / 14 visible development | Accepted below 95% purity | Pure surface / 7 visible held-out |
| --- | --- | --- | --- |
| Baseline | 5 | 5 | 0 |
| Expand 25% each side | 2 | 7 | 0 |
| Entire-visible-object prompt | 7 | 4 | 0 |
| Box verification | 5 | 5 | 0 |
| One correction allowed | 5 | 5 | 0 |
| YOLOE proposals + Qwen surface verification | 5 | 7 | 0 |

The below-purity column includes boundary contamination, not just wrong-object
selection. For example baseline bowl masks are 94.4%, 81.0%, and 77.2% pure;
YOLOE proposals improve those to 97.8%, 95.9%, and 98.8%, respectively. However,
the portrait YOLOE selection covers only 30.2% of the visible bowl, illustrating
why purity must be read with recall. Larger boxes generally worsen contamination.

Verification and repair accepted the existing accepted boxes without repairing
them; both preserve the same five contaminated acceptances. That does not mean
all verification is useless: this particular extra box check did not help the
downstream surface problem. The existing selector can reject wrong-only proposals
(e.g. sponge candidates showing produce and a bottle cap), but is not reliable.

### Manually verified wrong-object acceptance

Detector case 14, Molmo paper-towel/far: Qwen accepts candidate 0 as a paper-towel
roll. The true target has 78 visible pixels, but the selected 574-pixel mask has
zero target overlap. Simulator geometry identifies 460 selected pixels as a mug;
the remainder belongs to a coffee maker, plant, lettuce and counter. The original
frame and isolated candidate visibly support this diagnosis. This is a genuine
wrong-object acceptance, not merely the 95% threshold rejecting a near-perfect mask.

Inspect `14-surfaces.png`, `14-surface-0.png`, `14-support.npz` and case 14 in
`results.json` under the detector comparison output. The model's explanation
invented paper-roll evidence. Do not promote this verifier based on confident text.

Decision: whole-object prompting is the most promising box-only development
variant, not a validated winner. Neither it nor the detector-proposal pipeline
meets final OVMM acceptance. No production configuration or action tolerance was
changed; no new OVMM/task success is claimed. Next improve mask support and test
less target-leading candidate identification plus reacquisition on ambiguous
views. Retain this wrong-mug case as a regression test, alongside clean successes.
Only advance to a bounded integrated find/OVMM pilot after these failures improve.

Focused grounding tests: 31 passed. The detector full-image ROI is not a predicted
box; its box-overlap metrics are null in the corrected scorer. Selected-mask
scores above are unaffected by that reporting correction.
