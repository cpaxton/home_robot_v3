# GT-semantics removal + full-113 re-analysis (2026-08-15)

Branch `fix/no-gt-semantics`. Goal: make HM-EQA reflect real-world perception —
no ground-truth HM3D semantics — and re-read the full-113 numbers.

## Why remove GT semantics

"Ground-truth semantics" in HM-EQA are two things, both GT-information leaks:

1. **Habitat semantic sensor** (`.semantic.glb`): on scenes that have it, `obs.semantic`
   is a per-pixel GT instance-id map; `Hm3dSemanticLabeler` maps ids → category names,
   and `dynamem_graph_hooks.py` turns those into graph nodes with **GT labels + GT
   world positions**. A real robot has no such sensor.
2. **GraphEQA per-question enrich labels** (`hmeqa_enrich_labels.yaml.bundled`): a
   bundled `{question_id}_{scene} → "object hints"` mapping seeded into the graph at
   episode start via `seed_object_hints`, steering search by the target's GT label.

## Changes

- **`scripts/run_hmeqa_paper113_h2h.sh`**: pass `--no-hm3d-semantics` so the semantic
  sensor is off for all 113 scenes (consistent, no GT perception).
  `use_hm3d_semantics=False → _use_semantics=False → no Hm3dSemanticLabeler`.
- **`packages/emet_habitat/emet_habitat/runner.py`**: gate `enrich_labels_for_question`
  behind `use_hm3d_semantics` — GT object hints are only seeded when GT semantics are
  requested (latent leak closed even if scene IDs ever align).

## Re-analysis (2026-08-14 fixed run, both override gates on)

| Subset | dynagraph | static_graph | Δ (memory) |
|--------|-----------|--------------|-----------|
| **GT-free (76 scenes)** | **48.7%** (37/76) | **44.7%** (34/76) | **+3.9 pp** |
| with-GT (37 scenes) | 51.4% (19/37) | 54.1% (20/37) | −2.7 pp |
| pooled 113 | 49.6% (56/113) | 47.8% (54/113) | +1.8 pp |

GT-free memory delta (both=25, dyna-only=12, sg-only=9): **dynagraph +3.9 pp over
static_graph with no GT information** — the honest Dynagraph-memory claim.

## Bugs / issues found (flag)

1. **GT semantic sensor was inconsistently enabled** (auto-detect on 37/113). Fixed:
   `--no-hm3d-semantics` in the runner. The pooled number blended two perception
   stacks; stratification showed GT scenes score higher (both methods benefit).
2. **GT enrich-label seeding (latent leak)** — bundled per-question object hints
   seeded unconditionally at episode start. Currently inert (the bundled scene IDs
   `0_00006-HkseAnWCgqk` do not overlap the paper-113 episode scenes
   `0_00004-VqCaAuuoeWk`; overlap = 0/113), but gated now so it can never fire.
3. **Pooled memory delta (+1.8 pp) is masked by semantics unevenness** — the GT-free
   +3.9 pp is the defensible number; quote that, not the pooled delta, for the
   memory-vs-baseline claim.

## Open question for a truly clean claim

A fresh full-113 with `--no-hm3d-semantics` (not just the stratified subset) is the
on-record confirmation. The 2026-08-14 run had semantics auto-on for 37 scenes; the
GT-free subset (76) is our best estimate until that re-run.

## Prior art note

The GraphEQA paper uses **full HM3D GT semantics** (→ Hydra 3DSGs). Removing them
from our stack is *more* conservative than the prior art on the perception channel —
we compare GT-free vs their GT-full. If we want a semantics-matched comparison we'd
enable them for both (per the annotated-semantics slice, ~37 scenes).
