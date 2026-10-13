# Shared camera-view selection

`emet.memory.view_selection` is a standalone, NumPy-only evidence selector for
EQA, OVMM and TAMP. It creates no instances, acquires no images, runs no models,
and changes no live defaults. An opt-in shared-history adapter now connects it
to assessment and final answering. It is a follow-up to #199, whose two-image pilot
fixed evidence delivery but did not improve its four-question score.

## API and contracts

```python
from emet.memory.view_selection import SelectionConfig, ViewFrame, select_views

selection = select_views(
    retained_frames,                 # list[ViewFrame], owned RGB snapshots
    current_id="arrival:42",
    pre_turn_id="arrival:41",        # optional; must be the actual pre-turn capture
    config=SelectionConfig(policy="pose", max_images=4),
)
images = [view.frame.rgb for view in selection.views]
labels = selection.prompt_labels()
audit = selection.to_dict()          # JSON-safe; does not embed image bytes
```

- Pin current RGB first and an explicitly supplied pre-turn view second when
  budget allows. Never infer that the last pre-approach frame was pre-turn.
- Fill remaining slots greedily; ties use recency and stable view ID. Exact
  duplicate pixels cannot consume another slot, even for a pinned pre-turn ID.
- Candidates default to the most recent 32 eligible snapshots (pins protected),
  acquired no more than 60 seconds before current. These are experimental
  bounds, not claimed optima. The caller still owns bounded acquisition/storage.
- Required `scope_id` identifies one episode/world/map frame; timestamps must
  use the same acquisition clock. Reject other scopes and future/old frames.
  A caller-specified invalid pre-turn frame raises rather than silently lying.
- Pose is an orthonormal world-from-optical transform, +Z forward, translation in meters.
  Reflected world bases such as Habitat's X,Z,Y map axes are supported; one scope
  must use consistent handedness. No projection or navigation transform is changed.
  Missing pose is unknown, not zero displacement. Roll alone adds no pose novelty.
- Embeddings require `embedding_space` naming encoder/checkpoint/preprocessing;
  mismatched dimensions/spaces are rejected. Supply cached CLIP/SigLIP features;
  the selector never invokes the encoder. Query features use that same space.
- `visible_cells` are stable string IDs of actually observed map cells in the
  same grid. To study immediate-area coverage, pass cells intersecting a fixed
  local ROI for every candidate. Do not substitute a nominal camera frustum or
  inferred traversability for observed coverage.
- `saliency` is an optional caller-computed [0,1] score. Its definition must be
  frozen per experiment; saliency is not correctness, target presence, or safety.
- Labels explicitly separate current visibility from historical context.
  Manipulation still needs current RGB-D grounding; selection cannot authorize
  an action or prove a target is visible.

## Selection scores

| Component | Definition | Important limitation |
| --- | --- | --- |
| Recency | `exp(-age / tau)`, default tau 20s | Recent does not mean informative. |
| Pose novelty | Minimum over selected views of `min(1, max(distance/1m, view-angle/60deg))` | Geometric diversity is not semantic relevance. |
| Embedding novelty | Minimum normalized cosine distance `(1-cos)/2` to selected embeddings | Novel images can be distractors. |
| Coverage gain | Newly observed cells divided by the eligible, capped, deduplicated pool's cell union | Not whole-scene coverage or target recall. |
| Query relevance | `(1+cos(query, frame))/2` | Similarity is not a probability or detection. |
| Saliency | Caller-supplied scalar | Can prioritize visually striking irrelevant content. |

Policies `recency`, `pose`, `embedding`, and `coverage` isolate those components;
`hybrid` uses configurable nonnegative weights in the order above. Defaults are
illustrative, **not tuned**. One-hot hybrid weights give relevance-only and
saliency-only ablations. Missing components score zero without renormalization;
when all relevant scores are missing, recency/ID break ties. Feature support and
individual marginal scores are recorded so this fallback is visible.

## Audit metrics

Every result includes selected IDs/roles/timestamps/ages, per-selection feature
scores, rejection reasons, image/pixel counts, maximum age, mean pairwise camera
translation and view angle, embedding distance, visible-cell union and fraction
of the candidate union, available-feature counts, and selector wall time.
Unavailable geometric/semantic metrics are null, not fabricated zeros.
Selector timing excludes acquisition, embedding inference, and VLM inference.

These are explanatory metrics, not evidence that answers improve. For paired
pilots also measure answer/task success, abstention and wrong commitments,
annotated supporting-landmark recall in the selected packet, current-target
visibility, VLM latency/token cost, and total episode latency. Any simulator GT
or manual annotations used for scoring must stay out of online selection.

## Offline comparison

`scripts/compare_view_selection.py` runs all five policies over the same frozen
JSON manifest. Paths are relative to the manifest (absolute paths also work).

```json
{
  "current_id": "arrival:42",
  "pre_turn_id": "arrival:41",
  "frames": [
    {"view_id": "arrival:41", "scope_id": "episode-1:grid-v1",
     "timestamp_s": 100.0, "image_path": "before.png"},
    {"view_id": "arrival:42", "scope_id": "episode-1:grid-v1",
     "timestamp_s": 102.0, "image_path": "after.png"}
  ]
}
```

Optional frame fields: `camera_pose`, `embedding`, `embedding_space`,
`visible_cells`, `saliency`. Optional top-level query fields: `query_embedding`
and `embedding_space`. Without optional features, the corresponding policies
fall back to recency, explicitly reflected in the audit; that is not a valid
embedding/coverage efficacy experiment.

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_view_selection.py manifest.json --max-images 4
```

Start with matched **2/4/8-image** comparisons on the same candidate histories,
model, image resolution, capture policy and task budgets. Keep pins constant;
with budget two, both pinned views exhaust the budget and ranking policies
cannot differ. Compare selectors offline before spending GPU time on answers.
Separate acquisition ablations from selection ablations: a selector cannot
recover curtains that were never captured.

## Remaining integration

### Opt-in adapter (2026-10-12)

Set `eqa.view_selection: {policy: pose, max_images: 4}` in the controller
parameters, or `EMET_EQA_VIEW_SELECTION_POLICY=pose` for a frozen experiment.
Default behavior is unchanged. Selection uses existing voxel-frame history,
not a second rolling RGB cache, and obeys the final-answer image limit (at most
four). Missing current acquisition metadata explicitly falls back to the old
packet and emits `view_selection_unavailable`; do not score that as selector
acceptance. Only pose and recency have live features in this first adapter.

Mapping preserves `Observations.image_timing` in frame metadata. Habitat stamps
its synchronous render completion with a monotonic timestamp and retains one
observation at the actual path-to-final-yaw boundary, without issuing extra
motion. That is the pre-turn pin. Other adapters do not yet supply this capture;
they must not manufacture it from a pre-approach image. Existing hardware capture
timestamps are preserved when supplied; missing simulator/legacy timestamps
remain missing.

The selected images and individual time/role labels reach both assessment and
final answering. Audit rows record selection/exclusions. Grounding still uses
only current RGB-D. A labeled packet that would be truncated or deduplicated at
submission is rejected rather than silently relabeled.

This implementation is not an accuracy result. Run the four-question diagnostic
before the broader paired dev set; separately audit answer-policy disagreement.

### Remaining gates

1. Validate Habitat's actual pre-turn capture in saved pilot images and extend
   the same acquisition contract to the MuJoCo/physical adapters.
2. Verify clock/frame scopes across episode resets and supply cached embeddings
   or observed map cells before testing those selectors online.
3. Audit the exact selected RGB packet and labels in live assessment/final
   answering, including graph-backed and voxel-only paths.
4. Re-run the EQA diagnostic pair and small OVMM gate before changing defaults.
   This module alone does not claim to repair #199's remaining failures.

## Target-local weighted coverage (2026-10-13, offline opt-in)

Objective: maximize `sum(w[cell] for cell in union(selected.visible_cells))`
under an image budget, with the assessment image pinned first. Gaussian weights
form a hill around the supplied object hypothesis, not the camera. Each XY
cell earns credit once. Greedy selection adds the greatest marginal weight;
it stops if no positive gain remains, rather than padding with redundant views.
Existing recency, pose, embedding, uniform coverage and hybrid policies remain
unchanged. Semantic similarity is not secretly mixed into this objective.

Modular boundaries:

- `view_coverage.CoverageGrid`: bins observed world XY surface points into stable
  cells. No ray filling, inferred free space, or dilation. Filter invalid depth
  before passing points; all views must share map origin/resolution/axes.
- `gaussian_coverage_objective`: builds target weights; sigma is explicit and
  independently ablatable. `CoverageObjective` also accepts arbitrary fixed
  nonnegative weights for uniform/other-kernel comparisons.
- `select_views(..., coverage_objective=...)` with `weighted_coverage`: selection
  only. Requires explicit visibility for every eligible view and matching scope.
  Empty visibility is allowed; missing visibility is an error, not fabricated
  coverage. The report includes per-view marginal weight, selected union cells,
  total objective/candidate/selected weight and visibility provenance.
- `compare_view_selection.py`: a frozen manifest optionally supplies
  `coverage_objective` (`scope_id`, `cell_weights`, `visibility_source`) and
  `selection_config`. All policies receive the identical candidate pool.

This is **not enabled in the live history adapter**: that adapter currently
provides neither target-local weights nor measured visibility. Do not set the
new policy in live config until those inputs are wired and audited. A clock
domain alone is not a sufficient cross-episode map identity.

`scripts/audit_local_view_coverage.py` builds numbered RGB comparison sheets
and optional serial Qwen assessments from saved Habitat episodes. It freezes
pixels with hashes, matches the assessment image exactly, retains only frames
through that image, and records selected IDs, prompts, raw responses and seeds.
Example invocation (add `--infer` only inside a CPU-safe, GPU-exclusive job):

```sh
PYTHONPATH=src python scripts/audit_local_view_coverage.py \
  --episode /path/to/q0056_lazy_graph --episode /path/to/q0012_lazy_graph \
  --output /tmp/new-view-audit --sigma-m 1.5
```

Its geometry is explicitly an **unoccluded base-pose/FOV proxy**, not recovered
depth coverage. Sequence indices replace timestamps only to disable age pruning;
no synthetic time is shown to the model. Saved rendered frames form a larger
pool than the original agent's admitted observations, so this tests potential
selection, not reproduction of an online score. Repeated calls on two frozen
examples are diagnostics, not independent benchmark episodes.

Visual inspection: q56 weighted coverage recovers bookshelf/curtain context
that recency loses. On q12 it recovers a wider bedroom view but also selects a
wall-facing view because the unoccluded FOV overclaims coverage. Measured surface
visibility is the next required comparison before online promotion. Counting
one visible bedside table cannot establish the room total.

### Completed frozen-image diagnostic

Serial job `20261013_130237_4b9632` completed 30 calls using local
`Qwen/Qwen3-VL-8B-Instruct`, int4, SDPA. Final RGB fixed, four-image maximum,
Gaussian sigma 1.5 m, identical prompts across selection policies. Current-only
is a one-image control. No inventory or segmentation hints were supplied.

| Policy | q56 bookshelf/curtains (3 calls) | q12 bedside-table count (3 calls) |
|---|---|---|
| Current only | 3 abstentions | 3 wrong: one |
| Recency | 3 abstentions | 3 abstentions |
| Pose diversity | 3 correct: blue curtains | 3 correct: two |
| Uniform XY/FOV coverage | 3 correct: blue curtains | 3 wrong: one |
| Target-weighted XY/FOV coverage | 3 correct: blue curtains | 3 wrong: one |

Full-resolution manual audit of q12 frames 57 and 59 finds two distinct bedside
tables in both (one between beds and one to the right); the right wall also has
a mirror. Weighted coverage includes frame 59, so its wrong answer cannot be
attributed simply to absent count evidence. Do not infer that maximizing proxy
coverage maximizes answer accuracy, or that pose diversity wins generally from
two examples. These are three repeated calls, not three independent episodes.

Artifacts (local, not uploaded):
`/tmp/emet-weighted-view-inference-20261013/{q0056_lazy_graph,q0012_lazy_graph}/`
contains frozen frame pools, manifests and `comparison.jpg`; root
`responses.jsonl` preserves prompts, raw replies and selected IDs.
Next: measured depth-surface cells versus the optimistic FOV proxy on a newly
captured pool; then additional frozen tasks before any online default change.

### Measured-depth follow-up

`depth_surface_cells` projects optical-Z depth with calibrated intrinsics and
the full optical-to-map transform. It bins only observed surface endpoints in
XY, rejects invalid depth and credits no free ray segments or hidden cells.
Grid resolution, maximum depth and pixel stride are explicit parameters. This
accepts Habitat's left-handed map-axis permutation as well as robot world axes.

`scripts/recapture_view_depth.py` re-renders saved poses with semantic sensing
disabled and saves paired RGB/depth/intrinsics/optical poses plus per-frame RGB
differences. It is an offline sensor replay, **not evidence of navigation
success**. Navmesh elevation must be restored: assuming constant spawn height
mis-reproduced q12 by 20 cm. After correction, worst per-channel RGB MAE is
0.00181/255 on q12 and 0.00256/255 on q56. Q56's final image matches exactly;
q12 final MAE is 0.00055/255. Always use co-rendered RGB with the replayed depth,
not a mixture of original pixels and unverified reconstructed geometry.

The audit script accepts one `--depth-replay` directory per `--episode` and
`--max-images 2|4` for the image-budget ablation. The Gaussian width (1.5 m),
grid (0.2 m), depth range (4 m) and stride (4 pixels) were held fixed. Traces
retain the original unverified navigation target, not a hand-labeled target.
All policies share the replayed RGB pool; the current assessment frame stays
pinned. Comparisons disable history-age pruning to separate selection from
retention. Live defaults remain unchanged.

Jobs: `20261013_134830_123217` sensor replay, `20261013_135046_6a22e6`
elevation-corrected q12 replay, `20261013_135208_21e02f` measured-depth budget
comparison, `20261013_135257_67b850` unchanged-parameter q14 fan check.

All follow-up jobs completed. 84 additional Qwen calls (60 budget comparisons,
15 fan checks, 9 isolated-frame checks), serial; these are NOT 84 episodes.

| Measured-depth policy | q56 N=4 / N=2 correct | q12 N=4 / N=2 correct | q14 N=4 correct |
|---|---|---|---|
| Recency | 0/3 / 0/3 | 0/3 / 0/3 | 1/3 |
| Pose diversity | 3/3 / 0/3 | 3/3 / 2/3 | 0/3 (abstained) |
| Uniform coverage | 3/3 / 3/3 | 0/3 / 0/3 | 0/3 (abstained) |
| Weighted coverage | 3/3 / 3/3 | 0/3 / 0/3 | 0/3 (wrong TV relation) |

Current-only: abstained on q56/q14, wrong one-table count on q12, all repeats.
No policy is promoted. Weighted coverage fixes the useful-context shortage on
q56 at N=2 but does not dominate pose diversity. q12 weighted coverage reaches
98.2% of available weighted cells at N=4, yet answers incorrectly. Coverage is
not calibrated answer confidence. Full-resolution manual checks find the two
tables in the wider selected frame, smaller than in the closer pose-selected
frame. A 2.5 m maximum-depth ablation (selection only) selects the same q12
weighted pair; no extra model calls were wasted on identical pixels.

Isolated-frame diagnostic (`20261013_135634_ca3f22`): q12 frame 55 (wide,
coverage-selected) yields two in 1/3 calls; frame 65 (closer, pose-selected)
yields one in 3/3; final frame 78 also yields one in 3/3. The combined final+65
packet yields two in 2/3 calls. Thus neither a simple resolution cutoff nor
"the final image always distracts the model" is established as the explanation.
Do not introduce a task-specific count prompt or tune a cutoff to these frames.

q14's weighted packet visibly contains fan/window/sofa and room context. All
three answers select "Next to the TV"; one raw reply explicitly says the fan
is NOT next to a TV and the living room is the only plausible option, while
its answer field still selects the TV. Raw reply and parsed field agree on
that incorrect answer: this is not answer-parser corruption. Current evidence
does not justify asserting that further coverage optimization repairs it.

Validation: 75 focused selector/history/captured-view/assessment tests pass;
ruff and diff whitespace checks pass. Live defaults and main are untouched.
The behind-wall coverage defect is repaired; broad task-performance restoration
is **not achieved**. Next acceptance requires more frozen examples with manual
evidence-sufficiency labels, then a live adapter providing target identity,
map scope and paired surface visibility. Retain pose diversity as the baseline;
do not silently turn the offline objective into a live default.

Artifacts: `/tmp/emet-depth-weighted-inference-n{2,4}-20261013/`,
`/tmp/emet-depth-fan-inference-20261013/`, and
`/tmp/emet-depth-isolated-views-20261013.jsonl`. The render audit for q14 has
worst RGB MAE 0.00372/255 and final MAE 0.00055/255. Figures named
`comparison.jpg` show the exact selected evidence per policy.
