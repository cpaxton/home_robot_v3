# Shared camera-view selection

## Repaired evidence integration — 2026-10-15

Final owner-default EQA gate at `63016107` completed: 7/12 versus main's 9/12.
Do not promote it based on the earlier experiment's 26/36 score. The repaired
owner head did not yet include that experiment's approach buffer and spatial
labels. Reconcile those already-tested opt-in features with the reviewed map
scope/current-capture contracts before running final evidence-mode acceptance.
The integration retains explicit map epochs, actual capture timestamps, protected
pins, post-budget basis validation, and clears prior-epoch approach history.
Frames predating the new scoped map history are excluded, including before the
first assessment in that epoch. No navigation reversion or task-specific answer
rules are added. Image selection and labels remain opt-in; defaults are unchanged.

Room controls with YOLOE completed without infrastructure errors after resource
paths were corrected, but neither main nor candidate passed S0/RoboCasa/Molmo.
Eight S0 views did expose the red cylinder to Qwen; grounding still failed with
zero YOLOE detections. Existing VLM/depth-candidate/SAM provider diagnostic
`20261015_211654_2470ed` is running on the frozen owner integration. Keep these
provider/coverage failures separate from evidence-contract unit validation.

## Review repairs, 2026-10-15

Map reset now mints an observation scope, recorded by both voxel ingestion paths
in each retained frame. The adapter rejects earlier scopes even when the clock
domain and timestamps are recent. Clock domain is no longer the sole identity.
Captured views carry acquisition timing and map scope; assessment passes its
actual retained capture to selection. A timed capture absent from the voxel
history can anchor the packet, so pre-turn age is relative to the assessed
capture rather than the last map frame. Missing acquisition metadata still
causes an explicit fallback. Pre-turn evidence predating the current scoped
history is rejected conservatively.

Current/pre-turn pins survive candidate truncation (image count remains bounded),
and basis compatibility is checked on the retained pool, so a budget-discarded
candidate cannot break assessment. 87 focused selector/history/capture/OVMM
routing/real voxel ingestion tests pass, including reused-map and post-yaw
unmapped-capture regressions. This is contract validation, not a new live score.

The broader experiment branch's matched 12-question/3-seed pilot completed:
candidate 26/36, main 25/36; 144 versus 171 decision rounds. These modest
development differences do not establish a meaningful accuracy improvement or
validate this repaired owner head. The value under review is robust, auditable
shared tooling. Small Molmo/RoboCasa OVMM and final integrated-head acceptance
remain open. Merged main includes native multi-object TAMP tools (#212); preserve
that common tool interface rather than introducing a separate planner here.

Reviewed integration `test/reviewed-shared-agent` now includes main `801dce82`
(native multi-object TAMP and capture timing), #182's mode/status repair, and
#184's bounded recovery repair. Resolve base-pose query overlap using main's
shared `check_base_poses` helper; retain one sweep validation with the shared
rejection status. Combined evidence, routing, floor recovery, real voxel ingest,
and TAMP bridge/native/base-query/gate checks: **392 passed**, two dependency
deprecation warnings. No new model calls or separate TAMP interface were added.

`scripts/run_shared_room_pairs.py` freezes main and candidate revisions and
runs S0 Stretch, RoboCasa Stretch, and MolmoSpaces rby1 serially, seed 0. Both
arms use local Qwen3-VL int4, the same existing YOLOE grounding provider,
physical navigation, no teleportation/cache, four scan headings, six agentic
rounds and three navigation decisions. This is default-provider find
non-regression, not a claim to test new SAM providers or manipulation. Record
commands, configuration/registry/lock/driver hashes, original simulator logs,
and boolean object/receptacle outcomes. Missing/error results fail the gate;
task failure also fails acceptance even when the process exits zero.

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

### Context/detail quality ablation

`quality_coverage` is a separate experimental policy; binary weighted coverage
and the old pose selector are retained unchanged. Its objective is
`sum(w[c] * max(q(view,c) for view in selected) for c in cells)`.
Repeated sightings do not add credit, but an observation with higher pixel
support can improve a cell's best quality. `ViewFrame.cell_quality` is explicit,
immutable, bounded to [0,1] and must describe exactly its visible cells. Setting
every quality to one reproduces binary weighted coverage (unit tested).

The tested proxy is `0.5 + 0.5 * min(pixel_support / 64, 1)` for visible cells.
The context floor preserves credit for wide views rather than selecting only
close-ups. Pixel support is estimated from valid depth samples times stride
squared at a fixed RGB resolution. The floor, reference support and Gaussian
width were not fitted to answer labels. This is **not** calibrated perceptual
confidence: vertical surfaces share XY cells, repeated pixels within one cell
can depict different heights, and sufficient pixel support does not establish
semantic recognition. The generic selector accepts other quality estimators
without changing the objective or the legacy policies.

The offline audit's `--quality-coverage` compares old pose, binary weighted,
quality weighted and current-only on identical paired RGB-D frames, prompts,
model, seeds and a four-image maximum. This isolates selection; it is **not**
an end-to-end comparison against old main, whose acquisition/retention differs.
Job `20261013_160824_cf2553`; artifacts under
`/tmp/emet-context-detail-inference-20261013/`. Live defaults remain unchanged.

The first 36 calls completed: all three selectors answer q56 correctly 3/3;
pose answers q12 correctly 3/3 while binary and quality coverage answer one
3/3. On q14 pose abstains 3/3; both coverage policies choose the wrong TV
relation 3/3. At the 64-pixel reference, 95–99% of weighted cells in key context
views saturate, and the quality policy selects identical packets to binary
coverage for q12/q14. This setting does not test a meaningful detail trade-off
on those cases, and is not a positive result.

One stronger, fixed 1024-pixel reference was queued based on that feature
saturation, not a search over answer accuracy. The context floor stays 0.5;
all other settings stay fixed. It changes q12 to frames 78/57/63/40 and q56 to
78/50/59/55; q14 stays 47/2/22/40. Old pose and binary coverage remain controls.
Job `20261013_161112_bbe6c8`, artifacts
`/tmp/emet-context-detail1024-inference-20261013/`.

The second 36 calls completed with the following answers (three repeated
inferences per packet, not independent episodes):

| Diagnostic | Old pose | Binary coverage | Quality coverage, 1024 pixels |
| --- | --- | --- | --- |
| q56 bookshelf room | Correct 3/3 | Correct 3/3 | Correct 3/3 |
| q12 bedside-table count | Correct 3/3 | Incorrect 3/3 | Correct 3/3 |
| q14 fan location | Abstain 3/3 | Incorrect 3/3 | Incorrect 3/3 |

Current-only answers q12 incorrectly and abstains on q56/q14 in all repeats.
Manual inspection of the comparison sheets confirms the new q12 packet
contains both a wide bedroom view and a closer supporting view showing the
tables; q56 retains bookshelf/curtain room context. q14's packet is unchanged.
The quality objective recovers the count diagnostic but does not outperform
the old selector overall: its wrong fan answer is worse than abstention.
This supports testing context plus detail, not declaring performance restored.
Do not tune further against these three answer labels. Freeze this candidate
for additional unseen frozen episodes before any live default change.

Validation: 80 focused selector/history/captured-view/assessment tests pass,
including legacy pose invariance when quality metadata is supplied. Both
experimental policies remain opt-in; main and live defaults are unchanged.

## Staged live hill-climb (2026-10-13)

Implemented opt-in RGB-D history integration: retained depth/K/optical pose
provide cached surface support; the investigated hypothesis is explicitly
associated with the arrival capture so camera XYZ cannot become the target.
Coverage uses fixed 0.2 m cells, sigma 1.5 m, range 4 m, stride 4, quality
reference 1024 and context credit 0.5. Missing target/registered geometry emits
`view_selection_fallback` and uses pose. Acquisition failures retain the
existing unavailable path. World scope resets with the retained history.

`scripts/run_view_selection_pairs.py` runs fresh processes serially, alternating
arm order, recording source/config fingerprints, exact commands, elapsed time,
metrics and uniquely named episode bundles. Invoke only under CPU-safe,
GPU-exclusive `emet jobs`. It requires clean frozen revisions and pins local
Qwen3-VL-8B int4; one infrastructure retry is retained, wrong answers are not
retried. Defaults are the four-question diagnostic, seed 0, four arms:

- Historical #181: `dc106273f4f87cc2cb3a911aa6f99dac7adb2c25`.
- Current harness unchanged evidence path.
- Current harness pose selection.
- Current harness context/detail quality selection.

Next gates (not completed by unit tests or frozen-image answers):

1. Audit q6/q12/q14/q56 live packets, policy activation/fallbacks and first
   divergent actions. Separate failure to acquire, retain, select, deliver,
   reason, or reconcile assessment/final answers.
2. Historical/pose/quality on dev IDs 2,6,12,14,15,16,25,28,31,56,65,68;
   seeds 0,1,2, with seed 0 first. Require positive total paired accuracy gain
   over pose, gains in at least two seeds, no extra wrong commitments, and
   accuracy at least the freshly rerun historical system.
3. Reserve 8,17,21,29,38,49,79,88 (seeds 0,1) for confirmation of that winner;
   these are not globally unseen. No promotion after a confirmation loss.
4. Pair pose/candidate on RBY1 table S0, RoboCasa S1 and Molmo iTHOR index 0,
   seeds 0,1, through shared agentic find/current-view grounding. No new unsafe
   grounding or net task-success loss; zero-success controls are not acceptance.

Median runtime must remain within 10% of pose. Report per-question paired
wins/losses, seed variability, abstentions, wrong commitments, movement and
infrastructure errors. Repeated seeds are not independent new questions.
Keep live defaults unchanged; no Habitat OVMM/full sweep/hardware expansion.
If quality loses, retain pose; if pose loses to history, diagnose the harness
rather than continuing to tune image-selection thresholds.

### Completed live attribution: no promotion

Job `20261013_200028_8e8cbc` completed all 16 episodes, no retries or
infrastructure failures. Current source `0206f3e5`; historical source
`dc106273`. Results and frozen command/config manifests are in
`/tmp/emet-view-hillclimb-eqa4-20261013/`; episode bundles are under
`~/.cache/habitat_eqa/episodes/emet-view-hillclimb-eqa4-20261013/`.

| Arm | q6 | q12 | q14 | q56 | Correct | Median seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Historical #181 | T | T | T | T | 4/4 | 41.51 |
| Current unchanged | T | F | F | F | 1/4 | 35.59 |
| Current pose | T | F | F | T | 2/4 | 32.70 |
| Current quality | T | F | F | T | 2/4 | 34.15 |

Historical q6 is `uniform_prior` (0.25 confidence); historical q12 is
`mcq_debias` after the assessor abstained. These are not four verified visual
successes. Both selector arms recover q56, with the bookshelf and blue curtains
visibly together in the saved context image. The quality arm actually used
quality selection on q12/q14/q56; q6 was assessed after frontier exploration
without a localized target, so its explicit pose fallback is appropriate and
must not count as quality-policy acceptance.

Manual inspection and retained-history audit establish:

- q12 has 14 mapped observations: 13 at spawn, then one at arrival. Rendered
  approach frames exist in the episode video/frame log but were not candidates
  in the live history. The submitted packet has near-identical current and
  pre-turn closeups plus hallway/living-room context. The offline wider bedroom
  packet cannot be reproduced merely by rescoring these retained frames.
- q14 historical/current arrival XY matches. Current final yaw differs by
  approximately 30 degrees and removes the sofa from the primary view. Qwen
  still chooses the TV relation when history supplies wider room context.
  This is not proof that a different coverage weight will repair reasoning.
- Some per-assessment `query_png` trace paths do not exist after export. Saved
  final `eqa_decisions` packets and rendered frames supported the visual audit,
  but per-call artifact completeness needs repair before claiming that gate.

Decision: the quality objective does not improve over pose here; defaults stay
unchanged. Do not launch the 12-question/3-seed, reserved confirmation, or OVMM
promotion stages for this candidate. Regression recovery is **not achieved**.
The next isolated experiment should address evidence retention during approach,
using a bounded shared captured-view mechanism without creating extra object
instances, with pose held fixed. Audit current/pre-turn redundancy separately;
do not combine acquisition, pinning, prompt and answer-precedence changes.
Do not revert safe object-facing navigation just to recover lucky fallback
answers. Keep historical and current pose controls for any subsequent repair.

Validation: 84 focused selector/history/assessment/captured-view tests passed;
commit hooks including ruff and mypy passed. No main edits, pushes, model
changes, hardware motion, or parallel GPU experiments were performed.

### Approach-retention repair and diagnostic

Source `27d9a6a7`, job `20261013_221001_6a7747`, artifacts
`/tmp/emet-approach-retention-eqa4-20261013/`. All eight paired episodes
completed without retries or infrastructure failures. Pose and pose+approach
both score **2/4** (q6/q56 correct; q12/q14 wrong). Median episode times are
32.23 s and 32.72 s respectively. This is a contract repair, not an accuracy win.

`EMET_EQA_RETAIN_APPROACH_VIEWS=1` enables a bounded 32-frame, query-local
RGB-D evidence buffer. It samples already acquired images after 0.25 m camera
translation or 15 degrees optical-axis change. Habitat feeds action renders;
the shared ZMQ waypoint path feeds waypoint-arrival observations (not yet
continuous in-motion transport). No maps or object instances are inserted by
this buffer. Missing acquisition metadata is not fabricated. Captures own their
arrays; snapshot count is bounded and the buffer is replaced per query.

The paired driver has a `pose_approach` arm; selector, pins, image budget,
navigation, prompts and answer precedence remain unchanged. Exact ordered
packets are saved under each episode's `selected_views/NNNN/`, with PNGs,
geometry and pixel hashes. These files were verified after episode completion.
99 focused tests pass, including ownership/bounds/motion sampling, missing
timestamps, history integration, Habitat capture, serial waypoint aborts and
packet export. Commit hooks pass.

Visual audit: q12 now selects approach views showing **both** bedside tables.
Qwen nevertheless says one and claims no second table appears in history.
The previous missing-candidate issue is fixed for that example; answer
correctness is not. q14's pose selector retains the same misleading-to-Qwen
context despite additional candidates. Do not adjust answer labels or invent
task-specific count/relation rules to force these diagnostics to pass.

Next: broaden to the fixed 12-question dev set, first seed 0, with historical,
pose and pose_approach arms. This is an attribution pilot for a retention repair,
not promotion of the failed quality objective. Keep three-seed confirmation
and small OVMM pending; no merge/default change follows from a single seed.

### Twelve-question result and frozen-packet attribution (2026-10-14)

Seed 0 completed 36/36 episodes at `241a5248`, no infrastructure failures.
Historical: 8/12 (5/5 VLM-suggested, 3/7 fallback); pose: 7/12 (5/7
VLM-suggested, 2/5 fallback); pose+approach: 6/12 (5/7 VLM-suggested,
1/5 fallback). Fallback successes count toward accuracy. No recovery claim.
q12 changes from historical budget fallback to early wrong commitment; q14
already stops after one round in both versions, so its error is not earlier
stopping. A single seed cannot establish the retention effect.

Next diagnostics keep model and budgets fixed:

1. `scripts/assess_saved_view_packets.py`: export original, uncropped packets
   with question, image roles, age, original image links, and episode-level
   parsed answer. HTML is portable with adjacent PNGs and print-friendly for
   documentation; `cases.json` records source and image hashes. Do not mistake
   an intermediate packet for final submission. Compare historical-style
   unlabeled single image, labeled single, labeled pair, and full labeled packet
   from identical captures, three repeats. Inventory/target hints are omitted
   uniformly: this isolates image/presentation effects, not exact live replay.
   Model responses are saved separately, including abstentions and reasons.
2. Paired live driver arms `stop_single` / `stop_corroborated`: both use pose
   plus approach retention. Only `EMET_EQA_AGENTIC_SINGLE_VIEW_CONFIRM` changes.
   Corroborated is NOT strict independent two-view agreement: phrase evidence
   can still confirm, and different observation IDs need not be distinct poses.
   Keep confirmation and no-early-unverified guards enabled. Run the same 12
   questions, seed 0; inspect confirmation reasons alongside accuracy and cost.

No navigation-distance changes in these comparisons. The q12 primary crops
out counting context, q14 loses the sofa, and q56 wider context shows bookshelf
and blue curtains together. Investigate a wider viewing position separately;
do not relabel an earlier image as the robot's current physical view, weaken
manipulation visibility checks, or tune on answer labels.

### Completed packet/stopping comparison and provenance follow-up

At `52f1bcd8`, both jobs completed: 216 offline assessments (18 saved packets,
four presentation arms, three repeats) and 24 live episodes. Single-view
confirmation scored 7/12 versus corroborated 6/12; total decision rounds were
47 versus 63. q12 remained wrong after four corroborated rounds; q14 confirmed
immediately via phrase corroboration. Do not promote this stopping change.

Fixed captures: q12 answered One in all arms/repeats. q14 single-image arms
abstained; pair/full answered the wrong TV relation. q56 single-image arms
abstained; pair/full correctly selected blue curtains. q31 single-image arms
were correct, pair/full wrong in this offline diagnostic (live runs were
correct). Inventory and target hints were omitted uniformly, so these are
presentation diagnostics, not exact reproductions of live decisions.

Historical q12's latest correct result was `mcq_debias` after eight rounds,
following an initial visual abstention. It was not a successful early visual
count. Earlier offline wider-view successes are different captured evidence.

Follow-up adds capture-sequence ranks (earliest=1 within packet), world camera
XYZ in meters, and world optical-forward direction. Display order and capture
order remain distinct; simultaneous timestamps share a sequence rank, unknown
poses remain unknown. No object locations are inferred. The shared formatter
serves live selected-view prompts and frozen replay; the audit manifest saves
the actual prompt labels. Compare `labeled_full` / `spatial_full` on identical
saved pixels with `--provenance-only`, three repeats. No camera motion, image
selection, stopping-policy, or task-specific prompt changes in this comparison.

The provenance replay completed 108/108 assessments. q12 changed wrong One to
correct Two in all three repeats; q14 wrong TV became abstention in all three;
q56 remained correct. q31 stayed wrong twice and abstained once. q68 had mixed
packet-level abstention changes, including loss of a previously correct answer.
This is not a general no-regression result.

Next live test: `labels_role` versus `labels_spatial`, the same 12 questions,
seed 0, 24 serial episodes. Both use pose+approach and single-view confirmation;
only `EMET_EQA_VIEW_SPATIAL_LABELS` differs. Spatial labels remain opt-in via
`SelectionConfig.spatial_labels` (default false). Score paired accuracy,
confirmation provenance, rounds, and whether deferred answers recover before
budget exhaustion. Freeze code before launch; do not change viewing distance
or stopping alongside the labeling intervention.

## Checkpoint report: live provenance pilot, 2026-10-14

Status: promising candidate, **not merge/default acceptance**. Feature branch
`feat/target-local-view-coverage`; evaluated source `939e5563`. No changes to
main. Spatial labels remain opt-in. The shared harness uses packet-local capture
sequence and camera-world geometry; no task-specific answer rules were added.

### Evidence and reproducibility

Job `20261014_084459_1efb95` completed 24/24 episodes without infrastructure
failures. Paired seed 0, question IDs `2 6 12 14 15 16 25 28 31 56 65 68`.
Both arms use local Qwen3-VL-8B-Instruct int4, pose selection, approach retention,
single-view confirmation, and identical exploration/movement budgets. Only
`EMET_EQA_VIEW_SPATIAL_LABELS` differs. Artifacts and invocation manifests:
`/tmp/emet-provenance-eqa12-s0-20261014/`; images and traces:
`~/.cache/habitat_eqa/episodes/emet-provenance-eqa12-s0-20261014/`.

| Arm | Correct | VLM-suggested correct/total | Fallback correct/total | Decision rounds |
| --- | --- | --- | --- | --- |
| Role-only labels | 6/12 | 5/7 | 1/5 | 47 |
| Sequence + XYZ + direction | 8/12 | 7/9 | 1/3 | 39 |

Paired wins: q12 (One → Two), q25 (wrong fallback → correct visual answer),
q15 (wrong → correct fallback). Paired loss: q68 (correct fallback → wrong
committed answer). q14 remains wrong. This is net +2 on one development seed,
not a statistically established improvement. Fallback successes count; q15's
gain is less diagnostic than q12. No latest small-OVMM acceptance has been run.

### Failure audit

- q14 offline/live packets have **identical PNG SHA-256 hashes for all four
  images**. Their camera poses also match. Live supplies target `standing fan`
  and inventory (`observations_seen=2`, `rounds=0`, `nav_steps=1`, graph labels
  `frontier, leave fan`, tried observation ID); offline omitted target/inventory.
  Relative ages and the pre-turn ID differ between captures. Offline abstention
  therefore cannot be credited as live recovery. Text context/model sampling
  remains to be isolated; camera framing alone does not explain this mismatch.
- q68 initially says No without seeing the ironing board, later abstains, then
  commits Yes at round 6. Its reason places the ironing board in historical
  Image 4 and says inventory/context imply co-location with the sofa. The
  `single_view_present` confirmation accepts that assessment. This shows a
  context/co-location failure, not evidence that additional exploration reliably
  recovered the answer. Do not add a question-specific rejection rule.

### Validation and next gates

61 focused selector/history/assessment/runner tests pass; commit hooks pass.
Gallery exporter keeps original PNGs and separates assessment packets from
episode-level final answers. The earlier gallery is at
`/tmp/emet-eqa-packet-gallery-20261014/index.html`; the exporter and manifests,
not machine-local image files, are version-controlled for reuse in docs/paper.

1. Freeze this checkpoint and run paired seeds 1 and 2 with the same 12 questions
   and unchanged two arms, serial GPU/CPU-safe execution. Aggregate paired
   differences across seeds, provenance, rounds and failures; no threshold tuning.
2. Reproduce q14 with exact live prompt plus identical saved pixels, then remove
   target/inventory separately if needed. Inspect q68's final historical/current
   packet before changing any evidence-consistency contract.
3. If gains survive repeats, run small OVMM non-regression before promotion.
   Wider primary viewpoints remain a separate intervention. Do not combine
   camera-distance, stopping, model, and label changes in one claimed ablation.

### Three-seed replication and text attribution

Jobs `20261014_084459_1efb95` and `20261014_094809_553991` together completed
72/72 paired live episodes. All are development questions, repeated across
seeds; do not treat them as 72 independent scenes.

| Seed | Role labels | Spatial labels |
| --- | --- | --- |
| 0 | 6/12 | 8/12 |
| 1 | 6/12 | 8/12 |
| 2 | 9/12 | 9/12 |
| Total | 21/36 | 25/36 |

q12's direct visual recovery repeats in seeds 0 and 1; both arms are correct
in seed 2. Across paired question/seed cases there are five wins and one loss.
Two wins change fallback answers (q15 twice); q12 twice and q25 once are
visual-answer gains. The loss is
q68 seed 0. Aggregate VLM-suggested answers: 15/20 correct for role labels,
18/22 for spatial labels; these conditional buckets are not randomized subsets.

Text attribution retry `20261014_103218_08ba56` completed 27 calls after the
initial runner failed before inference (wrong multi-image API). Same saved
pixels and exact live prompts, separately removing target/inventory:

- q14: exact live yielded one wrong TV answer and two abstentions; removing
  inventory yielded three abstentions; removing target yielded three wrong TV
  answers. Removing both or replaying the saved offline prompt abstained three
  times. Inventory sensitivity is supported; a specific field is not isolated.
- q68: exact live and inventory-removed each yielded two wrong Yes answers and
  one **truncated, invalid JSON response**, not an abstention. Removing target,
  with or without inventory, yielded No/Yes/No. Removing inventory does not fix
  this case. The current image is a wall/door; the model's present=true conflicts
  with current-view evidence. No safe rule based only on prose is established.

Follow-up `text-fields-20261014` isolates graph-label text from status counters
on identical pixels and retains the original controls. No answer-policy or
navigation changes are bundled into the labeling candidate.

### Draft stack landing plan (GitHub inventory checked 2026-10-14)

| PR | Dependency / scope | Remaining gate before ready |
| --- | --- | --- |
| #181 | main; optional grounding providers | Exact-head provider audit and default EQA/small-OVMM checks |
| #182 | #181; measured navigation | Exact-head route, physical wheel-drive smoke, EQA/small-OVMM |
| #184 | #182; inspection/floor recovery | Same-waypoint recovery, missing-anchor and precision approach checks |
| #199 | #184; assessed/context evidence | Carry shared-contract repairs to owner; paired EQA and current-view localization |
| #202 | #199; generic view selector | Selector-only review; integrate provenance/retention as separately reviewable slices after acceptance |
| #185 | #184; acceptance tooling | Refresh frozen revisions and results after owner slices land |
| #198 | #184; optional measured scanning | Independent corrected scan ablation; do not block the core stack on it |
| #183 | legacy experiment branch | Reconcile disposition against extracted owners, then retire with links |

#213 native TAMP and #214 reactive manipulation are separate work streams.
Do not change or merge them as part of this evidence-selection checkpoint.

Order: audit current main and ownership; land #181 → #182 → #184 → #199 →
#202 with exact-head review/tests at each step. Refresh acceptance #185 as
parents move. Keep optional scan #198 separate. This branch includes 43 changed
files beyond the local #202 tracking ref, including shared camera timing,
navigation and task-routing repairs; do not merge the whole experimental branch
as if it were only a labeling patch. Attribute those repairs to their owner PRs
and retain lightweight provenance/approach-buffer/visual-audit slices above
#202. Review current remote heads before cherry-picking or retargeting.

Promotion requires the 3-seed EQA comparison plus small Molmo/RoboCasa OVMM
non-regression and review of the remaining current-view contract failures.
The 25/36 result is promising but not full stack acceptance. No PR was marked
ready or merged during this inventory; descriptions still contain older gates
and must be rewritten against their final heads when those gates are cleared.

The field-level diagnostic `20261014_105129_189c99` completed 39 calls. q14:
removing status counters/tried IDs yielded three abstentions; removing only
graph labels yielded three wrong TV answers. Exact-live control repeated
wrong/abstain/abstain. Thus progress/status text, rather than graph-label text
alone, changes answerability in this example; this remains a three-repeat
development diagnostic. A status-free assessment prompt is a reasonable next
isolated ablation. Preserve planning progress for the high-level agent; whether
it belongs in a visual evidence judgment is a separate contract question.

q68: removing either graph labels or status still yielded three wrong Yes
answers. Removing target yielded No/Yes/No. Exact-live and inventory-removed
each produced two wrong Yes plus one truncated JSON. Text cleanup does not
resolve current-image presence/co-location here. Keep the failure visible in
acceptance and assess wider primary views separately before adding new checks.

### Repair ownership and simple prompt ablation

The 25/36 versus 21/36 comparison is between labeling arms on the integrated
stack. It is not a paired comparison to current main. A fresh matched main gate
is required before stating that the new stack is ahead/behind main.

Assign repairs by changed behavior, retaining original commits for review:

| Owner | Repair commits / scope | Verification |
| --- | --- | --- |
| #182 measured navigation | `68e263ff` waypoint alignment as progress; `da022908` navigation-map fixtures; `35d2a7f4` stowed Stretch collision envelope | Arrival-monitor and footprint tests, physical route smoke, small OVMM |
| #184 inspection/recovery | `8afd343f` preserve task while routing frontier hints | Task-routing tests and frontier investigation smoke |
| #199 assessed evidence | `ac3a3abb` historical context cannot authorize current localization; `b6fdda7f` retain captured arrival view after grounding | Assessment and OVMM routing tests; current-view find gate |
| #198 optional coverage scan | `005f0d9a` grid-edge rays; `9df2a382` full-heading distribution (already inherited in #199/#202) | Map-edge and heading tests, optional scan ablation |
| Transport/simulator repair below consumers | `17211f35` acquisition timestamps through Stretch camera IPC/ZMQ | Camera timing and ZMQ observation tests; old-message compatibility |
| #202 selector and follow-up slices | Model-free coverage module; opt-in live adapter; `27d9a6a7` bounded approach captures; `654f33f1` provenance labels; visual audit/paired runner | Selector/history tests, three-seed EQA, small OVMM |
| #199 follow-up if accepted | Existing assessment inventory without planning status | Exact prompt ablation, paired live EQA; preserve planner status |

These are ownership assignments, not claims that owner remote heads already
contain every repair. Before moving commits, reconcile each with current main
and destination head, remove duplicates, and preserve separate evidence. The
timestamp repair is cross-cutting and should not be buried in selector review.
Approach capture touches shared waypoint execution but its purpose is evidence
retention, so review it as a separate bounded-history slice above contracts.

The prompt experiment adds `include_progress` to the existing inventory builder
and `EMET_EQA_ASSESS_PROGRESS=0` as an evaluation override. It omits observation
counts, navigation/round counters, and tried IDs while retaining graph labels.
There is no extra model call, confirmation stage, retry, or new assessment
module. Planning still receives its usual state. The original default remains
enabled. Runner arms `spatial_progress` and `spatial_visual` otherwise match;
run 12 questions × two arms, seed 0. 63 focused tests pass. Do not promote this
text change merely because it causes abstention in q14.

The live progress-text ablation completed 24/24 episodes: existing inventory
9/12 with 40 total rounds; progress-free inventory 8/12 with 48 rounds. q25
lost its correct visual answer; q14 still ended wrong. Keep the original prompt
default. The offline abstention effect does not justify promoting removal.

Next matched-main gate freezes main `b2b71216` and the current candidate, uses
the same 12 questions, seeds 0/1/2, local Qwen3-VL-8B int4, and the same CLI
budgets. Main uses its native implementation; the candidate uses pose+approach
and spatial labels with the original progress inventory. Record config hashes
for both. This compares complete systems rather than attributing every default
or implementation difference to provenance. 72 serial episodes; inspect paired
accuracy, evidence provenance, runtime, and infrastructure failures.
