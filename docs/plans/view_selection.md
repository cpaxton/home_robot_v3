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
