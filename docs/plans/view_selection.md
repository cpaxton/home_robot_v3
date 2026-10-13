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
- Pose is a rigid world-from-optical transform, +Z forward, translation in meters.
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
