# Graph-object fusion: how detections become scene-graph nodes

**GraphObjectFusion** decides when a per-frame detection (YoloE instance, HM3D
semantics, or VLM label) becomes — or merges into — a node in the scene graph that
the EQA prompt (SCENE_GRAPH) and count/FIND recall read from. There is one shared
graph; this policy is what keeps it from silently becoming two.

The config lives in the `graph_object_fusion` block
(`src/emet/config/agents/default_graph_object_fusion.yaml`, or any file pointed at by
`EMET_GRAPH_FUSION_CONFIG`). See [dynagraph.md](dynagraph.md) for the graph itself and
[lazy_graph.md](lazy_graph.md) for the "close-look-only" ingest alternative.

## Why it matters (the singleton flood)

Feeding every YoloE instance of every frame in as a first-class node triples-to-
quadruples the object-node count per episode (q8: 19 → 241 object nodes at an
unchanged VLM budget) and drives 40–53% of nodes to *singleton* (support = 1). A
sweep of the 3D-bounds IoU threshold showed the flood is **not threshold-dependent**
(0.0 → 0.6: nodes 213–233, singletons ~0.5) — it is structural: YoloE class strings
drift frame to frame, mask-median centroids drift past the spatial gate, and
partial-view 3D bounds rarely overlap. The policy below is therefore *ordered
evidence gates* plus admission, keep, and growth controls, not one knob.

## Policy blocks

```yaml
graph_object_fusion:
  enabled: true

  # Master switch: when false, YoloE/instance detections never become scene-graph
  # entries (count/FIND recall only) — the pre-instance-graph behavior of the
  # published 49.6% full-113 baseline.
  use_instance_nodes: true

  gates:
    identity:                      # exact persistent track/instance id
      on: true
    bounds:                        # 3D-bounds overlap (duplicate views of one object)
      on: true
      iou_floor: 0.08              # hard floor when both sides carry bounds_3d
      iou_merge_min: 0.3           # merge on overlap even if spatial/centroid drift (0=off)
    embedding:                     # appearance similarity (SigLIP crop embeddings)
      on: true
      min_cosine: 0.62
      blend_alpha: 0.35
      use_siglip_crops: true       # encode each instance bbox crop with shared SigLIP
      appearance_merge_min_cosine: 0.9  # merge across label drift when this similar
    spatial:                       # centroid proximity
      on: true
      xy_m: 0.42
      centroid_3d_m: 0.55
      fallback_xy_m: 0.45          # nearest-node radius when strict gates fail (0=off)

  labels:
    require_match: false
    require_match_for_instances: true   # countable instances need compatible labels
    synonyms: [["cab", "cabinet", "kitchen cabinet"]]  # extra groups beyond built-ins
    incompatible: [["person", "lamp"]]   # pairs that must never merge

  keep:
    prefer_support: true           # higher support_count survives (else newer)
    union_labels: true
    union_bounds: true             # grow bounds_3d to the union on merge
    blend_embedding: true
    update_xyz: true               # blend the anchor via position update (else keep)

  admission:
    instance_min_confidence: 0.12
    instance_min_mask_points: 25
    max_candidates_per_frame: 64
    match_xy_m: 0.55

  growth:
    max_object_nodes: 0            # 0 = unlimited; hard cap per episode (flood guard)
    temporal_window_steps: 0       # 0 = off; refuse merges into stale nodes
```

Legacy flat keys (`spatial_merge_xy_m`, `bounds_3d_iou_merge_min`,
`fallback_spatial_merge_xy_m`, `embedding_min_cosine`, `require_label_match*`,
`max_candidates`, `instance_min_*`, `match_xy_m`) still parse and remain readable/
writable, mapping onto the nested blocks above.

## Appearance merging via SigLIP

Instance detections carry no embeddings by default, so the embedding gate is a
no-op and label drift defeats merging. With `gates.embedding.use_siglip_crops: true`,
each instance bbox crop is encoded with the shared SigLIP encoder at ingest. When a
candidate and a *nearby* node have cosine similarity
`>= appearance_merge_min_cosine`, they merge even if the label gate fails — while the
spatial gate still enforces proximity, so identical-but-distinct instances sitting
apart stay separate (preserving count recall).

## Lazy graph (alternative ingest)

`LazyGraphController` never streams YoloE detections into the scene graph during
passive mapping; it records viewpoint stamps and runs Qwen label-extract +
`add_observation` only on nav-arrival close looks. See [lazy_graph.md](lazy_graph.md).

## Eval / paper use

- **HM-EQA method:** `emet-habitat run-batch --method lazy_graph` selects the
  close-look-only row (`harness.habitat_eqa.lazy_graph` in
  `configs/benchmarks/dynagraph.yaml`). `static_graph` / `dynagraph` are unchanged.
- **Strategy configs:** `configs/benchmarks/fusion_strategy_b.yaml` (instance nodes +
  deep merge policy) and `fusion_strategy_c.yaml` (`use_instance_nodes: false`) are
  consumed by `scripts/eval_merge_strategies.sh` via `EMET_GRAPH_FUSION_CONFIG`.
- **Env override:** `EMET_GRAPH_FUSION_CONFIG=<path>` swaps the default fusion YAML
  for a run (used by the IoU sweep `scripts/sweep_fusion_iou.sh` and the strategy
  comparison).
- **Pinned row guard:** `test_habitat_eqa_dynagraph_paper_row_knobs_pinned` freezes
  the published row's knobs so a silent ingest change can never re-baseline it.
