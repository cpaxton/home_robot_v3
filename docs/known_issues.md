# Known issues

Tracked bugs and investigation notes.

**See also:** [cli.md](cli.md) · [zmq_obs.md](zmq_obs.md) · [dynagraph.md](dynagraph.md) · [graph_eqa.md](graph_eqa.md) · [robots/innate_mars.md](robots/innate_mars.md) · [robots/innate_mars_hardware.md](robots/innate_mars_hardware.md) · [environment_variables.md](environment_variables.md)

---

## Dynagraph graph node explosion on stationary hardware stream

**Status:** Mitigated (2026-06) · **Seen:** 2026-06 · **Backends:** `dynagraph` (not `voxel_only`)

### Symptoms

`emet stream --backend dynagraph` on a **non-moving** real robot (e.g. Innate Mars / Herman) grows the graph without bound:

- Example: **94 → 518 graph nodes** in ~17 mapping steps while the base does not move.
- Voxel **explored cell count** also swings (e.g. 398 → 473 → 300) step-to-step.
- Terminal shows repeated **DA3** forward passes per update (hardware has no ZMQ depth → [`dynav_innate_mars.yaml`](../src/emet/config/dynav_innate_mars.yaml) / `depth_source: auto`; see [innate_mars hardware](robots/innate_mars_hardware.md)).

Typical log pattern:

```
step 1: 1 voxel obs, 398 explored cells, 94 graph nodes
[INFO ] Processed Images Done …
step 3: … 171 graph nodes
…
step 17: … 518 graph nodes
```

### Repro (hardware smoke)

```bash
emet mars start --connection herman
emet stream --connection herman --backend dynagraph --headless
# Robot stationary; watch graph node count in periodic status lines
```

Sim with sensor depth may **not** show the same severity (dedup is more stable when depth is consistent).

### Fix (2026-06)

GraphObjectFusion now has a **fallback merge tier** (`fallback_spatial_merge_xy_m`, default **0.45 m**, aligned with `dynagraph_merge_xy_m`). When strict spatial/embedding/bounds gates fail, detections merge into the nearest object node within that XY radius. Innate Mars stream/dynav loads relaxed gates from [`graph_object_fusion_innate_mars.yaml`](../src/emet/config/agents/graph_object_fusion_innate_mars.yaml) via [`dynav_innate_mars.yaml`](../src/emet/config/dynav_innate_mars.yaml).

**Offline regression:** [`src/test/memory/test_graph_dedup_offline.py`](../src/test/memory/test_graph_dedup_offline.py) replays noisy calibration JSONL fixtures ([`calibration_frames_stationary_noisy.jsonl`](../src/test/fixtures/calibration_frames_stationary_noisy.jsonl), [`calibration_frames_long_explore_noisy.jsonl`](../src/test/fixtures/calibration_frames_long_explore_noisy.jsonl)). Regenerate fixtures with [`scripts/build_dedup_calibration_fixtures.py`](../scripts/build_dedup_calibration_fixtures.py).

Hardware validation on Herman is still recommended after deploy; stationary-stream throttle (base pose delta) remains a follow-up.

### Why dedup is failing (hypothesis)

Several merge paths exist; on stream + hardware they appear **too weak** for noisy stationary observations:

| Mechanism | Config / code | Limitation on stationary IRL |
|-----------|---------------|------------------------------|
| **GraphObjectFusion** | [`graph_object_fusion`](../src/emet/config/agents/default_graph_object_fusion.yaml) in dynav (enabled on stream via [stream_agent_factory](../src/emet/app/stream_agent_factory.py)) | Merges when XY ≤ `spatial_merge_xy_m` (0.42 m), 3D centroid ≤ `min_centroid_dist_m` (0.55 m), bounds IoU, embedding cosine ≥ 0.62. **DA3 depth jitter** and pose/camera noise can push repeated views of the same object outside these gates → **new node every step**. See [fusion.py](../src/emet/memory/graph_eqa/graph_object_fusion/fusion.py) and [dynagraph.md § Configuration keys](dynagraph.md#configuration-keys). |
| **Label-based pre-dedup** | `graph_instance_dedup_xy_m` (default **0.4 m**; [graph_eqa.md](graph_eqa.md)) | `_graph_dedup_skips` only skips when the **same primary label** is within XY radius. VLM / detector **label drift** (`"mug"` vs `"coffee cup"`) bypasses this. See [controller_graph_eqa.py](../src/emet/controller/controller_graph_eqa.py). |
| **Dynagraph spatial merge** | `dynagraph_merge_xy_m` (default **0.45 m** on stream; [dynagraph.md](dynagraph.md)) | `GraphEQAMemory.add_observation` merges same primary label within XY — but **`spatial_merge_m` is cleared to 0** when GraphObjectFusion is enabled ([setup.py](../src/emet/memory/graph_eqa/graph_object_fusion/setup.py)). |
| **Staleness prune** | `dynagraph_staleness_horizon` (default **256**) | `maintain()` does not drop nodes until they are stale for hundreds of steps — fine for explore loops, **not** for short stationary streams. |

Additional contributors:

- **Multiple detections per frame** (YoloE / OwlSam) → several fusion candidates per `update()`.
- **Viewpoint / frontier nodes** (Dynagraph enables frontier coverage features not used in baseline GraphEQA).
- **Micro head motion** or timestamp/pose noise on the bridge even when the base is still.

### Workarounds (today)

| Goal | Command / config |
|------|------------------|
| Voxel map only, no graph | `emet stream --backend voxel_only` |
| Full dynamem, no graph | `emet stream --backend dynamem` |
| Short smoke with bounded graph steps | `--max-steps 3` (still may add nodes; does not fix dedup) |
| Quieter terminal | default `stream.da3_log_level: WARN` in dynav ([zmq_obs.md](zmq_obs.md), [environment_variables.md](environment_variables.md)) |

Do **not** use `emet run dynagraph` on hardware for stationary mapping — it may rotate the head / explore unless `-N` and flags are carefully set (see [experiments/innate_mars.md](experiments/innate_mars.md) and [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md)).

### Investigation directions

1. **Stationary-stream profile** — skip or throttle graph perception when base pose delta &lt; ε for N steps.
2. **Tighter fusion for DA3** — [`graph_object_fusion_innate_mars.yaml`](../src/emet/config/agents/graph_object_fusion_innate_mars.yaml) tuning or pose-stabilized unprojection before fusion.
3. **Re-enable or unify merge** — ~~reconcile GraphObjectFusion with `dynagraph_merge_xy_m`~~ **Done:** fallback tier + innate_mars YAML (see Fix above).
4. **Aggressive staleness** for stream-only sessions (lower horizon when `emet stream` not explore-loop).
5. **Regression test** — ~~sim GT graph with fixed camera should keep node count flat~~ **Done:** `test_graph_dedup_offline.py` + fixture generator; hardware replay from saved `capture` metadata remains optional.

### Code touchpoints

- `src/emet/memory/graph_eqa/graph_memory.py` — `add_observation`, `merge_object_detection`, `spatial_merge_m`
- `src/emet/memory/graph_eqa/graph_object_fusion/fusion.py` — merge gates
- `src/emet/controller/controller_graph_eqa.py` — `_graph_dedup_skips`
- `src/emet/app/stream_agent_factory.py` — `dynagraph_merge_xy_m` / `graph_object_fusion` defaults for stream
- `src/emet/config/agents/default_graph_object_fusion.yaml`
