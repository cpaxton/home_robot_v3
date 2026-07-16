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

Optional **DA3 post-filters** (speckle open on inferred depth; voxel PCD DBSCAN on navigation PCD) can reduce floating ``world/point_cloud`` blobs; they are **off by default** — see [dynav_config.md](dynav_config.md#depth--voxel-post-filters-da3-hardware-opt-in).

### Fix (2026-06)

Three changes address stationary hardware stream growth:

1. **GraphObjectFusion fallback tier** (`fallback_spatial_merge_xy_m`, default **0.45 m**, aligned with `dynagraph_merge_xy_m`). When strict spatial/embedding/bounds gates fail, instance detections merge into the nearest object node within that XY radius.
2. **Innate Mars fusion YAML wired correctly** — `attach_graph_object_fusion` now reads `graph_object_fusion` from `Parameters` / yacs config (controllers no longer pass `None` when parameters is not a plain `dict`). [`dynav_innate_mars.yaml`](../src/emet/config/dynav_innate_mars.yaml) supplies relaxed gates (embedding off, fallback **0.55 m**).
3. **VLM sensor labels through fusion** — Qwen-extracted labels in [`dynamem_graph_hooks.py`](../src/emet/memory/graph_eqa/dynamem_graph_hooks.py) use `apply_detection` instead of `add_observation` with dedup disabled (~8 new nodes/step before).

**Stream status** now reports object / viewpoint / frontier breakdown: `graph 11 obj / 12 vp / 1 fr (24 total)` ([`stream_agent_factory.py`](../src/emet/app/stream_agent_factory.py)).

**Offline regression:** [`src/test/memory/test_graph_dedup_offline.py`](../src/test/memory/test_graph_dedup_offline.py) + [`scripts/build_dedup_calibration_fixtures.py`](../scripts/build_dedup_calibration_fixtures.py). **Hardware diagnostic:** [`scripts/diagnose_stream_graph_growth.py`](../scripts/diagnose_stream_graph_growth.py).

Herman smoke after fix: object nodes plateau ~10–11 after step 1 with default VLM+instance stream (was +8–9 object nodes/step).

### Why dedup is failing (hypothesis)

Several merge paths exist; on stream + hardware they appear **too weak** for noisy stationary observations:

| Mechanism | Config / code | Limitation on stationary IRL |
|-----------|---------------|------------------------------|
| **GraphObjectFusion** | [`graph_object_fusion`](../src/emet/config/agents/default_graph_object_fusion.yaml) in dynav (enabled on stream via [stream_agent_factory](../src/emet/app/stream_agent_factory.py)) | Merges when XY ≤ `spatial_merge_xy_m` (0.42 m), 3D centroid ≤ `min_centroid_dist_m` (0.55 m), bounds IoU, embedding cosine ≥ 0.62. **DA3 depth jitter** and pose/camera noise can push repeated views of the same object outside these gates → **new node every step**. See [fusion.py](../src/emet/memory/graph_eqa/graph_object_fusion/fusion.py) and [dynagraph.md § Configuration keys](dynagraph.md#configuration-keys). |
| **Label-based pre-dedup** | `graph_instance_dedup_xy_m` (default **0.4 m**; [graph_eqa.md](graph_eqa.md)) | `_graph_dedup_skips` uses :func:`~emet.memory.graph_eqa.graph_stats.labels_compatible_for_dedup` (exact / substring / shared tokens / synonym groups) so ``mug`` vs ``coffee cup`` no longer bypasses XY dedup. See [controller_graph_eqa.py](../src/emet/controller/controller_graph_eqa.py). |
| **Dynagraph spatial merge** | `dynagraph_merge_xy_m` (default **0.45 m** on stream; [dynagraph.md](dynagraph.md)) | `GraphEQAMemory.add_observation` merges **compatible** labels within XY — but **`spatial_merge_m` is cleared to 0** when GraphObjectFusion is enabled ([setup.py](../src/emet/memory/graph_eqa/graph_object_fusion/setup.py)); fusion fallback covers that path. |
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
| Short smoke with bounded graph steps | `--max-steps 3` |
| Instance-only graph (no VLM labels) | `emet stream --backend dynagraph --no-sensor-perception` |
| Quieter terminal | default `stream.da3_log_level: WARN` in dynav ([zmq_obs.md](zmq_obs.md), [environment_variables.md](environment_variables.md)) |

Do **not** use `emet run dynagraph` on hardware for stationary mapping — it may rotate the head / explore unless `-N` and flags are carefully set (see [experiments/innate_mars.md](experiments/innate_mars.md) and [dynagraph_robocasa_e2e.md](dynagraph_robocasa_e2e.md)).

### Investigation directions

1. **Stationary-stream profile** — optional throttle when base pose delta &lt; ε (not required after 2026-06 fusion + VLM fixes).
2. **Tighter fusion for DA3** — ~~[`graph_object_fusion_innate_mars.yaml`](../src/emet/config/agents/graph_object_fusion_innate_mars.yaml) tuning~~ **Done:** wired via Parameters + innate_mars dynav block.
3. **Re-enable or unify merge** — ~~reconcile GraphObjectFusion with `dynagraph_merge_xy_m`~~ **Done:** fallback tier + innate_mars YAML (see Fix above).
4. **Aggressive staleness** for stream-only sessions (lower horizon when `emet stream` not explore-loop).
5. **Regression test** — ~~sim GT graph with fixed camera should keep node count flat~~ **Done:** `test_graph_dedup_offline.py` + fixture generator; hardware replay from saved `capture` metadata remains optional.
6. **Kitchen label filter (2026-07)** — Robocasa sessions deny bathroom ScanNet classes at graph attach (`graph_eqa_label_filter` / [`graph_label_filter.py`](../src/emet/memory/graph_eqa/graph_label_filter.py)) so ``bathroom stall`` cannot dominate kitchen graphs.

### Code touchpoints

- `src/emet/memory/graph_eqa/graph_memory.py` — `add_observation`, `merge_object_detection`, `spatial_merge_m`
- `src/emet/memory/graph_eqa/graph_object_fusion/fusion.py` — merge gates
- `src/emet/controller/controller_graph_eqa.py` — `_graph_dedup_skips`
- `src/emet/app/stream_agent_factory.py` — `dynagraph_merge_xy_m` / `graph_object_fusion` defaults for stream
- `src/emet/config/agents/default_graph_object_fusion.yaml`

---

## NVIDIA driver hang / Cursor agent crash during stacked GPU evals

**Status:** Mitigated (2026-06) · **Seen:** 2026-06-28 · **Hardware:** RTX 4090 workstation (GPU drives display + CUDA)

### Symptoms

- Chaining Robocasa dynagraph explore → full pytest (MuJoCo-native tests) → Habitat HM-EQA with VLM in one session.
- Full-system **live lock**: mouse moves, GUI and SSH unresponsive; journal stops mid-line.
- Separately: **Cursor agent** (`node` MainThread) or `emet` can log `trap invalid opcode` in kernel after Habitat-Sim / VLM teardown — agent session dies but eval subprocess may have finished (check log files and JSON artifacts).

### Mitigation

- **One GPU-heavy job at a time** — use [`scripts/gpu_preflight.sh`](../scripts/gpu_preflight.sh) (`--kill-stale`, `--wait`, `--check`).
- Cross-track smoke: [`run_overnight_cross_track_smoke.sh`](../scripts/run_overnight_cross_track_smoke.sh) defaults **`RUN_DEEP_EVAL=0`**; run [`run_overnight_eval_smoke.sh`](../scripts/run_overnight_eval_smoke.sh) on a **separate night**.
- Safe no-sim pytest: source `gpu_preflight.sh` and pass **`emet_pytest_no_sim_ignore_args`** (excludes unmarked MuJoCo paths under `src/test/simulation/`).
- Long evals: **`nohup … &`** or dedicated terminal — not blocking Cursor agent inline runs.

Docs: [evaluation.md](evaluation.md#gpu-preflight-all-overnight--vlm-jobs), [cross_track_smoke.md](experiments/cross_track_smoke.md).
