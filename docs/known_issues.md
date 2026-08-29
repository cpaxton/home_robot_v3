# Known issues

Tracked bugs and investigation notes.

**See also:** [cli.md](cli.md) · [pythonpath.md](pythonpath.md) · [zmq_obs.md](zmq_obs.md) · [dynagraph.md](dynagraph.md) · [graph_eqa.md](graph_eqa.md) · [robots/innate_mars.md](robots/innate_mars.md) · [robots/innate_mars_hardware.md](robots/innate_mars_hardware.md) · [environment_variables.md](environment_variables.md)

---

## `emet serve mujoco` dies on `import scipy` / numpy ABI

**Status:** Mitigated · **Seen:** 2026-08 (sim server child)

### Symptoms

`uv run emet serve mujoco` (or a harness that `Popen`s `python -m emet.simulation.mujoco_server`) fails at import with scipy/numpy ABI errors (`undefined symbol`, `module compiled against API version …`), while `uv run python -c "import scipy"` in the same checkout works.

### Cause

`sanitize_emet_subprocess_env` used to prepend **every** `.venv/lib/python*/site-packages` glob. A leftover `python3.12` directory inside a 3.10 venv put 3.12 wheels on `PYTHONPATH` ahead of the venv’s own. `mujoco_server` imports numpy at **module load**, before in-process `ensure_venv_site_packages_first()`.

### Fix

Prepend only `python{tag}/site-packages` for the tag in `.venv/pyvenv.cfg` `version_info`. Details and debug commands: [pythonpath.md](pythonpath.md).

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
emet mars start --connection mars
emet stream --connection mars --backend dynagraph --headless
# Robot stationary; watch graph node count in periodic status lines
```

Sim with sensor depth may **not** show the same severity (dedup is more stable when depth is consistent).

Optional **DA3 post-filters** (speckle open on inferred depth; voxel PCD DBSCAN on navigation PCD) can reduce floating ``world/point_cloud`` blobs; they are **off by default** — see [dynav_config.md](dynav_config.md#depth--voxel-post-filters-da3-hardware-opt-in).

### Fix (2026-06)

Three changes address stationary hardware stream growth:

1. **GraphObjectFusion fallback tier** (`fallback_spatial_merge_xy_m`, default **0.45 m**, aligned with `dynagraph_merge_xy_m`). When strict spatial/embedding/bounds gates fail, instance detections merge into the nearest object node within that XY radius.
2. **Innate Mars fusion YAML wired correctly** — `attach_graph_object_fusion` now reads `graph_object_fusion` from `Parameters` / yacs config (controllers no longer pass `None` when parameters is not a plain `dict`). [`dynav_innate_mars.yaml`](../src/emet/config/dynav_innate_mars.yaml) supplies relaxed gates (embedding off, fallback **0.55 m**).
3. **VLM sensor labels through fusion** — Qwen-extracted labels in [`dynamem_graph_hooks.py`](../src/emet/memory/graph_eqa/ingest/dynamem_graph_hooks.py) use `apply_detection` instead of `add_observation` with dedup disabled (~8 new nodes/step before).

**Stream status** now reports object / viewpoint / frontier breakdown: `graph 11 obj / 12 vp / 1 fr (24 total)` ([`stream_agent_factory.py`](../src/emet/app/stream_agent_factory.py)).

**Offline regression:** [`src/test/memory/test_graph_dedup_offline.py`](../src/test/memory/test_graph_dedup_offline.py) + [`scripts/build_dedup_calibration_fixtures.py`](../scripts/build_dedup_calibration_fixtures.py). **Hardware diagnostic:** [`scripts/diagnose_stream_graph_growth.py`](../scripts/diagnose_stream_graph_growth.py).

Herman smoke after fix: object nodes plateau ~10–11 after step 1 with default VLM+instance stream (was +8–9 object nodes/step).

### Why dedup is failing (hypothesis)

Several merge paths exist; on stream + hardware they appear **too weak** for noisy stationary observations:

| Mechanism | Config / code | Limitation on stationary IRL |
|-----------|---------------|------------------------------|
| **GraphObjectFusion** | [`graph_object_fusion`](../src/emet/config/agents/default_graph_object_fusion.yaml) in dynav (enabled on stream via [stream_agent_factory](../src/emet/app/stream_agent_factory.py)) | Merges when XY ≤ `spatial_merge_xy_m` (0.42 m), 3D centroid ≤ `min_centroid_dist_m` (0.55 m), bounds IoU, embedding cosine ≥ 0.62. **DA3 depth jitter** and pose/camera noise can push repeated views of the same object outside these gates → **new node every step**. See [fusion.py](../src/emet/memory/graph_eqa/graph_object_fusion/fusion.py) and [dynagraph.md § Configuration keys](dynagraph.md#configuration-keys). |
| **Label-based pre-dedup** | `graph_instance_dedup_xy_m` (default **0.4 m**; [graph_eqa.md](graph_eqa.md)) | `_graph_dedup_skips` uses :func:`~emet.memory.graph_eqa.graph_stats.labels_compatible_for_dedup` (exact / substring / shared tokens / synonym groups) so ``mug`` vs ``coffee cup`` no longer bypasses XY dedup. See [controller_graph_eqa.py](../src/emet/controller/controller_graph_eqa.py). |
| **Dynagraph spatial merge** | `dynagraph_merge_xy_m` (default **0.45 m** on stream; [dynagraph.md](dynagraph.md)) | `GraphEQAMemory.add_observation` merges **compatible** labels within XY — but **`spatial_merge_m` is cleared to 0** when GraphObjectFusion is enabled ([attach.py](../src/emet/memory/graph_eqa/graph_object_fusion/attach.py)); fusion fallback covers that path. |
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

- `src/emet/memory/graph_eqa/ingest/graph_mutate.py` — `add_observation`, `merge_object_detection`
- `src/emet/memory/graph_eqa/store.py` — `spatial_merge_m` on `GraphStore`
- `src/emet/memory/graph_eqa/graph_object_fusion/fusion.py` — merge gates
- `src/emet/controller/controller_graph_eqa.py` — `_graph_dedup_skips`
- `src/emet/app/stream_agent_factory.py` — `dynagraph_merge_xy_m` / `graph_object_fusion` defaults for stream
- `src/emet/config/agents/default_graph_object_fusion.yaml`

---

## EQA hangs after “Qwen3-VL ready for inference”

**Status:** Mitigated (2026-07-22) · **Seen:** overnight improve/paper-cell smokes (`STALE_KILL` ~30 min after VLM load; no metrics)

### Cause (two layers)
1. **SigLIP + Qwen VRAM pressure:** Dynagraph kept SigLIP on GPU for CONFIRMED_MEMORY while loading Qwen3-VL-8B int4. Weight load ~125 s; first text `generate` never finished → `STALE_KILL`.
2. **MuJoCo EGL + vision prefill:** After SigLIP release, text generate completed (~100 s) but vision EQA (`prompt≈4500`, `max_new=512`) ran with no log growth for ~30 min until `STALE_KILL`. Isolated 4-image EQA finishes in ~3.5 s; the full path still called `look_front` / nav posture over ZMQ while the VLM ran on the same GPU.
3. **Silent SDPA fallback:** when `flash-attn` was missing, CUDA VL loads quietly used PyTorch SDPA. Habitat MCQ still finished (~4–5 min/ep), but Robocasa multi-image `query_answer` (`prompt≈4500`, 4 RGB) decoded at ~0.02 tok/s (~45 s/token) and looked “stuck” at low GPU util.

### Mitigation
- [`prepare_dynagraph_vram_for_eqa`](../src/emet/eval/dynagraph_vram.py) warms SigLIP phrase caches **and visual FIND top-k ranks**, then **always releases** SigLIP before the EQA VLM. Voxel `find_all_images` cannot run after that release. **OVMM find** re-attaches the shared encoder per phase (`re_attach_siglip_encoder` in `emet.eval.ovmm_agentic_find`; CUDA if available, otherwise CPU / `cpu_only`), so FindObj **and** FindRec can still `localize_text` the finished map; HM-EQA keeps the released state.
- [`release_shared_mask_siglip_encoder`](../src/emet/perception/encoders/siglip_encoder.py) moves weights to CPU and empties the CUDA cache.
- Answer-only EQA skips robot head/posture I/O (`allow_navigation=False` → `skip_perception_prelude`).
- Before EQA, [`release_zmq_ports`](../src/emet/utils/port_utils.py) kills MuJoCo **LISTEN** sockets on the session ports so EGL is not sharing the GPU with Qwen (must not use plain `lsof -i:PORT`, which also matches the dynagraph client and SIGTERMs it — exit 241).
- `[vl] generate heartbeat` every 30 s (`EMET_VL_GENERATE_HEARTBEAT_S`) + `[vl] decode started` when prefill ends.
- Hard VL generate timeout: `EMET_VL_GENERATE_TIMEOUT_S` (default **180 s**) raises `VlGenerateTimeoutError` instead of multi‑hour SDPA crawls; set `0` to disable, or lower if hangs return.
- `eqa_vl.answer_max_new_tokens` (default `384`, env override `EMET_EQA_ANSWER_MAX_NEW_TOKENS`) caps answer-only decode length. Tune it per VLM; too low and the model never reaches `answer:`.
- Improve smoke raises `EMET_DYNAMIC_EXPLORE_STALE_*` / `EMET_EQA_QUESTION_TIMEOUT_S`.
- CUDA VL loads **require Flash-Attn 2** by default ([`attn_impl.py`](../src/emet/llms/attn_impl.py)); missing package raises instead of silent SDPA. Escape hatch: `EMET_ALLOW_SDPA_ATTN=1`.

### Repro / check
- `EMET_AGENT_MODEL_DEBUG=1 timeout 600 uv run python scripts/debug_eqa_vlm_hang.py --with-image --eqa-prompt --n-images 4`

---

## Orphan / zombie eval processes after timeouts

**Status:** Mitigated (2026-07) · **Seen:** dynamic-exploration smoke (EQA hang left 11 GiB `emet run dynagraph` for days; 14-day `uv run emet test` with `<defunct>` child)

### Cause
- Timeouts / `terminate()` on the direct child only (`uv` or a thin wrapper) leave Python/GPU grandchildren alive.
- Sim servers started without a process group were not reaped by parent cleanup.

### Mitigation
- Shared helpers in [`src/emet/utils/process_tree.py`](../src/emet/utils/process_tree.py): `popen_session` + `terminate_process_tree` / `kill_process_tree` (wired into dynamic exploration, sim/Habitat subprocess spawn, OVMM find-phase, sim eval sessions).
- CLI: **`uv run emet eval kill-stale`** / `check` / `wait` / `status` ([`emet.utils.gpu_preflight`](../src/emet/utils/gpu_preflight.py)); [`scripts/gpu_preflight.sh`](../scripts/gpu_preflight.sh) delegates to that CLI.
- Smoke scripts that are themselves `run_dynagraph_dynamic_*` must **not** call `kill-stale` on themselves (use `emet eval wait` only).

---

## NVIDIA driver hang / Cursor agent crash during stacked GPU evals

**Status:** Mitigated (2026-06) · **Seen:** 2026-06-28, 2026-07-24 · **Hardware:** RTX 4090 workstation (GPU drives display + CUDA)

There are **two distinct segfault modes**. Do not conflate them.

### Mode A — Habitat episode `libcuda` SIGSEGV (`exit=139`)

- **What dies:** `emet-habitat run-episode` / `.venv-habitat` Python. Orchestrator logs `FAIL … exit=139` and `timeout: the monitored command dumped core`.
- **Kernel:** `python[…]: segfault at 43/44 … in libcuda.so.*` (same IP offset across repeats).
- **When:** Qwen3-VL **vision** `generate` (look-around / mid-episode) while Habitat-Sim **EGL** still owns the same GPU. Worst scenes: `00167-yogvKWUrdnw` (q104/q105); also flaky on `00094-WT4QWwXrMzs` (q68).
- **Artifacts:** empty `OUT/agentic_qN.jsonl` (touched then never written). Job may continue to later IDs unless `NATIVE_CRASH_ABORT=1` (default).
- **Stack that triggers it:** Habitat torch `+cu130` + `EMET_ALLOW_SDPA_ATTN=1` (no flash-attn wheel) + int4 bitsandbytes + windowless EGL on a display GPU.
- **Partial mitigations:** `torch.cuda.synchronize()` before multimodal generate in `qwen3_vl_client`; H2H default **`NATIVE_CRASH_POLICY=skip`** (settle + retry + continue) with **`NATIVE_CRASH_STREAK_ABORT=2`** (early abort when consecutive native crashes imply a wedged harness); optional `EMET_GRAPH_EQA_EXTRACT_VLM=0` if mid-nav extract forces the race (do **not** auto-tie this to agentic verify — that caused `n_object=0` on q104/q105). Failures remain **flaky**. Prefer **`emet hmeqa resume`** after **`emet eval recover`**.
- **See:** [agentic_scale.md](experiments/agentic_scale.md#fail-set-regressions-found-2026-07-24).

### Mode B — Cursor agent / `emet` null-IP SIGSEGV

- **What dies:** the Cursor agent process (`emet[…]: segfault at 0` null IP, or `trap invalid opcode`) when a turn runs or probes Habitat / tears down GPU context. Also seen as V8 `Illegal instruction` in the `agent`/`node` binary itself (`traps: MainThread[…] trap invalid opcode … in node`).
- **What survives:** a detached **`emet jobs`** child often keeps running — check registry + `OUT/` before re-launching.
- **Post-sim job wrapper (2026-08-27):** immediately after a MuJoCo OVMM job releases `gpu.lock`, `emet eval affinity --apply` (full CLI, which used to import MuJoCo via `export-sim-gt`) segfaulted in `mujoco/_functions.so`. The OVMM command never started. Mitigations: **cpu-safe** wrappers pin with `python -m emet.utils.cpu_affinity`; `emet.cli` lazy-loads sim commands so `emet jobs` / `emet eval` / `emet --help` do not import MuJoCo. Do not requeue a GPU smoke in the same agent turn as a just-finished sim job; do not use `--need-mib` / inline `emet eval wait` from the agent (08:04 libc SIGSEGV).
- **First command on the way back (from the owning checkout):** **`uv run emet status tail`** — the last record says what state the run reached and the literal next command ([evaluation.md](evaluation.md#first-command-after-an-agent-death-uv-run-emet-status-tail)). Do not `tail ~/runs/emet/STATUS.log` (flat path is shared across sibling checkouts). `uv run emet status latest` points at that checkout's newest `OUT/`.
- Also: full-system **live lock** when chaining Robocasa dynagraph explore → full pytest (MuJoCo-native) → Habitat HM-EQA with VLM in one session (mouse moves; GUI/SSH dead).
- **Not** explained by a busy GPU: Mode A/B and EGL map failures have happened with `nvidia-smi` showing only Xorg/gnome-shell and ~full free VRAM.

### Habitat EGL failure with “idle” GPU (2026-07-24)

- Failset log: `Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among 2 EGL devices` / `WindowlessContext: Unable to create windowless context` on every classic episode while NVML reported free VRAM.
- Later the same machine ran agentic H2H successfully after `/dev/nvidia*` refresh — treat repeated EGL errors as driver/EGL map breakage, not “need more kill-stale”.
- H2H orchestrator aborts after consecutive EGL failures (`EGL_FAIL_ABORT`, default 2). Diagnose with **`uv run emet eval diagnose`**.

### Mode C — Host hard freeze / forced reboot (2026-07-25)

- **What dies:** the whole machine (GUI + SSH dead; journal ends mid-line with no oops/Xid). After reboot, `uptime` is minutes and the previous boot’s last timestamp matches the in-flight episode.
- **Seen:** bal-32 classic **q48**, planning step 4, mid-VLM decode (`[vl] decode tokens=161`). `classic_q48.jsonl` empty; `classic.log` ends with trailing NUL bytes (unclean write). Job `hmeqa-bal32-aff` → `failed` / `exited without DONE`. Progress was **26/64**, not a clean native SIGSEGV.
- **Affinity pitfall:** excluding only CPUs **8–9** is incomplete on this i9-14900KF — CPUs **10–11** are the second 6.0 GHz P-core. `taskset -c 0-7,10-31` still schedules onto turbo silicon. H2H now auto-pins via [`emet.utils.cpu_affinity`](../src/emet/utils/cpu_affinity.py) (exclude ≥6000 MHz → keep `0-7,12-31`) plus `EPISODE_COOLDOWN_SEC` between episodes.
- **Also:** do not queue MuJoCo/EGL jobs from a sibling checkout while Habitat HM-EQA is live (even with `wait_pids`) — a freeze kills the waiter too and multiplies GPU/driver stress after reboot.
- **2026-08-22 recurrence (count/clock resume):** the `cc-singleview-resume` wrapper (hand-built, `eval wait` but **no mutex**) ran while another GPU experiment was live → box froze again mid-q12. **Fix:** `emet jobs run --need-mib` now wraps the command in a **shared singleflight `flock`** on `~/runs/emet/gpu.lock` (all checkouts coordinate; `EMET_GPU_LOCK` / `EMET_GPU_LOCK_TIMEOUT`). Never hand-build a jobs wrapper that skips the flock — `eval wait`'s free-VRAM gate alone has a TOCTOU race.

### Mitigation

- **One GPU-heavy job at a time** — use **`uv run emet eval kill-stale` / `wait` / `check` / `diagnose`** ([`emet eval`](cli.md#emet-eval-gpu-preflight--stale-cleanup); bash [`scripts/gpu_preflight.sh`](../scripts/gpu_preflight.sh) delegates).
- Cross-track smoke: [`run_overnight_cross_track_smoke.sh`](../scripts/run_overnight_cross_track_smoke.sh) defaults **`RUN_DEEP_EVAL=0`**; run [`run_overnight_eval_smoke.sh`](../scripts/run_overnight_eval_smoke.sh) on a **separate night**.
- Safe no-sim pytest: source `gpu_preflight.sh` and pass **`emet_pytest_no_sim_ignore_args`** (excludes unmarked MuJoCo paths under `src/test/simulation/`).
- Long evals: **`uv run emet jobs run --name … -- CMD`** (or dedicated terminal) — **not** blocking Cursor agent inline runs; do not hard-kill Habitat mid-episode from the agent (use **`emet jobs cancel`**).
- On Mode A: leave `NATIVE_CRASH_ABORT=1`, inspect `native_crash_*.log` + `journalctl -k` for `libcuda`; retry the failed qid only after `emet eval diagnose` / free GPU — do not treat empty jsonl as a scored miss without a crash capsule.
- On Mode C: after reboot, `uv run emet status tail`, confirm GPU idle, resume with H2H’s built-in affinity (no manual incomplete `taskset`); empty mid-episode jsonl is re-run by `RESUME=1`.

Docs: [evaluation.md](evaluation.md#gpu-preflight-all-overnight--vlm-jobs), [cross_track_smoke.md](experiments/cross_track_smoke.md), [`.cursor/rules/gpu-eval-workflow.mdc`](../.cursor/rules/gpu-eval-workflow.mdc).
