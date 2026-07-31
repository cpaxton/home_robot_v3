# Evaluation runbook

Canonical guide for paper-relevant benchmarks: Habitat HM-EQA, OVMM find-phase (Habitat + sim), and SQA3D. Use this doc for **overnight smoke**, **diagnostics artifacts** (maps, video, crops), and **figure export**.

Deep dives:

| Track | Doc |
|-------|-----|
| Habitat EQA | [habitat_eqa.md](habitat_eqa.md) → [habitat/](habitat/README.md) |
| OVMM find-phase | [ovmm_find_phase_benchmark.md](ovmm_find_phase_benchmark.md), [ovmm.md](ovmm.md) |
| SQA3D | [sqa3d.md](sqa3d.md), [sqa3d_compute.md](sqa3d_compute.md) |
| Paper tables / LaTeX | [paper_benchmarks.md](paper_benchmarks.md) |
| Dynagraph sim | [dynagraph_benchmarks.md](dynagraph_benchmarks.md) |

## Agentic GraphEQA verify + offline tuning

Post-explore / world-change EQA can run a bounded **tool loop** (`eqa.agentic_verify` /
`EMET_EQA_AGENTIC_VERIFY=1`): recall place evidence → navigate / explore → SigLIP
**proposal** → VLM assess → submit (or explore `finish`). Improve smokes enable this by
default and write `agentic_trace.jsonl` (SigLIP embeds + sim GT) under each export dir when
`EMET_EQA_TRACE=1`. Detectors do not unlock submit; answerability is VLM-first.

**Modes (same skills library, different packs):** Discord / terminal house chat is
``AgentMode.CHAT`` (`emet run agent`). Scored verify/explore is ``AgentMode.EQA_EPISODE``
(`build_agentic_eqa_tools` / `AgenticEQAExecutor`). Shared skill membership lives in
[`src/emet/agent/skills/`](../src/emet/agent/skills/). Habitat MCQ scoring never routes through
Discord chat turns; use `run_eqa` / `emet-habitat` / `--eqa-eval`. See [AGENT_RUN.md](AGENT_RUN.md#skill-library-vs-orchestrator-modes).

Tool picks come from the shared Qwen3-VL via text-only JSON tool-calling turns (same
`{"tool_calls": ...}` contract as the Discord agent; the fixed system prompt gets prefix
KV-cache hits). Each round the router sees a small set of **evidence cards** recalled from
the graph / SigLIP / frontiers (`EMET_EQA_HYP_RECALL_K`, default 6) — retrieval only; the
VLM decides among listed `obs_id`s. Visited frontier nodes are retired from the graph.
`eqa.agentic_vlm_router: false` or `EMET_EQA_AGENTIC_ROUTER=0` disables the VLM router and
uses the deterministic fallback policy only (walks recalled cards; reproducible evals);
parse failures always fall back. Explore mode (`run_agentic_eqa(agent, question=None, goal=...)`)
drives `explore_frontier`/`look_around` and ends with a `finish` coverage summary once
frontiers or the nav budget are exhausted. Trace rows record `picked_by`,
`router_parse_ok`, `router_raw_reply_chars`, and `router_tool_calls` so the tuner can
compare VLM-router vs fallback quality. Approach notes:
[agentic_qwen_context.md](experiments/agentic_qwen_context.md#approach-current).

```bash
# After a run that produced traces:
uv run python scripts/tune_agentic_verify.py ~/runs/emet/dynamic_exploration/<run> \
  -o /tmp/agentic_tune_report.json
```

The tuner sweeps verify thresholds (precision/recall/F1 vs `gt_present`) and budget knees
(accuracy vs `max_tool_rounds`). Nav-distance candidates are correlational only — confirm with
a real smoke. Contract tests: `uv run emet test src/test/eval/test_agentic_eqa_verification.py`.

## Prerequisites

```bash
uv sync
./scripts/install_habitat.sh
uv run python scripts/download_habitat_eqa_data.py --fetch-csv --fetch-hm3d train
uv run python scripts/download_ovmm_benchmark_assets.py   # optional verify
uv run python scripts/download_sqa3d_data.py --fetch-annotations
# ScanNet (SQA3D): uv run python scripts/download_scannet_data.py --accept-tos --scenes-from-sqa3d --with-sens
```

## Shared diagnostics

All embodied tracks can write a **consistent episode bundle** via [`src/emet/eval/episode_diagnostics.py`](../src/emet/eval/episode_diagnostics.py):

```
~/.cache/habitat_eqa/episodes/<run_tag>/
  q0003_graph_eqa/              # HM-EQA
  ovmm_hm3d_lamp_bed_00006_dynamem/
  sqa3d_220602000049_dynagraph/
    topdown_map.png
    topdown_gt_navmesh.png       # Habitat: navmesh GT (same crop as agent map)
    topdown_map_overlay.png      # Habitat: GT + explored + trajectory
    obstacles_2d.npy, explored_2d.npy, grid_meta.json
    trajectory.jsonl
    nav_attempts.jsonl             # per nav attempt (goal_xy, path_xy, note, …)
    frames/rgb_*.png, metadata.jsonl
    episode_rgb.mp4              # head-camera first-person (manifest: head_camera_mp4)
    topdown_exploration.mp4      # map timelapse (GT overlay when Habitat navmesh available)
    maps/step_NNNN.png           # optional stride snapshots
    maps/overlay_step_NNNN.png   # GT overlay stride snapshots (when export_map_overlay)
    floor_metrics.json
    diagnostics_manifest.json
    (track-specific: raw_eqa.txt, memory/, …)
```

**Environment variables** (see [environment_variables.md](environment_variables.md)):

| Variable | Default (smoke) | Effect |
|----------|-----------------|--------|
| `EMET_EVAL_EXPORT_MAP` | on | `topdown_map.png` (+ trajectory when `EMET_EVAL_EXPORT_TRAJECTORY=1`) |
| `EMET_EVAL_EXPORT_GT_MAP` | on (Habitat) | `topdown_gt_navmesh.png` from HM3D navmesh |
| `EMET_EVAL_EXPORT_MAP_OVERLAY` | on (Habitat) | `topdown_map_overlay.png` (GT + agent + trajectory) |
| `EMET_EVAL_EXPORT_MAP_VIDEO` | on | `topdown_exploration.mp4` from stride map frames |
| `EMET_EVAL_MAP_VIDEO_STRIDE` | `5` | Steps between map frames when `EMET_EVAL_MAP_STRIDE=0` |
| `EMET_EVAL_MAP_MAX_SIDE` | `1280` | Max map width/height in pixels (was 640) |
| `EMET_EVAL_MAP_MIN_SIDE` | `1024` | Upscale small crops to at least this size |
| `EMET_EVAL_FILTER_MAP_ISLANDS` | on | Drop explored blobs disconnected from robot path |
| `EMET_EVAL_EXPORT_VIDEO` | on | `episode_rgb.mp4` |
| `EMET_EVAL_EXPORT_VIDEO_SUBSTEPS` | on (Habitat) | Record RGB after each Habitat `sim.step` (nav / rotate substeps). YAML: `eval.export_video_substeps`. |
| `EMET_EVAL_VIDEO_MOTION_PACED` | on | Motion-paced MP4 (repeat frames by planar / yaw delta). YAML: `eval.video_motion_paced`. |
| `EMET_EVAL_EXPORT_FRAMES` | on | RGB frame PNGs |
| `EMET_EVAL_MAP_STRIDE` | 0 | Intermediate `maps/step_NNNN.png` (when >0; else `map_video_stride` drives auto stride for map video) |
| `EMET_EVAL_EXPORT_GRAPH` | off | Full graph checkpoint (heavy) |
| `EMET_EVAL_EXPORT_VOXEL_HISTORY` | on (Habitat) | Per-observation `observations_history.jsonl` |
| `EMET_EVAL_EXPORT_VOXEL_PICKLE` | off | Full `voxel_debug.pkl` (heavy) |

**YAML-only video pacing** (no env alias): `eval.video_meters_per_frame`, `eval.video_radians_per_frame`, `eval.video_crossfade_teleport_m`, `eval.video_fps` — see [`src/emet/config/eval/default.yaml`](../src/emet/config/eval/default.yaml).

Habitat aliases: `HABITAT_EQA_EXPORT_MAP`, `HABITAT_EQA_EXPORT_VIDEO`, `HABITAT_EQA_EXPORT_GRAPH`, `HABITAT_EQA_MAP_STRIDE`.

**Unified config:** the same settings live under **`eval:`** in [`configs/emet/default.yaml`](../configs/emet/default.yaml) (`export_map_video`, `export_video`, `map_video_stride`, …). Override with **`--set eval.export_map_video=false`** or a preset YAML block. Precedence: CLI/runner kwargs → env (when set) → YAML → defaults. See [emet_config.md](emet_config.md).

**Recording:** eval runners call `bind_diagnostics_recorder()` which registers a step callback on the agent. After each successful `DynamemController.update()` (navigation / mapping step), the callback buffers RGB, pose, and optional stride maps — no monkey-patching of `agent.update`. On Habitat, when `export_video_substeps` is on, `HabitatRobotClient` post-step hooks append one RGB frame per discrete sim action (smoother head-camera MP4 during navmesh following).

**CLI flags** (`.venv-habitat/bin/emet-habitat`): `--export-map`, `--export-video`, `--map-stride` on `run-episode` / `run-batch`; OVMM batch adds `--run-tag`. With `--map-stride N`, intermediate maps are written under `<bundle>/maps/step_NNNN.png` at episode end.

### Habitat frame sanity (before trusting map colors)

HM-EQA depth and `gps` must share the same voxel-world frame (`src/emet/habitat/coordinates.py`). Misalignment used to produce wall-to-wall red maps or orphan “satellite” explored blobs.

After an episode with `EMET_EVAL_EXPORT_VOXEL_HISTORY=1`:

```bash
uv run python scripts/audit_habitat_voxel_map.py \
  ~/.cache/habitat_eqa/episodes/<run_tag>/q0006_dynagraph --json
```

Check:

| Field | Healthy signal |
|-------|----------------|
| `pcd_planar_x_mismatches` | empty (no Habitat X sign flip vs `base_pose`) |
| `explored_obstacle_frac` | not ~0.9 (false wall-to-wall obstacles) and not ~0.0 from height-band miss |

Optional live regression: `RUN_HABITAT_FRAME_TESTS=1 uv run emet test src/test/eval/test_audit_habitat_voxel_map.py -k live_habitat`.

## GPU preflight (all overnight / VLM jobs)

Before any GPU-heavy eval, kill stale jobs and wait for headroom. Shared helpers live in [`scripts/gpu_preflight.sh`](../scripts/gpu_preflight.sh) (sourced by overnight scripts):

```bash
# Preferred (canonical CLI):
uv run emet eval kill-stale
NEED_MIB=12000 uv run emet eval wait
NEED_MIB=14000 uv run emet eval check
uv run emet eval status

# Track queued / running experiments (required for long GPU runs):
uv run emet jobs                 # list + progress/ETA columns
uv run emet jobs status JOB_ID   # detail + derived ETA
uv run emet jobs cancel JOB_ID   # pause/stop one managed job (then resume via overnight --base / hmeqa resume)

# Launch (prefer over bare nohup):
uv run emet jobs run --name my-eval --need-mib 12000 -- ./scripts/…

# Bash helpers still work (delegate to emet eval):
./scripts/gpu_preflight.sh --kill-stale
NEED_MIB=12000 ./scripts/gpu_preflight.sh --wait
NEED_MIB=14000 ./scripts/gpu_preflight.sh --check
```

### Pause / resume HM-EQA overnight (official)

There is **no** separate `emet pause` — cancel the managed job, then relaunch with the same tree:

```bash
uv run emet jobs cancel JOB_ID
uv run emet jobs                 # confirm no unmanaged emet-habitat orphans
uv run emet eval status          # GPU should clear (~no compute apps)

# Resume the full holdout→bal32 ladder (skips DONE phases; RESUME=1 on partial H2H):
uv run emet hmeqa overnight --base ~/runs/emet/hmeqa_overnight_<stamp> --job-name hmeqa-overnight

# Or resume only the bal-32 / holdout H2H OUT:
uv run emet hmeqa resume ~/runs/emet/hmeqa_overnight_<stamp>/bal32 --preset paper-router
```

Do **not** re-run overnight with a **new** `--base` if you want to keep scored episodes. Mid-episode cancel leaves an empty `*_qN.jsonl`; resume retries those. Prefer `emet jobs cancel` over raw `kill` / `emet eval kill-stale` while a managed job is the thing you want to stop. Details: [cli.md](cli.md#emet-jobs-queued--running-eval-experiments), [cli.md](cli.md#emet-hmeqa-hm-eqa-h2h).

`kill-stale` SIGTERM→SIGKILL matching sim/eval/`uv run emet` trees (skips the caller ancestry; optional `EMET_GPU_PROTECT_PIDS`). Eval code should spawn via `emet.utils.process_tree` so timeouts reap GPU grandchildren — see [known_issues.md](known_issues.md#orphan--zombie-eval-processes-after-timeouts).

Also sets `PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_ALLOC_CONF` to `expandable_segments:True` when scripts call `emet_export_pytorch_alloc`.

**Rule:** do not chain Robocasa dynagraph smoke, full pytest with MuJoCo tests, and Habitat VLM eval in one uninterrupted GPU session — that pattern caused full-system freezes (GUI + SSH unresponsive) on a 4090 workstation. Run cross-track smoke and deep eval on **separate nights** (see below).

### Cursor / agent sessions

Long Habitat evals and overnight orchestrators should run via **`uv run emet jobs run --name … -- CMD`** (or a dedicated terminal), not as blocking inline Cursor agent commands and not as unmanaged bare `nohup` when avoidable. Monitor with **`emet jobs`** / **`emet jobs status`** (progress + ETA from meta / `OUT/progress.json`). Native GPU teardown (Habitat-Sim, VLM unload) can crash the agent process while eval subprocesses finish — check `~/runs/emet/`, `~/.cache/habitat_eqa/results/`, and per-step logs before re-running. See [cross_track_smoke.md](experiments/cross_track_smoke.md#cursor--long-agent-sessions) and [cli.md](cli.md#emet-jobs-queued--running-eval-experiments).

### First command after an agent death: `uv run emet status tail`

```bash
# From the checkout that owns the job (home_robot_v4 ≠ v3 ≠ v2):
uv run emet status tail     # what happened + the literal next command
uv run emet status path     # ~/runs/emet/status/<repo>/STATUS.log
uv run emet status latest   # symlink to that checkout's newest OUT dir
```

Do **not** `tail ~/runs/emet/STATUS.log` — that flat path is shared across sibling checkouts and would interleave recovery instructions from other agents. The default is namespaced:

`~/runs/emet/status/<repo_basename>/STATUS.log`

Orchestrators that source [`scripts/status_log.sh`](../scripts/status_log.sh) append one self-contained record per state change (`START`, `RUNNING`, `OK`, `FAIL`, `CRASH`, `EGL`, `EXIT`, `DONE`, `BLOCKED`), mirrored to `OUT/STATUS.log`. Each record includes `repo:` and ends with a **`next:`** line:

```text
=== 2026-07-25T10:39:37-04:00  hmeqa-bal32-rerun  CRASH  5/64 classic q14
    repo: /home/cpaxton/src/home_robot_v4
    out:  /home/cpaxton/runs/emet/hmeqa_agentic_bal32_20260725_101519
    job:  20260725_101522_3b3b11  (cd /home/cpaxton/src/home_robot_v4 && uv run emet jobs status …)
    what: SIGSEGV in classic q14 — batch aborted at 5/64 units
    next: 1) read …/native_crash_classic_q14.log  2) sudo dmesg -T | rg -i 'segfault|invalid opcode'  3) uv run emet eval diagnose  4) only then resume: uv run emet jobs run …
```

An `EXIT` record is written by a bash `EXIT` trap when the orchestrator dies without a final state (SIGKILL, driver hang, unhandled error); it carries the run's `STATUS_RESUME_CMD`. `DONE`/`CRASH`/`EGL`/`BLOCKED` disarm the trap. To add this to another orchestrator:

```bash
source "$ROOT/scripts/status_log.sh"
status_open "$OUT" my-eval
STATUS_RESUME_CMD="RESUME=1 ./scripts/my_eval.sh $OUT"
STATUS_PROGRESS="3/40 phase-2"
status_note RUNNING "phase 2 started" "nothing — wait"
status_close DONE "all units finished" "review $OUT/summary.txt"
```

Override with `EMET_STATUS_LOG` / `EMET_STATUS_DIR` (see [environment_variables.md](environment_variables.md#large-paper-eval-orchestrator)).

## Simulation smoke battery (seven tracks)

Paper-facing sequential validation: Habitat EQA → Habitat OVMM → Robocasa search → Molmo iTHOR search → SQA3D → Robocasa world-change → Molmo explore-loop. **Run before multi-day sweeps** when changing eval harnesses or sim wiring.

```bash
./scripts/run_simulation_smoke_battery.sh
```

Details: [simulation_testing_plan.md](simulation_testing_plan.md) · `paper/sections/04_experiments.tex` (*Simulation smoke battery*).

## Cross-track smoke (extended overnight)

Validates SQA3D, Habitat EQA, OVMM, Robocasa explore/world-change, and a safe unit-test pass before multi-day sweeps. Details: [experiments/cross_track_smoke.md](experiments/cross_track_smoke.md).

```bash
./scripts/run_overnight_cross_track_smoke.sh
# Optional: chain deep Habitat eval (not recommended same night):
# RUN_DEEP_EVAL=1 ./scripts/run_overnight_cross_track_smoke.sh
```

Logs: `~/runs/emet/overnight_cross_track/<RUN_ID>/`.

## Overnight smoke (Habitat + OVMM + SQA3D matrix)

Run **after** cross-track passes, on a clean GPU (or the next day). One script runs HM-EQA, OVMM Habitat, and SQA3D (if ScanNet verify passes), then builds figures:

```bash
./scripts/run_overnight_eval_smoke.sh
# Dry layout check (no VLM):
MOCK_LLM=1 ./scripts/run_overnight_eval_smoke.sh
# Skip SQA3D if ScanNet not installed:
SKIP_SQA3D=1 ./scripts/run_overnight_eval_smoke.sh
```

**Matrix (~21 GPU episodes with real VLM):**

| Phase | Benchmark | Units | Methods |
|-------|-----------|-------|---------|
| 1 | HM-EQA | Q `3,14,17` | `static_graph`, `dynagraph` |
| 2 | OVMM Habitat | 3 HM3D proxy episodes | **`dynamem`**, `static_graph`, `dynagraph` |
| 3 | SQA3D | val Q `0–2` | `dynagraph`, `dynamem` |

Outputs:

- JSONL: `~/.cache/habitat_eqa/results/<TAG>_hmeqa_*.jsonl`
- OVMM JSON: `~/runs/emet/ovmm_habitat/<TAG>_*/`
- SQA3D: `~/runs/emet/sqa3d/<TAG>_*/`
- Bundles: `~/.cache/habitat_eqa/episodes/<TAG>_*/`
- Logs: `~/.cache/habitat_eqa/overnight/<RUN_ID>/`
- Figures: `~/runs/emet/eval_smoke/<RUN_ID>/figures/`

**`RUN_ID` vs `TAG`:** the overnight script writes artifact paths with `TAG` (defaults to `RUN_ID`). Figure aggregation uses `--run-id` (`RUN_ID`). If you override `TAG` without setting `RUN_ID` to match, the script prints a warning and passes `--artifact-tag "$TAG"` to `build_eval_figure_pack.py`. Prefer keeping them equal, or set `RUN_ID="$TAG"` when customizing tags.

The script kills stale GPU jobs at start/end and waits for `NEED_MIB` (default **14000**) before each VLM phase when `MOCK_LLM=0`.

Post-run only:

```bash
uv run python scripts/build_eval_figure_pack.py --run-id eval_smoke_YYYYMMDD_HHMMSS
# When TAG differed from RUN_ID during the smoke:
uv run python scripts/build_eval_figure_pack.py --run-id "$RUN_ID" --artifact-tag "$TAG"
```

## Success criteria (smoke)

OVMM find-phase episodes are **pick/place localization** tasks: move `{object}` from `{start_recep}` (object on that receptacle) to `{goal_recep}`. Metrics follow the [OVMM](https://ovmm.github.io/) find phases:

| Phase | JSON field | Meaning |
|-------|------------|---------|
| FindObj | `find_object_success` | Localized the target object on `start_recep` |
| FindRec | `find_recep_success` | Localized the **goal** receptacle (`goal_recep`) for placement |
| Partial | `find_partial_success` | Mean of FindObj and FindRec |

**Difficulty:** FindObj is usually **easier** than FindRec (real OVMM paper: ~70% vs ~30%). Object-only success means the agent found the object on `start_recep` but not the goal receptacle — useful partial signal, not full task progress. For smoke, treat **FindRec** as the harder bar; `find_both_success_rate` in `summary.json` is the strictest aggregate.

| Track | Metric | Minimum |
|-------|--------|---------|
| HM-EQA | MCQ accuracy (3 Qs) | **> 0%** (≥1 correct) |
| OVMM Habitat | FindObj **or** FindRec (FindRec preferred) | **> 0** on ≥1 episode × backend |
| SQA3D | EM@1 (3 Qs) | **> 0%** when phase 3 runs |
| All | Artifacts | `topdown_map.png` + `diagnostics_manifest.json` per completed episode |

If HM-EQA, OVMM, **and** SQA3D (when present) metrics are all zero, `build_eval_figure_pack.py` sets `"status": "INVESTIGATE"` in `summary.json`.

## Per-track quick commands

### HM-EQA (interactive QA)

```bash
.venv-habitat/bin/emet-habitat run-batch \
  --method dynagraph --question-ids 3,14,17 \
  --export-map --export-video --resume \
  --output ~/.cache/habitat_eqa/results/smoke_dynagraph.jsonl
```

### OVMM find-phase (Habitat) — includes **dynamem**

```bash
.venv-habitat/bin/emet-habitat run-ovmm-find-batch \
  --backend dynamem --run-tag smoke_ovmm \
  --export-map --export-video \
  --output-dir ~/runs/emet/ovmm_habitat/smoke_dynamem
```

### SQA3D

```bash
uv run emet sqa3d run-real-sweep --split val --question-start 0 --question-end 2 \
  --method dynagraph --replay-mode sens --no-download \
  --export-root ~/.cache/habitat_eqa/episodes/smoke_sqa3d
uv run emet sqa3d plot-results -p ~/runs/emet/sqa3d/dynagraph_val_q0-2.jsonl -o /tmp/sqa3d_figs
```

## Method coverage

| Backend | HM-EQA | OVMM Habitat | SQA3D |
|---------|--------|--------------|-------|
| `static_graph` | yes | yes | — |
| `dynagraph` | yes | yes | yes |
| `dynamem` | follow-up PR | **yes** | yes |

## Paper outputs

Figure pack (`build_eval_figure_pack.py`) writes:

| File | Contents |
|------|----------|
| `summary.json` | HM-EQA accuracy; OVMM FindObj / FindRec / both / object-only rates per backend; SQA3D `em@1` per method; `status` (`OK` or `INVESTIGATE`) |
| `summary.csv` | Long-form metrics for tables (includes `sqa3d` rows when phase 3 ran) |
| `topdown_map_grid.png` | All episode top-down maps |
| `ovmm_findobj_findrec.png` | FindObj vs FindRec bar chart per backend |

**Top-down map colors:** green = explored free space, red = explored obstacle, white = unmapped margin, yellow dot = robot. Eval exports crop to the explored footprint on a white background (not the dark-gray “unknown” fill used for Discord sharing on the full 1024×1024 grid). Maps that look mostly red within the explored blob usually mean depth marked most observed cells as obstacles (common with `explore_steps: 0` Habitat runs); mostly green means few obstacles in the explored region.

OVMM section of `summary.json` also lists per-episode `outcome`: `both`, `object_only`, `recep_only`, or `neither`.

Copy into `paper/figures/eval_smoke/` and cite in `paper/sections/05_results.tex`.
