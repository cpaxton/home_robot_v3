# Dynamic exploration benchmark

Paper section: `paper/sections/04b_dynamic_exploration.tex`. Batch harness for **active frontier exploration** (Phase 1), **scripted world-change / staleness** episodes (Phase 2), and **lifelong K-cycle checkpoint/fuzz/reload** episodes (Phase 3) on Stretch in Robocasa (S1) and MolmoSpaces iTHOR train (S2).

**Index:** [experiments/dynamic_exploration.md](experiments/dynamic_exploration.md). For a single-scene backend localization figure on the same Robocasa stack, see [experiments/backend_localization.md](experiments/backend_localization.md).

## Quick start

```bash
# Config + dry-run matrix
uv run python scripts/eval_dynamic_exploration.py --dry-run

# Phase 1: one Robocasa seed, short explore budget (smoke)
uv run python scripts/eval_dynamic_exploration.py --smoke --output-dir /tmp/dynamic_explore_smoke

# Same as --smoke (explicit flags):
uv run python scripts/eval_dynamic_exploration.py \
  --phase explore --env robocasa --episode-id robocasa_seed0 \
  --backend dynagraph --explore-max-iters 3 --mapping-mode explore \
  --output-dir /tmp/dynamic_explore_smoke

# Graph dedup spot-check: sim + explore-loop + --compare-to-gt (creates export dir before logging)
uv run python scripts/run_dedup_sim_validate.py --export-dir /tmp/dedup_sim_validate

# Phase 1: rotate-only contrast row (OVMM-style, no explore-loop)
uv run python scripts/eval_dynamic_exploration.py \
  --phase explore --episode-id robocasa_seed0 \
  --mapping-mode rotate_only --backend dynagraph --cpu-only

# Phase 2: world-change (Robocasa obj_main relocation)
uv run python scripts/eval_dynamic_exploration.py \
  --phase world-change --episode-id robocasa_seed0_world_change \
  --backend dynagraph --cpu-only

# Lifelong: K-cycle checkpoint → fuzz → reload episodes
uv run python scripts/eval_dynamic_exploration.py \
  --phase lifelong --episode-id robocasa_seed0_lifelong \
  --backend dynagraph --cpu-only

# Full paper matrix (standalone or via large eval queue)
./scripts/run_dynamic_exploration_full.sh
# same as: ./scripts/run_large_paper_eval.sh dynamic-explore

# Included in full paper queue (after SQA3D + OVMM unless skipped)
SKIP_DYNAMIC_EXPLORE=1 ./scripts/run_large_paper_eval.sh   # skip
./scripts/run_large_paper_eval.sh                          # runs all three tracks
```

Env defaults (set by harness): `EMET_SIM_NAV_TELEPORT=1`, `EMET_ZMQ_STARTUP_TIMEOUT=120`, `MUJOCO_GL=egl`.

**Nav frame (important):** Dynagraph plans in MuJoCo **world** coordinates (`execute_trajectory(..., world_frame=True)` / `nav_world`). ZMQ `gps`/`base_pose` are **episode-relative** to `navigation_origin_xyt`. `StretchZmqClient.wait_for_waypoint` / `_wait_for_base_motion` must compare in the world frame when `world_frame=True`; otherwise teleports succeed on the server but the client burns 10s timeouts and exploration barely moves (`explored_fraction` stays ~0.1). With teleport enabled, client waits are capped at ~3s.

MolmoSpaces requires `.venv-molmospaces` (`./install.sh --molmospaces -y`).

## Config

| File | Role |
|------|------|
| `configs/benchmarks/dynamic_exploration.yaml` | Episodes, explore budgets, output path, world-change episodes |
| `src/emet/config/benchmarks/dynagraph_questions.yaml` | Per-seed EQA questions + world-change pre/post |
| `configs/benchmarks/dynagraph.yaml` | Profiles: `interactive` (Dynagraph) vs `graph_eqa_baseline` |

Default output: `~/runs/emet/dynamic_exploration` (override with `--output-dir` or `EMET_DYNAMIC_EXPLORE_OUTPUT`).

## Phase 1 — Active exploration

Per run: start sim → `emet run dynagraph` with optional `--explore-loop --explore-max-iters K` → export → `compute_dynagraph_eval` → `{run_id}.json`.

**Contrast row:** `mapping_mode=rotate_only` uses rotate-in-place mapping only (`explore_max_iters=0`, no explore-loop). Same env/backend as explore-loop rows for paper Table `tab:dynamic_explore_phase1`.

**Primary metrics** (in `aggregate_dynamic_exploration.csv`):

| Column | Source |
|--------|--------|
| `explored_fraction`, `explored_area_m2` | eval `explore` |
| `spatial_recall`, `label_recall` | eval `fusion` |
| `node_count`, `edge_count` | eval `graph` |
| `eqa_accuracy` | eval `eqa` + question bank |
| `episode_wall_s` | harness timer |

Backends: `dynagraph` (interactive profile) vs `graph_eqa` (merge/staleness off via `graph_eqa_baseline`).

## Phase 2 — World-change

Robocasa only (v1): explore → EQA pre → ZMQ `sim_set_body_pose` on `obj_main` → recovery explore → EQA post → export.

Metrics in `aggregate_dynamic_exploration_world_change.csv`: `answer_correct_pre/post`, `n_stale_nodes_after_move`, `n_pruned_by_maintain`, `recovery_steps`, `localization_err_m`.

Sim API: ZMQ `sim_set_body_pose` (same as [ovmm_full_benchmark.md](ovmm_full_benchmark.md)); client helper `robot_zmq_set_body_pose()` in `sim_manipulation.py`.

## Lifelong — K-cycle checkpoint/fuzz/reload

One sim server stays alive for the whole episode. Each cycle runs `emet run dynagraph`
in a **fresh subprocess** that reloads the previous checkpoint (`--input-path`), explores
briefly, answers the cycle's questions, and exports checkpoint `cycle_t`. Between cycles
the world is fuzzed over ZMQ and moved bodies are verified against live GT placements.

- **Checkpoints** (`exports/{run_id}/cycle_t/`): `graph.json` now persists `last_seen`,
  `support_count`, `is_viewpoint`, `extent_half`, `bounds_3d` per node; `manifest.json`
  records the final controller step (`final_step`); `voxel_map.pkl` restores obstacles /
  explored area (enable on any dynagraph run with `--export-voxel-pickle`). On reload the
  controller resumes `obs_count` so staleness `maintain()` does not prune the resumed graph.
- **Fuzzing** (`src/emet/eval/world_fuzz.py`): scripted per-cycle `moves:` (body + `delta`
  or absolute `pos`) and `doors:` (joint + qpos value) from the `lifelong:` config section;
  seeded-random mode picks freejoint bodies from GT placements and `*doorhinge*` joints.
- **Sim API**: object teleports use ZMQ `sim_set_body_pose`; door/drawer joints use the
  ZMQ `sim_set_joint_qpos` action (`robot_zmq_set_joint_qpos()` in `sim_manipulation.py`,
  Stretch server only). The physics subprocess acks with the qpos measured on the live
  model, logged as `sim_set_joint_qpos: <joint> requested=… measured=…`.
- **Config** (`lifelong:` in `configs/benchmarks/dynamic_exploration.yaml`): episodes with
  `cycles: K`, per-cycle `questions` (optional; omit for GPU-light smoke runs without VLM
  scoring), per-boundary `changes`, and `explore_iters_first` / `explore_iters_resume`.

Metrics per cycle in `{run_id}.json` and `aggregate_dynamic_exploration_lifelong.csv`:
`eqa_accuracy` (scored against live GT each cycle), `object_node_count` / `total_node_count`, and per-cycle **`graph_health`** (`n_object`, `singleton_frac`, `mean_support`, `failure_class`, …). Triage: `uv run python scripts/summarize_graph_health.py RUN/lifelong.json`.
(graph churn), and per-move adaptation flags (`adapted` = node appears near the new
position, `stale` = node lingers at the old one, within 0.75 m).

## Resume

Pass `--resume` to skip runs whose `{run_id}.json` already exists under the output directory.

Each Phase 1 run writes `{export_dir}/dynagraph.log` (full ``emet run dynagraph`` stdout) for debugging timeouts.

**Timing:** GPU Robocasa smoke (`--smoke`, K=3) is typically **60–75 min** (rotate + explore + EQA/VLM). The harness scales subprocess timeout with explore budget (~105 min for K=3); override with `EMET_DYNAMIC_EXPLORE_DYNAGRAPH_TIMEOUT_S` if needed. `--cpu-only` roughly doubles wall time.

**Progress / failure logs** (under `--output-dir`):

| File | Contents |
|------|----------|
| `runner.log` | Matrix-level start/skip/ok/fail lines with wall time |
| `progress.jsonl` | Structured events (`matrix_start`, `run_start`/`run_end`, subprocess heartbeats, `stale_log`) |
| `exports/<run_id>/dynagraph.log` | Full `emet run dynagraph` stdout/stderr |

Heartbeats every `EMET_DYNAMIC_EXPLORE_HEARTBEAT_S` (default 120s). If `dynagraph.log` stops updating for `EMET_DYNAMIC_EXPLORE_STALE_LOG_S` (default 900s), the harness prints a `STALE_LOG` warning with a log tail — useful when EQA hangs after VLM load.

## Tests

```bash
uv run emet test src/test/eval/test_dynamic_exploration_config.py -q
RUN_DYNAMIC_EXPLORE_TESTS=1 uv run emet test src/test/eval/test_dynamic_exploration_config.py -q
```

Environment variables (`EMET_DYNAMIC_EXPLORE_OUTPUT`, `SKIP_DYNAMIC_EXPLORE`, …): [environment_variables.md](environment_variables.md).  
Full multi-track queue: [paper_benchmarks.md](paper_benchmarks.md) (Large paper eval queue).
