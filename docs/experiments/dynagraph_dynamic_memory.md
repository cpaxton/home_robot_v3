# Dynagraph dynamic memory — experiment operator notes

Tracks from the paper plan (`paper/sections/04b_dynamic_exploration.tex`) for **find objects** and **update dynamic scenes** on Robocasa + MolmoSpaces.

## Code landed (2026-07-20)

| Piece | Location |
|-------|----------|
| Invalidate + clear EQA cache | `GraphEQAMemory.invalidate_nodes_near`, `clear_eqa_working_memory` |
| Phase 2 world-change uses invalidate | `dynamic_exploration_runner.run_world_change_episode` |
| Lifelong checkpoint patch after fuzz | `invalidate_checkpoint_nodes_near_moves` |
| Explore `graph_health` fields in CSV | `flatten_eval_metrics` + WARN on empty/thin graphs |
| OVMM find backend matrix | `scripts/run_ovmm_find_backend_matrix.py` |
| Agent FIND after relocate | `scripts/smoke_dynagraph_agent_world_change_find.py` |
| Full GPU queue (OVMM → dynamic full) | `scripts/run_dynagraph_dynamic_memory_eval.sh` |
| CPU chain after OVMM | `scripts/chain_dynagraph_cpu_smokes.sh` |

## Environment notes (Robocasa / Molmo smokes)

- Unset ROS ``PYTHONPATH`` when launching evals (``env -u PYTHONPATH …``) so a broken system ``cv2`` cannot shadow the venv.
- Do **not** install ``opencv-python`` alongside ``opencv-contrib-python`` (cv2 becomes a namespace stub). ``pyproject.toml`` overrides block the non-contrib packages.
- After ``uv sync``, reinstall editable sim forks with ``uv pip install --no-deps -e third_party/robosuite -e third_party/robocasa -e third_party/robosuite_models`` and ensure ``gymnasium`` is present (``uv pip install 'gymnasium>=0.29,<1'``).
- ``scripts/run_dynagraph_dynamic_memory_eval.sh`` waits for free VRAM and does **not** run ``gpu_preflight --kill-stale`` (that would pkill in-flight ``mujoco_server``).

## Unit gates (CI)

```bash
uv run emet test \
  src/test/memory/test_dynagraph_known_scene_attach.py \
  src/test/memory/test_dynagraph_staleness_disappearance.py \
  src/test/eval/test_lifelong_checkpoint_invalidate.py \
  --no-sim -q
```

## Fill paper tables

1. Free GPU: `NEED_MIB=12000 ./scripts/gpu_preflight.sh --wait`
2. `nohup ./scripts/run_dynagraph_dynamic_memory_eval.sh &`
3. Aggregates under `~/runs/emet/dynamic_exploration/` and OVMM matrix JSON
4. Copy numbers into `paper/sections/05_results.tex` tables `tab:dynamic_explore_*` and OVMM find tier table

## Smoke battery merge gate

```bash
nohup ./scripts/run_simulation_smoke_battery.sh \
  >> ~/runs/emet/simulation_smoke/nohup.log 2>&1 &
```

Tracks 3–4 (find) and 6–7 (dynamic) exercise the same memory stack.

## Verified this branch (2026-07-20)

| Check | Result |
|-------|--------|
| Unit: known-scene attach + invalidate + lifelong checkpoint | pass |
| Robocasa OVMM GT oracle S1 | `find_partial_success=1.0`, `localization_err=0` (`gt_placement`) |
| Molmo OVMM GT oracle S2 | `find_partial_success=1.0` |
| Agent FIND after relocate | `pass=true`, `n_pruned=2`, did not reuse old pose |
| OVMM perception backends | use explore-budget aliases (`*_exploreN`) + rotate; query labels strip instance hashes |
| Full GPU matrix / dynamic explore | `./scripts/run_dynagraph_dynamic_memory_eval.sh` (flock; OVMM then dynamic; process-group kill on timeout) |

### Overnight failure fixes (2026-07-21)

1. **Serialize GPU jobs** — launcher holds `eval.lock`, runs OVMM then dynamic-explore in one process (no concurrent relaunch).
2. **Kill process groups** — `run_logged_subprocess` uses `start_new_session` + `killpg`; stale-log escalate via `EMET_DYNAMIC_EXPLORE_STALE_KILL_S`.
3. **EQA wall-clock** — `EMET_EQA_QUESTION_TIMEOUT_S` (default 900s) aborts stuck GraphEQA nav/VLM loops.
4. **OVMM mapping/query** — perception matrix forces explore aliases + `--explore-steps`; `semantic_label_from_instance` strips Molmo hashes (`bowl_abc…` → `bowl`); Robocasa episode object is `jar`.

## Scene map cache (skip live explore)

Prebuild each scene's baseline once (perfect-depth rotate + frontier explore), then OVMM find / dynamic-explore P1 / world-change / lifelong cycle-0 load `graph.json` + `voxel_map.pkl` and skip mapping.

```bash
# Build (GPU recommended)
NEED_MIB=8000 ./scripts/gpu_preflight.sh --wait
env -u PYTHONPATH uv run python scripts/build_scene_map_cache.py

# Optional HF share
export EMET_SCENE_MAP_HF_REPO=org/emet-scene-maps
uv run python scripts/scene_map_cache_hub.py push <key>
uv run python scripts/scene_map_cache_hub.py pull <key>

# Consumers use cache by default; force live mapping with:
#   EMET_USE_SCENE_MAP_CACHE=0  or  --no-scene-cache
```

Cache root: `~/.cache/emet/scene_maps/<key>/` (`EMET_SCENE_MAP_CACHE_DIR`). Module: `emet.eval.scene_map_cache`.
