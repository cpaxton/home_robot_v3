# Cross-track smoke validation

One-episode (or unit-test) sanity check per paper track before multi-day GPU sweeps.
See [experiments README](README.md) and [evaluation runbook](../evaluation.md).

**Last validated:** `unified_pr68_smoke` (2026-06-28) — see also the canonical **seven-track simulation battery** in [simulation_testing_plan.md](../simulation_testing_plan.md).

**Prior overnight run:** `cross_track_20260628_012931` — tracks 0–4 **PASS**; track 5 import bug fixed; pytest uses `gpu_preflight.sh` ignore list.

## Seven-track simulation battery (paper-facing)

Sequential embodied validation before multi-day sweeps. Full commands: **[simulation_testing_plan.md](../simulation_testing_plan.md)**.

```bash
./scripts/run_simulation_smoke_battery.sh
# or: nohup ./scripts/run_simulation_smoke_battery.sh >> ~/runs/emet/simulation_smoke/nohup.log 2>&1 &
```

| # | Track | Script step |
|---|-------|-------------|
| 1 | Habitat EQA | `emet-habitat run-episode` Q17 |
| 2 | Habitat OVMM | `emet-habitat run-ovmm-find-episode` GT |
| 3 | Robocasa search | `eval_ovmm_find_phases.py` S1 |
| 4 | MolmoSpaces / iTHOR search | `eval_ovmm_find_phases.py` S2 |
| 5 | SQA3D | `emet sqa3d run-episode` mock-LLM |
| 6 | Robocasa dynamic env | `eval_dynamic_exploration.py --phase world-change` |
| 7 | MolmoSpaces dynamic search | `eval_dynamic_exploration.py --phase explore` on `molmo_ithor0` |

Logs: `~/runs/emet/simulation_smoke/<RUN_ID>/`.

## Cursor / long agent sessions

Multi-hour Habitat evals and overnight orchestrators should run via **`emet jobs run`** (or a dedicated terminal), not as blocking Cursor agent turns. Empty `nvidia-smi` does **not** mean Habitat EGL is healthy; agent crashes here are usually `emet` segfaults after Habitat/VLM teardown — see [known_issues.md](../known_issues.md#nvidia-driver-hang--cursor-agent-crash-during-stacked-gpu-evals) and `emet eval diagnose`. After a crash, check log files and artifact paths before re-running.

## Orchestrator (recommended — extended overnight)

Runs tier-0 unit tests, tracks 1–5, safe pytest, and optional deep eval. **Prefer the seven-track battery above** for paper-facing simulation validation; use this orchestrator for full overnight regression.

```bash
# Foreground
./scripts/run_overnight_cross_track_smoke.sh

# Background
nohup ./scripts/run_overnight_cross_track_smoke.sh \
  >> ~/runs/emet/overnight_cross_track/nohup.log 2>&1 &
```

Logs: `~/runs/emet/overnight_cross_track/<RUN_ID>/` (`summary.txt` + per-step `*.log`).

| Env | Default | Effect |
|-----|---------|--------|
| `RUN_ID` | `cross_track_YYYYMMDD_HHMMSS` | Log subdirectory name |
| `NEED_MIB` | `12000` | Min free VRAM before GPU tracks |
| `RUN_DEEP_EVAL` | `0` | Set `1` to run `run_overnight_eval_smoke.sh` after cross-track (separate GPU night recommended) |
| `TIMEOUT_DYNAMIC` | `28800` | Track 4 Robocasa explore (seconds) |

**Before starting:** ensure sim deps (`emet install sim`) and Habitat wrapper (`.venv-habitat/bin/emet-habitat`). Optional preflight:

```bash
uv run emet eval kill-stale
NEED_MIB=12000 uv run emet eval wait
```

Deep Habitat eval (HM-EQA + OVMM + SQA3D matrix) — **run on a separate night**, not chained after track 4:

```bash
uv run emet eval kill-stale
./scripts/run_overnight_eval_smoke.sh
```

## Summary (manual / per-track)

| # | Track | Status (2026-06-28) | Pass criterion | Artifact |
|---|-------|---------------------|----------------|----------|
| 0 | Unit tests (focused) | **PASS** (278 tests, PR #68 battery) | pytest green | `tier0_unit_tests.log` |
| 1 | SQA3D | **PASS** | mock-LLM episode; prediction matches gold | train q220602000000 → `brown` |
| 2 | Habitat EQA | **PASS** | non-empty letter + episode completes | `~/.cache/habitat_eqa/results/unified_pr68_smoke_q17.jsonl` |
| 3 | Habitat OVMM find-phase | **PASS** | find_object or find_recep success | `~/runs/emet/ovmm_habitat/unified_pr68_smoke/` |
| 4 | Robocasa explore (Phase 1) | **PASS** (overnight) | explored_fraction > 0, no crash | `~/runs/emet/dynamic_exploration/<RUN_ID>_explore/` |
| 5 | Robocasa world-change (Phase 2) | **retry** | aggregate CSV written | `ImportError: EQAExecuter` fixed in `dynamic_exploration_runner.py` |
| — | Full `--no-sim` pytest | **retry** | no segfault | uses `emet_pytest_no_sim_ignore_args` from `gpu_preflight.sh` |

### Notes

- **Track 4 (robocasa missing):** earlier ~1.4 s failures were `ModuleNotFoundError: No module named 'robocasa'` — fixed with `emet install sim`.
- **Full pytest segfault:** `-m "not sim"` still ran MuJoCo-native tests under `src/test/simulation/`; orchestrator now passes explicit `--ignore` list from [`scripts/gpu_preflight.sh`](../../scripts/gpu_preflight.sh).
- **System freeze (~4:32 AM):** stacking `run_overnight_eval_smoke.sh` immediately after Robocasa + pytest on one GPU session wedged the NVIDIA driver (GUI + SSH dead, mouse still moved). Keep **one GPU-heavy job at a time**; use `RUN_DEEP_EVAL=0` default.

## Commands (repro — single track)

### Tier 0 — focused unit tests (~15 min)

Same set as the orchestrator’s tier 0 step (see `run_overnight_cross_track_smoke.sh`). Includes unified config loader tests (`src/test/config/`) when validating PR #68+ branches.

```bash
uv run emet test src/test/config/ \
  src/test/benchmarks/sqa3d/ \
  src/test/habitat/test_metrics.py src/test/eval/test_dynagraph_vram.py \
  src/test/memory/test_graph_eqa_memory.py src/test/memory/test_mcq_debias.py \
  src/test/memory/test_ovmm_find_phase_metrics.py \
  src/test/memory/test_habitat_ovmm_find_loader.py \
  src/test/eval/test_dynamic_exploration_config.py \
  src/test/eval/test_dynamic_exploration_runner.py \
  src/test/memory/test_memory_backends_smoke.py \
  src/test/app/test_stream_stats.py \
  src/test/memory/test_dynamem_graph_hooks_fusion.py \
  src/test/memory/test_graph_object_fusion_default_yaml.py \
  src/test/eval/test_episode_diagnostics_export.py \
  src/test/eval/test_habitat_cli_diagnostics.py \
  src/test/robots/test_innate_mars_backend.py -q
```

Config loader one-liner (innate_mars overlay):

```bash
uv run python -c "
from emet.core.parameters import get_parameters
p = get_parameters('dynav_innate_mars.yaml')
assert str(p.get('depth_source')).lower() == 'auto'
assert (p.get('graph_object_fusion') or {}).get('bounds_3d_iou_merge_min') == 0.40
print('config_smoke OK')
"
```

### Full no-sim pytest (orchestrator step)

```bash
# shellcheck source=scripts/gpu_preflight.sh
source scripts/gpu_preflight.sh
mapfile -t ignore_args < <(emet_pytest_no_sim_ignore_args)
uv run pytest src/test -m "not sim" --tb=no -q "${ignore_args[@]}"
```

### Track 1 — SQA3D

```bash
uv run emet sqa3d run-episode --split train --mock-llm --question-id 220602000000
```

### Track 2 — Habitat EQA (Q17 regression)

```bash
.venv-habitat/bin/emet-habitat run-episode \
  --question-id 17 --method dynagraph \
  --eqa-vl-family qwen3_vl --eqa-hf-model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda --export-map \
  --output ~/.cache/habitat_eqa/results/smoke_cross_track_q17.jsonl
```

### Track 3 — Habitat OVMM

```bash
.venv-habitat/bin/emet-habitat run-ovmm-find-episode \
  --episode-id hm3d_lamp_bed_00006 \
  --backend ground_truth --cpu-only --not-rotate \
  --output ~/runs/emet/ovmm_habitat/smoke_track3/hm3d_lamp_bed_00006_gt.json
```

### Track 4 — Robocasa explore

Requires `emet install sim`. ~60–75 min with real VLM.

```bash
uv run emet eval kill-stale
uv run python scripts/eval_dynamic_exploration.py --smoke \
  --output-dir ~/runs/emet/dynamic_exploration/smoke_cross_track_v3
```

### Track 5 — World-change (after track 4)

```bash
uv run emet eval kill-stale
uv run python scripts/eval_dynamic_exploration.py \
  --phase world-change --episode-id robocasa_seed0_world_change \
  --backend dynagraph \
  --output-dir ~/runs/emet/dynamic_exploration/smoke_world_change
```

## Next after all smokes pass

1. SQA3D val subset aggregate
2. Habitat hold-out-8 or balanced-32 @ Qwen3-VL-8B (`run_overnight_habitat_eval.sh`)
3. OVMM S0 ladder CSV refresh
4. Dynamic explore full matrix (robocasa + molmo lifelong)

Prior-art tables: [habitat_eqa_results.md](habitat_eqa_results.md).
