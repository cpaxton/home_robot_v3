# Simulation testing plan (seven-track smoke battery)

Canonical **sequential smoke battery** for embodied sim + Habitat before multi-day paper sweeps. Runs one GPU-heavy job at a time with shared preflight ([`emet eval`](cli.md#emet-eval-gpu-preflight--stale-cleanup); bash [`scripts/gpu_preflight.sh`](../scripts/gpu_preflight.sh) delegates).

**Orchestrator:** [`scripts/run_simulation_smoke_battery.sh`](../scripts/run_simulation_smoke_battery.sh)

**Paper:** `paper/sections/04_experiments.tex` (subsection *Simulation smoke battery*).  
**Related:** [evaluation.md](evaluation.md), [experiments/cross_track_smoke.md](experiments/cross_track_smoke.md), [paper_benchmarks.md](paper_benchmarks.md).

---

## Seven tracks (run in order)

| # | Track | Environment | Harness | Pass criterion |
|---|-------|-------------|---------|----------------|
| 1 | **Habitat EQA** | HM3D / HM-EQA | `.venv-habitat/bin/emet-habitat run-episode` | Episode completes; MCQ letter emitted; optional map export |
| 2 | **Habitat OVMM** | HM3D proxy | `.venv-habitat/bin/emet-habitat run-ovmm-find-episode` | `find_partial_success > 0` (GT backend smoke) |
| 3 | **Robocasa search** | Robocasa S1 | `eval_ovmm_find_phases.py` | OVMM FindObj/FindRec on kitchen (`robocasa_pp_s1`) |
| 4 | **MolmoSpaces / iTHOR search** | MolmoSpaces S2 | `eval_ovmm_find_phases.py` | OVMM find-phase on iTHOR train idx 0 |
| 5 | **SQA3D** | ScanNet replay | `emet sqa3d run-episode` | Mock-LLM EM@1 on one train question |
| 6 | **Robocasa dynamic env** | Robocasa S1 | `eval_dynamic_exploration.py --phase world-change` | World-change episode completes (explore + pre/post EQA); JSON has `n_stale_nodes_after_move` / `answer_correct_pre|post` without `error`. Prefers GPU + `prepare_dynagraph_vram_for_eqa` before Qwen. |
| 7 | **MolmoSpaces dynamic search** | MolmoSpaces S2 | `eval_dynamic_exploration.py --phase explore --skip-eqa` | Explore-loop K=3 completes; graph export written (no post-explore VLM EQA in smoke) |

Tracks 1–2 require `./scripts/install_habitat.sh` and HM3D assets. Tracks 3–6 require `emet install sim` (Robocasa). Tracks 4 and 7 require `.venv-molmospaces` (`./install.sh --molmospaces -y`). Track 5 needs ScanNet mesh for full replay (mock-LLM still validates harness wiring).

**Interactive agent (same Dynagraph memory):** for live skill checks outside the battery, use `emet run agent --memory-backend dynagraph` (default) against the same sim servers; see [AGENT_RUN.md](AGENT_RUN.md) skill checklist and [paper_benchmarks.md](paper_benchmarks.md).

---

## One-command battery

```bash
# Foreground (sequential, GPU preflight between GPU steps)
./scripts/run_simulation_smoke_battery.sh

# Background (recommended for Cursor / long sessions)
nohup ./scripts/run_simulation_smoke_battery.sh \
  >> ~/runs/emet/simulation_smoke/nohup.log 2>&1 &
tail -f ~/runs/emet/simulation_smoke/*/summary.txt
```

Logs: `~/runs/emet/simulation_smoke/<RUN_ID>/` (`summary.txt` + `track{N}_*.log` + `inspection_report.md`).

After a run, inspect metrics and visual artifact paths (topdown maps, scene graphs, exports):

```bash
uv run python scripts/inspect_simulation_smoke_battery.py --run-id sim_smoke_agent_20260628 --write-report
# open ~/runs/emet/simulation_smoke/<RUN_ID>/inspection_report.md
```

The inspector checks **semantic** pass criteria (e.g. Habitat `correct`, OVMM `find_partial_success > 0`), not just harness exit codes.

| Env | Default | Effect |
|-----|---------|--------|
| `RUN_ID` | `sim_smoke_YYYYMMDD_HHMMSS` | Log subdirectory |
| `NEED_MIB` | `12000` | Min free VRAM before GPU tracks (1, 3, 5–7 when VLM loads) |
| `MOCK_LLM` | `1` for track 5 | Set `0` for real SQA3D VLM smoke |
| `HABITAT_GPU` | `1` | Track 1 uses CUDA + Qwen3-VL-8B; set `0` for mock-LLM Habitat only |
| `DYNAMIC_GPU` | `1` | Tracks 6–7 use GPU for VLM/perception; set `0` for `--cpu-only` (much slower) |
| `SKIP_TRACKS` | empty | Comma list e.g. `4,7` to skip Molmo legs |
| `TIMEOUT_DYN` | `28800` | Tracks 6–7 timeout (seconds); full world-change with EQA is typically 2–4h on GPU |

---

## Manual commands (per track)

Preflight before GPU tracks:

```bash
uv run emet eval kill-stale
NEED_MIB=12000 uv run emet eval wait
```

### 1 — Habitat EQA

```bash
.venv-habitat/bin/emet-habitat run-episode \
  --question-id 17 --method dynagraph \
  --eqa-vl-family qwen3_vl --eqa-hf-model-id Qwen/Qwen3-VL-8B-Instruct \
  --device cuda --export-map \
  --output ~/.cache/habitat_eqa/results/sim_smoke_hmeqa_q17.jsonl
```

### 2 — Habitat OVMM

```bash
.venv-habitat/bin/emet-habitat run-ovmm-find-episode \
  --episode-id hm3d_lamp_bed_00006 \
  --backend ground_truth --cpu-only --not-rotate \
  --output ~/runs/emet/ovmm_habitat/sim_smoke/hm3d_lamp_bed_00006_gt.json
```

### 3 — Robocasa search (OVMM find-phase S1)

```bash
uv run python scripts/eval_ovmm_find_phases.py \
  --episode-id robocasa_pp_s1 \
  --backend ground_truth --cpu-only --not-rotate \
  --output-dir ~/runs/emet/ovmm_find_phase/sim_smoke_robocasa
```

### 4 — MolmoSpaces / iTHOR search (OVMM find-phase S2)

```bash
uv run python scripts/eval_ovmm_find_phases.py \
  --episode-id molmo_ithor_s2_idx0 \
  --backend ground_truth --cpu-only --not-rotate \
  --output-dir ~/runs/emet/ovmm_find_phase/sim_smoke_molmo
```

### 5 — SQA3D

```bash
uv run emet sqa3d run-episode --split train --mock-llm --question-id 220602000000
```

Real VLM (optional, needs GPU + ScanNet):

```bash
uv run emet sqa3d run-episode --split val --question-id 0 --method dynagraph --replay-mode sens
```

### 6 — Robocasa dynamic env (world-change)

```bash
uv run python scripts/eval_dynamic_exploration.py \
  --phase world-change --episode-id robocasa_seed0_world_change \
  --backend dynagraph --explore-max-iters 3 \
  --output-dir ~/runs/emet/dynamic_exploration/sim_smoke_world_change
```

Use `--cpu-only` only when no GPU is available (roughly doubles wall time).

### 7 — MolmoSpaces dynamic search (explore-loop)

```bash
uv run python scripts/eval_dynamic_exploration.py \
  --phase explore --env molmospaces --episode-id molmo_ithor0 \
  --backend dynagraph --explore-max-iters 3 --mapping-mode explore \
  --skip-eqa \
  --output-dir ~/runs/emet/dynamic_exploration/sim_smoke_molmo_explore
```

`--skip-eqa` omits post-explore question-bank VLM calls (smoke validates mapping only). Full paper sweeps omit `--skip-eqa`.

Extended dynamic smoke (lifelong K-cycle on Molmo): `--phase lifelong --episode-id molmo_ithor0_lifelong`.

---

## Tier 0 unit tests (before or after battery)

Focused `--no-sim` suite (config loader + eval harness wiring):

```bash
uv run emet test src/test/config/ \
  src/test/benchmarks/sqa3d/ \
  src/test/habitat/test_metrics.py \
  src/test/eval/test_dynagraph_vram.py \
  src/test/memory/test_graph_eqa_memory.py \
  src/test/memory/test_mcq_debias.py \
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

---

## Relationship to cross-track overnight smoke

[`run_overnight_cross_track_smoke.sh`](../scripts/run_overnight_cross_track_smoke.sh) adds tier-0 unit tests, full safe pytest, and optional deep Habitat eval. The **seven-track battery** is the paper-facing **simulation validation ladder** documented here and in the paper experiments section — run it before claiming a branch is merge-ready for embodied eval changes.

**Do not chain** this battery immediately after a full Robocasa GPU explore + pytest on the same GPU session. Prefer `nohup` and separate nights for track 6–7 if tracks 3–4 used heavy VLM backends.
