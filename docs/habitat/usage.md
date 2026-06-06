# Running Habitat EQA

## Prerequisites

- [install.md](install.md) — `.venv-habitat` with `habitat-sim`
- [data.md](data.md) — CSVs + HM3D **train** split for HM-EQA questions

## Main CLI (delegates to `.venv-habitat`)

```bash
uv run emet habitat info
uv run emet habitat list-questions --limit 5
uv run emet habitat run-episode --question-id 0 --method dynagraph --mock-llm

# Smoke (mocked LLM)
uv run emet run graph-eqa-habitat --dataset hmeqa --question-id 0 --mock-llm

# GraphEQA reproduction with local Gemma multimodal (GPU; ~first run downloads weights)
uv run emet run graph-eqa-habitat \
  --method graph_eqa \
  --question-id 0 \
  --max-planning-steps 5 \
  --eqa-vl-family gemma4 \
  --device cuda

# HM-EQA batch (113 questions; long — use nohup / tmux)
.venv-habitat/bin/emet-habitat run-batch \
  --method graph_eqa \
  --question-start 0 --question-end 112 \
  --eqa-vl-family gemma4 \
  --device cuda \
  --output ~/.cache/habitat_eqa/results/graph_eqa_gemma4.jsonl

# GraphEQA vs Dynagraph on the same questions (smoke with mock LLM first)
.venv-habitat/bin/emet-habitat compare-batch \
  --question-start 0 --question-end 5 \
  --mock-llm \
  --output ~/.cache/habitat_eqa/results/compare_mock_q0-5.json
```

`--mock-llm` uses fixed EQA responses for smoke tests and CI (no OpenAI / Gemini key).

## Direct wrapper

```bash
.venv-habitat/bin/emet-habitat info
.venv-habitat/bin/emet-habitat list-questions --limit 5
.venv-habitat/bin/emet-habitat run-episode \
  --question-id 0 \
  --method dynagraph \
  --mock-llm \
  --max-planning-steps 3
```

Useful flags:

| Flag | Default | Notes |
|------|---------|-------|
| `--question-id` | `0` | Index into `questions.csv` |
| `--method` | `dynagraph` | `graph_eqa` or `dynagraph` |
| `--mock-llm` | off | Smoke / CI without real LLM |
| `--max-planning-steps` | `3` (wrapper) / `5` (`emet run`) | Exploration budget |
| `--hm3d-root` | `HM3D_SCENE_DIR` | Override train scene directory |
| `--data-dir` | `HABITAT_EQA_DATA_DIR` | Override CSV location |
| `--output` | none | Write episode JSONL |
| `--rotate-in-place` | off | Extra rotation action in loop |

## Methods

| `--method` | Memory config | Paper baseline |
|------------|---------------|----------------|
| `graph_eqa` | `dynagraph_merge_xy_m=0`, `dynagraph_staleness_horizon=0` | GraphEQA paper settings |
| `dynagraph` | Same graph settings as `graph_eqa` | Same EQA stack; exercises `DynagraphController` (rerun / `maintain()` noop here) |

On HM-EQA both methods should give **the same accuracy** (within VLM sampling noise). Dynagraph is a regression check, not a competing benchmark config. Long-horizon merge/staleness (`0.45m` / `256` steps) is for real-robot Dynagraph runs, not this short Habitat harness.

## Tests

Unit tests (main env, no GPU):

```bash
uv run emet test src/test/habitat/ -q
```

Integration smoke (loads Habitat-Sim; optional):

```bash
RUN_HABITAT_TESTS=1 uv run emet test src/test/habitat/ -k smoke
```

Full HM-EQA sweeps are GPU-heavy and not default CI.

## Episode flow

1. Load question + init pose from CSVs  
2. Open HM3D scene via `hm3d_scene_glb_path`  
3. Spawn agent at init pose  
4. Loop: observe → update voxel map / graph → plan → act  
5. Grade multiple-choice answer vs `questions.csv` label  

Metrics helpers live in `src/emet/habitat/metrics.py`.
