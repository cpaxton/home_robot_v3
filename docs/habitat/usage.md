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

# GraphEQA with local Qwen3-VL-8B int4 (GPU). Defaults come from ``dynav_config.yaml``
# (``eqa.vl_family: qwen3_vl``, ``Qwen/Qwen3-VL-8B-Instruct``). Override with
# ``--eqa-vl-family`` / ``--eqa-hf-model-id`` for ablations.
uv run emet run graph-eqa-habitat \
  --method graph_eqa \
  --question-id 0 \
  --max-planning-steps 20 \
  --device cuda

# HM-EQA paper batch (113 questions, indices 0–112; budget depends on GPU)
.venv-habitat/bin/emet-habitat run-batch \
  --method graph_eqa \
  --paper-subset \
  --device cuda \
  --resume \
  --output ~/.cache/habitat_eqa/results/graph_eqa_qwen3_vl8b_paper.jsonl

# Full Explore-EQA CSV (500 questions) — use only if you need the extended set
# .venv-habitat/bin/emet-habitat run-batch --all-questions --question-end 499 ...

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

## Debug artifacts (batch runs)

Each `run-batch --output …/my_run.jsonl` writes:

| Path | Contents |
|------|----------|
| `my_run.jsonl` | One JSON line per episode (scores + summary debug fields) |
| `my_run_manifest.json` | Run config: method, VLM id, question ids, git commit, frontier flags |
| `my_run.log` | Full stdout/stderr when using `tee` (see `scripts/run_habitat_frontier_experiments.sh`) |
| `~/.cache/habitat_eqa/episodes/my_run/q<id>_<method>/` | Per-episode bundle (below) |

Per-episode bundle (`metrics.json`, `raw_eqa.txt` full text, `eqa_history.json`, `scene_graph_report.txt`, `frontier_nodes.json`). Optional full graph checkpoint:

```bash
export HABITAT_EQA_EXPORT_GRAPH=1   # adds graph_checkpoint/ (frames, graph.json, …)
```

JSONL fields useful for triage: `parsed_answer_letter`, `frontier_nodes`, `eqa_iterations`, `eqa_action`, `debug_bundle_dir`, `error`. Re-grade a row:

```bash
python3 -c "
import json
from emet.habitat.metrics import extract_mcq_letter_from_raw_eqa, grade_mcq_answer
from pathlib import Path
row = json.loads(Path('~/.cache/habitat_eqa/results/my_run.jsonl').expanduser().read_text().splitlines()[0])
raw = Path(row['debug_bundle_dir']) / 'raw_eqa.txt'
print(grade_mcq_answer(extract_mcq_letter_from_raw_eqa(raw.read_text(), row['choices']), row['gold_answer_letter']))
"
```

## Methods

| `--method` | Memory config | Paper baseline |
|------------|---------------|----------------|
| `graph_eqa` | `dynagraph_merge_xy_m=0`, `dynagraph_staleness_horizon=0` | GraphEQA paper settings |
| `dynagraph` | Same graph settings as `graph_eqa` | Same EQA stack; exercises `DynagraphController` (rerun / `maintain()` noop here) |

On HM-EQA both methods should give **the same accuracy** (within VLM sampling noise). Dynagraph is a regression check, not a competing benchmark config. Long-horizon merge/staleness (`0.45m` / `256` steps) is for real-robot Dynagraph runs, not this short Habitat harness.

## Frontier exploration (fluid map vs graph nodes)

Both methods use the **same voxel map** for navigation. The **fluid frontier** is the boundary of reachable free space on the 2D grid (`sample_exploration` / `sample_frontier` in `voxel_map_dynamem.py`).

**Frontier v2** (default in `dynav_config.yaml`) additionally mirrors unexplored clusters as **graph frontier nodes** (`sync_frontier_nodes`) so the VLM can pick them in EQA `IMAGE_DESCRIPTIONS` and `action:` fields. See [dynagraph.md](../dynagraph.md#graph-frontier-nodes-eqa-guided).

| Config | CLI | Effect |
|--------|-----|--------|
| Paper-style fluid only | `--no-frontier-nodes --frontier-keyword-weight 0` | Time-heuristic grid frontier (closest to upstream fluid policy) |
| Question-biased fluid | `--no-frontier-nodes --frontier-keyword-weight 2` | Keyword overlap on grid cells, no graph nodes |
| Frontier v2 (default) | `--frontier-nodes --frontier-keyword-weight 2` | Graph nodes + keyword bias + fluid fallback |

**Ablation batch** (Q0–19, Qwen2.5-VL-3B default):

```bash
./scripts/run_habitat_frontier_ablation.sh
uv run python scripts/summarize_frontier_ablation.py --q-start 0 --q-end 19
```

Run a single arm: `ARM=fluid_kw ./scripts/run_habitat_frontier_ablation.sh`

Manifest JSONL sidecars record `graph_eqa_frontier_nodes` for each batch.

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
