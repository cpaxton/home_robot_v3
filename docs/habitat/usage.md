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
  --method static_graph \
  --question-id 0 \
  --max-planning-steps 20 \
  --device cuda

# Historical emet q0–112 batch (113 questions; not GraphEQA's filtered set)
.venv-habitat/bin/emet-habitat run-batch \
  --method static_graph \
  --paper-subset \
  --device cuda \
  --resume \
  --output ~/.cache/habitat_eqa/results/graph_eqa_qwen3_vl8b_paper.jsonl

# Preflight: uv run emet eval kill-stale && NEED_MIB=12000 uv run emet eval wait
# Or use helpers (resume + SUMMARY; they source gpu_preflight.sh which delegates to emet eval):
#   ./scripts/run_hmeqa_memory_confirm_gate.sh   # GE-only gate → optional annotated37 → paper113
#   ./scripts/run_hmeqa_annotated37_h2h.sh       # ~37 semantics-annotated ids, both methods
#   ./scripts/run_hmeqa_paper113_h2h.sh          # historical q0–112 head-to-head
# Slice taxonomy: docs/habitat/data.md and docs/experiments/habitat_eqa_results.md

# Full Explore-EQA CSV (500 questions) — use only if you need the extended set
# .venv-habitat/bin/emet-habitat run-batch --all-questions --question-end 499 ...

# GraphEQA vs Dynagraph on the same questions (smoke with mock LLM first)
.venv-habitat/bin/emet-habitat compare-batch \
  --question-start 0 --question-end 5 \
  --mock-llm \
  --output ~/.cache/habitat_eqa/results/compare_mock_q0-5.json

# Force navigation each planning step (mock LLM returns confidence:false)
.venv-habitat/bin/emet-habitat run-episode \
  --question-id 3 --method dynagraph \
  --mock-llm --mock-llm-explore \
  --max-planning-steps 5 --export-map --export-video
```

`--mock-llm` uses fixed EQA responses for smoke tests and CI (no OpenAI / Gemini key). With **`--mock-llm-explore`** (requires `--mock-llm`), the mock returns `confidence: false` every iteration so the agent runs the full nav loop without a real VLM — useful for movement / diagnostics smoke, not for grading accuracy.

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
| `--method` | `dynagraph` (`run-episode`); `static_graph` (`run-batch`) | `static_graph` (alias `graph_eqa`) or `dynagraph` |
| `--mock-llm` | off | Smoke / CI without real LLM |
| `--mock-llm-explore` | off | With `--mock-llm`: mock `confidence: false` so each planning step navigates |
| `--max-planning-steps` | `20` | Exploration budget |
| `--max-movement-step` | `10` | Nav substeps per planning iteration |
| `--hm3d-root` | `HM3D_SCENE_DIR` | Override train scene directory |
| `--data-dir` | `HABITAT_EQA_DATA_DIR` | Override CSV location |
| `--output` | none (`run-episode`); required (`run-batch`) | Write episode JSONL |
| `--rotate-in-place` | on | Initial heading sweep before the planning loop |
| `--export-map` / `--no-export-map` | on | Per-episode top-down map bundle (see [evaluation.md](../evaluation.md)) |
| `--export-video` / `--no-export-video` | on | Head-camera `episode_rgb.mp4` |
| `--map-stride` | `0` | Save `maps/step_NNNN.png` every N planning steps (`0` = auto stride when map video on) |
| `--habitat-perfect-nav` / `--no-habitat-perfect-nav` | on | Navmesh pathing vs voxel A* |
| `--use-hm3d-semantics` / `--no-hm3d-semantics` | auto for direct Habitat CLI; off in `emet hmeqa` | Use GT-derived HM3D semantic labels |
| `--enrich-labels` / `--no-enrich-labels` | off | Independently seed per-question GT-derived GraphEQA object hints |
| `--eqa-vl-quantization` | config default (`int4`) | Explicit local VLM quantization override |

## Debug artifacts (batch runs)

`--resume` validates the immutable batch manifest before skipping completed
rows. It rejects method/ID/model/quantization changes, effective parameter or
behavior-environment drift, git/dirty-state drift, changed CSV hashes, a changed
HM3D root, and historical JSONL files with no manifest. Without `--resume`,
`run-batch` refuses to append to an existing output or replace its manifest.

Each `run-batch --output …/my_run.jsonl` writes:

| Path | Contents |
|------|----------|
| `my_run.jsonl` | One JSON line per episode (scores + summary debug fields) |
| `my_run_manifest.json` | Frozen method/IDs/model, effective parameter + environment fingerprints, git state, CSV hashes, HM3D root |
| `my_run.log` | Full stdout/stderr when using `tee` (see `scripts/run_habitat_frontier_experiments.sh`) |
| `~/.cache/habitat_eqa/episodes/my_run/q<id>_<method>/` | Per-episode bundle (below) |

Per-episode bundle (`metrics.json`, `raw_eqa.txt` full text, `eqa_history.json`, `scene_graph_report.txt`, `frontier_nodes.json`). Graph-memory episodes also write `attempt_ledger.json` and `room_events.json`; shadow/agent graph-evidence modes additionally write `world_evidence.json` and, when enabled, `world_evidence_views/`. Set `EMET_EVAL_EXPORT_COMPACT_MEMORY=1` to write a graph-only `compact_memory/` checkpoint with semantic graph/runtime/evidence metadata but no voxel map, dense frames, navigation pixels, or evidence-view pixels. It can be reloaded for semantic inspection; it cannot reproduce post-hoc visual verification. H2H manifest schema v4 freezes the artifact and action-progress profiles. A unit becomes resumable only after its exact expected source bundle passes schema/content and symlink-containment checks, is copied and hashed in staging, and an atomic `OUT/bundles/<arm>_q<id>/COMPLETE.json` marker is published. Object-crop mosaics remain best-effort when no usable instance crops exist; a mosaic declared by `diagnostics_manifest.json` is still required to validate.

With diagnostics (default on via `EMET_EVAL_EXPORT_MAP`; see [evaluation.md](../evaluation.md)): `topdown_map.png`, `topdown_gt_navmesh.png`, `topdown_map_overlay.png`, `maps/overlay_step_*.png`, `topdown_exploration.mp4`, `obstacles_2d.npy`, `trajectory.jsonl`, `nav_attempts.jsonl` (includes navmesh `path_xy` when available), motion-paced `episode_rgb.mp4`, `diagnostics_manifest.json`. Optional full graph checkpoint:

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

| `--method` | Profile | Controller | Paper role |
|------------|---------|------------|------------|
| `static_graph` (alias `graph_eqa`) | `static_graph` (merge=0, staleness=0; no Dynagraph extras) | `GraphEQAController` | GraphEQA-inspired baseline |
| `dynagraph` | `unified_eqa` (0.45 m merge / staleness 256) + memory on, debias off, explore conservative, SigLIP | `DynagraphController` | **Our method** |

Config source: [`configs/benchmarks/dynagraph.yaml`](../../configs/benchmarks/dynagraph.yaml) via `apply_habitat_eqa_method_parameters`. Do **not** report Dynagraph under zero-merge / `smoke`. Cite episode **harness fingerprints** when quoting accuracy.

**Compare-batch** runs both methods on the same question ids; accuracy need not match (different merge + extras).

## Navigation (Habitat-only)

The Habitat runner enables navmesh pathing and frontier exploration by default (`packages/emet_habitat/emet_habitat/runner.py` → `_configure_habitat_nav`). These **`eqa:`** keys apply to HM-EQA only (Robocasa / ZMQ GraphEQA is unchanged unless you set them explicitly):

| Key | Default (Habitat runner) | Effect |
|-----|--------------------------|--------|
| `habitat_perfect_nav` | `true` | Follow HM3D navmesh via `habitat_navmesh_navigate` instead of voxel A* |
| `habitat_explore_frontiers` | `true` | When uncertain, prefer frontier targets over spin-in-place |
| `image_nav_min_approach_m` | `0.35` | Standoff distance when navigating to an **Image N** target and the robot is already at the capture viewpoint |

**Image-N routing:** when the VLM outputs `action: <image id>`, `GraphEQAMemory` resolves a **navigation waypoint** (capture viewpoint or standoff toward the object anchor), not the raw object centroid — this avoids spin-in-place at already-visited poses. See `src/emet/memory/graph_eqa/eqa/graph_nav.py` (`_navigation_waypoint_for_obs`). Package map: [graph_memory.md](../graph_memory.md).

**Mapping overrides (HM-EQA only):** `_configure_habitat_mapping` sets `max_depth=4.5` and, temporarily, **no obstacle dilation** (`pad_obstacles=0`, `filters.smooth_kernel_size=0`) so open-plan doorways are less likely to seal under morphological padding. Real-robot / Robocasa dynav defaults are unchanged. Restore prior Habitat padding with `EMET_HABITAT_PAD_OBSTACLES=1` (and non-zero smooth follows when pad &gt; 0). Frontier graph nodes also snap to a **reachable cell adjacent to the cluster** (not the arc centroid), so explore goals sit on the free-space rim instead of mid-floor.

Override: add under `eqa:` in [`src/emet/config/dynav_config.yaml`](../../src/emet/config/dynav_config.yaml) (the Habitat runner loads this file; `setdefault` in the runner will not clobber an explicit value):

```yaml
eqa:
  image_nav_min_approach_m: 0.5
```

Or uncomment / set in [`src/emet/config/mapping/default.yaml`](../../src/emet/config/mapping/default.yaml) when using unified config presets. The Habitat runner loads `dynav_config.yaml` directly today. See [emet_config.md](../emet_config.md).

**CLI:** disable navmesh with `--no-habitat-perfect-nav`. Via `emet run graph-eqa-habitat`, extra flags (e.g. `--mock-llm-explore`, `--export-video`) forward to `emet-habitat run-episode`.

**Dev helper:** `scripts/probe_habitat_eqa_exploration_need.py` scans question ids and reports which need navigation after the initial look-around (good demo picks vs spin-only questions).

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
