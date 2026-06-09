# SQA3D benchmark harness

[SQA3D](https://github.com/SilongYong/SQA3D) (Situated Question Answering in 3D Scenes, ICLR 2023) evaluates embodied agents that must understand a **situation** (where they stand in a ScanNet scene) and answer an open-ended **question** about their surroundings.

This repo provides **dataset loaders**, **EM@1 scoring** (official QA metric), **localization metrics**, **ScanNet replay** (posed `.sens` RGB-D with mesh fallback) for embodied **DynaMem / Dynagraph**, and **figure export** (TP/FP/FN breakdown).

## Quick commands

| Goal | Command |
|------|---------|
| Download annotations | `uv run python scripts/download_sqa3d_data.py --fetch-annotations` |
| Optional localization JSON | `uv run python scripts/download_sqa3d_data.py --fetch-localization` |
| Data status | `uv run emet sqa3d info` |
| Verify downloads + connections | `uv run emet sqa3d verify --run-embodied-smoke` |
| List questions | `uv run emet sqa3d list-questions --split val --limit 5` |
| Score predictions | `uv run emet eval-sqa3d -p preds.jsonl --split val -o sqa3d_eval.json` |
| Score (fixture paths) | `uv run emet eval-sqa3d -p preds.jsonl --questions-path … --annotations-path …` |
| Download ScanNet meshes | `uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00` |
| Download posed RGB-D (`.sens`) | `uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00 --with-sens` |
| Embodied episode | `uv run emet sqa3d run-episode --split train --question-id 220602000000 --mock-llm` |
| Embodied with real ScanNet RGB | `… run-episode … --replay-mode sens` (needs `.sens` on disk) |
| Tuned real-VLM batch | `uv run emet sqa3d run-batch --split val --question-start 0 --question-end 10` (omit `--mock-llm`; uses `--profile tuned`) |
| Unit tests | `uv run emet test src/test/benchmarks/sqa3d/ -v` |
| One-command smoke | `uv run python scripts/run_sqa3d_scannet_smoke.py` |
| Score batch JSONL | `uv run emet eval-sqa3d -p /tmp/sqa3d_batch.jsonl` (auto-detects episode format) |
| Paper figures (TP/FP/FN) | `uv run emet sqa3d plot-results -p /tmp/sqa3d_batch.jsonl -o /tmp/sqa3d_figs` |
| Real VLM sweep | `uv run emet sqa3d run-real-sweep --question-start 0 --question-end 30 --replay-mode sens --no-download` |
| GPU preflight sweep | `./scripts/run_sqa3d_gpu_sweep.sh --split val --question-start 0 --question-end 30 --replay-mode sens` |

CLI reference: [cli.md](cli.md#emet-sqa3d-subcommand).

## Data layout

| Env var | Default | Contents |
|---------|---------|----------|
| `SQA3D_DATA_DIR` | `~/.cache/sqa3d/data` | `sqa_task/balanced/*.json`, `answer_dict.json` |

Splits: `train`, `val`, `test` (Zenodo [record 7792397](https://zenodo.org/record/7792397)).

```bash
uv run python scripts/download_sqa3d_data.py --fetch-annotations
uv run python scripts/download_sqa3d_data.py --verify-split val
```

ScanNet **meshes** are downloaded separately (Terms of Use apply):

| Env var | Default | Contents |
|---------|---------|----------|
| `SCANNET_ROOT` | `~/.cache/scannet` | `scans/<scene_id>/<scene_id>_vh_clean_2.ply`, optional `<scene_id>.sens` |

Download helper: `scripts/download_scannet_data.py` (`--accept-tos`, `--scene`, `--scenes-from-sqa3d`, `--with-sens`, `--verify`).

```bash
# One scene smoke (~few MB mesh only)
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00

# Posed RGB-D for real ScanNet replay (~hundreds of MB per scene)
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00 --with-sens

# All scenes referenced by SQA3D val split (large — use --limit for dev)
uv run python scripts/download_scannet_data.py --accept-tos --scenes-from-sqa3d --split val --limit 10
```

Official script is vendored at `scripts/scannet/download-scannet.py` (from [ScanNet release](https://kaldir.vc.in.tum.de/scannet/download-scannet.py)).

## Prediction format

`emet eval-sqa3d` accepts:

- **JSONL** — one row per question: `{"question_id": 220602000000, "answer": "brown"}`
- **`eqa_results.json`** — Dynagraph export (`questions` list with `answer` or `discord_text`)

Answers are normalized with the SQA3D / LEO `clean_answer` protocol before exact match.

## Metrics

### QA (primary paper metric)

- **EM@1** — exact match against gold answer(s) after normalization
- **EM@1 refined** — substring match fallback (diagnostic)
- **Per-type** — `what`, `is`, `how`, `can`, `which`, `other` (first word of question)

### Localization (situation understanding track)

When you have predicted agent poses (up to 3 candidates per sample):

- **Acc@0.5m**, **Acc@1.0m** — XY position error vs GT
- **Acc@15°**, **Acc@30°** — yaw error vs GT

Use `emet.benchmarks.sqa3d.summarize_localization` from Python; localization JSON is optional (`--fetch-localization`).

## Methods (`--method`)

| Method | Stack | When to use |
|--------|-------|-------------|
| **`dynagraph`** (default) | **DynaMem** voxel map (nav + observations) + **GraphEQA** graph memory for EQA | Best of both — recommended for paper numbers |
| **`dynamem`** | DynaMem voxel map only (`query_answer` on semantic memory) | Ablations / voxel-only baseline |

Both use `prompt_variant: sqa3d` in `dynav_config.yaml`. Dynagraph tuned profile keeps merge/staleness from yaml; smoke disables merge for CI speed.

Each episode:

1. Agent at SQA3D annotation pose (posed `.sens` RGB-D when available, else Open3D mesh replay).
2. Rotate-in-place + map/graph updates.
3. Situated EQA (`situation` + `question`).

Score batch JSONL with `emet eval-sqa3d`.

## Embodied ScanNet replay

**Replay modes** (`--replay-mode` on `run-episode`, `run-batch`, `run-real-sweep`):

| Mode | Behavior |
|------|----------|
| **`auto`** (default) | Use posed ScanNet `.sens` RGB-D when the file is on disk and the agent stays within ~0.75 m XY of the SQA3D anchor pose; otherwise Open3D mesh rendering |
| **`sens`** | Require `.sens`; fail fast if missing. Best match to real ScanNet appearance at the annotated pose |
| **`mesh`** | Open3D offscreen rendering over `_vh_clean_2.ply` only (vertex-color unlit shading) |

When `.sens` is active, RGB and depth come from the nearest recorded frame to the agent camera pose; `Observations.camera_pose` uses the ScanNet camera extrinsics (OpenCV convention). Navigation away from the anchor falls back to mesh in `auto` mode.

Real-VLM runs use **640×480**; smoke mock uses **480×360**. The agent is placed at the SQA3D annotation pose (`position` + quaternion `rotation`), then explores and answers with `prompt_variant: sqa3d`.

**Profiles**

| Profile | When | Planning | Notes |
|---------|------|----------|-------|
| `smoke` | `--mock-llm` (default) | 8 steps, no nav | CI / fast |
| `tuned` | real VLM (default) | 15 steps, 3 nav steps | `dynav_config.yaml` defaults |

```bash
uv run python scripts/download_sqa3d_data.py --fetch-annotations
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00
uv run emet sqa3d run-episode --split train --question-id 220602000000 --mock-llm
uv run emet sqa3d run-batch --split train --question-start 0 --question-end 50 --mock-llm -o /tmp/sqa3d_smoke.jsonl
```

`run-batch` skips questions without required replay assets by default (`--skip-missing-scenes`; respects `--replay-mode`). Indices are positions in the split list, not `question_id` values.

**Batch vs sweep**

| Command | VLM | `--isolate-episodes` default | Typical use |
|---------|-----|------------------------------|-------------|
| `run-batch` | mock or real (`--profile`) | **off** (single process, faster if GPU is dedicated) | CI smoke, dev slices |
| `run-real-sweep` | real (`tuned`) | **on** (subprocess per episode) | Paper numbers on one GPU |

For real VLM on a shared or tight GPU, prefer `run-real-sweep` or `run-batch --isolate-episodes`. See [sqa3d_compute.md](sqa3d_compute.md).

```bash
# Real-VLM batch with GPU isolation (same as run-real-sweep without auto-eval)
uv run emet sqa3d run-batch --split val --question-start 0 --question-end 30 \
  --profile tuned --replay-mode sens --isolate-episodes \
  -o /tmp/sqa3d_batch.jsonl
```

## Paper figures (TP / FP / FN)

Episode JSONL from `run-batch` / `run-real-sweep` can be turned into outcome breakdowns and PNGs:

```bash
uv run emet sqa3d plot-results \
  -p /tmp/sqa3d_real_sweep/dynagraph_val_q0-30.jsonl \
  -o paper/figures/sqa3d_dynagraph_val30
```

Writes:

| Artifact | Description |
|----------|-------------|
| `sqa3d_outcomes_summary.json` | TP/FP/FN/infra counts, EM@1, per-type breakdown |
| `examples_{tp,fp,fn,infra}.jsonl` | Qualitative examples for the appendix |
| `outcomes_bar.png` | TP / FP / FN / infra bar chart |
| `em_by_question_type.png` | EM@1 by question prefix (`what`, `is`, …) |
| `outcomes_by_question_type.png` | Stacked TP/FP/FN by type |
| `confusion_matrix.png` | Gold vs predicted label heatmap (top-K answers) |

Outcome rules (generative QA — **TN is not defined**):

- **TP** — EM@1 correct
- **FP** — substantive wrong answer
- **FN** — abstain (`unknown`/empty) without infra failure
- **infra** — OOM / `ERROR:` (excluded from EM denominator by default)

Integration smoke (auto-runs when `scene0380_00` mesh is on disk):

```bash
uv run emet test src/test/benchmarks/sqa3d/ -v
```

## See also

- [sqa3d_compute.md](sqa3d_compute.md) — GPU memory, isolation, multi-GPU sharding (no pricing)
- [dynagraph_benchmarks.md](dynagraph_benchmarks.md) — Dynagraph sim harness (separate from SQA3D)
- [habitat/README.md](habitat/README.md) — HM-EQA Habitat harness
- Paper: `paper/sections/04_experiments.tex` (SQA3D row in evaluation plan)
