# SQA3D benchmark harness

[SQA3D](https://github.com/SilongYong/SQA3D) (Situated Question Answering in 3D Scenes, ICLR 2023) evaluates embodied agents that must understand a **situation** (where they stand in a ScanNet scene) and answer an open-ended **question** about their surroundings.

This repo provides **dataset loaders**, **EM@1 scoring** (official QA metric), **localization metrics**, **ScanNet mesh replay** for embodied GraphEQA / Dynagraph, and **figure export** (TP/FP/FN breakdown).

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
| Embodied episode | `uv run emet sqa3d run-episode --split train --question-id 220602000000 --mock-llm` |
| Unit tests | `uv run emet test src/test/benchmarks/sqa3d/ -v` |
| One-command smoke | `uv run python scripts/run_sqa3d_scannet_smoke.py` |
| Score batch JSONL | `uv run emet eval-sqa3d -p /tmp/sqa3d_batch.jsonl` (auto-detects episode format) |
| Paper figures (TP/FP/FN) | `uv run emet sqa3d plot-results -p /tmp/sqa3d_batch.jsonl -o /tmp/sqa3d_figs` |
| Real VLM sweep | `uv run emet sqa3d run-real-sweep --question-start 0 --question-end 10` |

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
| `SCANNET_ROOT` | `~/.cache/scannet` | `scans/<scene_id>/<scene_id>_vh_clean_2.ply` |

```bash
# One scene smoke (~few MB)
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00

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

## GraphEQA / Dynagraph integration

1. Set EQA prompt variant for open answers:

   ```yaml
   eqa:
     prompt_variant: sqa3d
   ```

2. Format each sample as situation + question (`SQA3DQuestion.formatted_prompt()` or `format_sqa3d_prompt`).

3. Export `eqa_results.json` from an episode batch and score:

   ```bash
   uv run emet eval-sqa3d -p runs/sqa3d_val/eqa_results.json --split val -o sqa3d_val.json
   ```

## Embodied ScanNet replay

Open3D offscreen rendering over ScanNet `_vh_clean_2.ply` meshes. The agent is placed at the SQA3D annotation pose (`position` + quaternion `rotation`), then GraphEQA / Dynagraph explores and answers with `prompt_variant: sqa3d`.

```bash
uv run python scripts/download_sqa3d_data.py --fetch-annotations
uv run python scripts/download_scannet_data.py --accept-tos --scene scene0380_00
uv run emet sqa3d run-episode --split train --question-id 220602000000 --mock-llm
uv run emet sqa3d run-batch --split train --question-start 0 --question-end 50 --mock-llm -o /tmp/sqa3d_smoke.jsonl
```

`run-batch` skips questions without a local ScanNet mesh by default (`--skip-missing-scenes`). Indices are positions in the split list, not `question_id` values.

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

- [dynagraph_benchmarks.md](dynagraph_benchmarks.md) — MuJoCo / MolmoSpaces harness
- [habitat/README.md](habitat/README.md) — HM-EQA Habitat harness
- Paper: `paper/sections/04_experiments.tex` (SQA3D row in evaluation plan)
