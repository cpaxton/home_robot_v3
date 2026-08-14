# HM-EQA paper-113 dynagraph — partial-run analysis (95/113)

Run: `subset_paper113_20260814_121450_dynagraph_qwen3_vl.jsonl` (dynagraph, Qwen3-VL-8B, pre-static_graph; job died at 95/113 — **no bug**, cancelled/relaunched under disk pressure at 97%).
Results jsonl is intact and resumable (`run-batch --resume`).

## Headline

- **95/113 done, 43 correct → 45.3%** on the done slice.
- This is the **local-VLM tier** (Qwen3-VL-8B, no API VLM) — GraphEQA's API-VLM baselines are 63–67%; our local tier is expected to trail.

## Accuracy by question type (the useful signal)

| Type | correct/total | acc |
|------|--------------|-----|
| state / yes-no | 23/38 | **61%** |
| location / where | 12/32 | 38% |
| other | 4/7 | 57% |
| count (how many) | 3/13 | **23%** |
| time / clock | 1/5 | 20% |

## Take-aways

1. **State/yes-no questions are our best class (61%)** — the graph + verify gate handles "is X on/closed/cleared" reasonably.
2. **Count and clock questions collapse (23% / 20%)** — these need close-look fine detail (counting objects, reading a clock face). This is the same class of failure the agentic **close-look / VLM-assess** path targets (and where a **SigLIP / dense-patch evidence** nudge can help — the fix in `feat/tamp-floor-experiments`).
3. **Location questions (38%)** — borderline; the "which rug at the shower vs bedroom" disambiguation cases fail.

## Per-scene trouble spots

- `ACZZiU` 0/5 (0%), `HeSYRw` 0/3, `H1D2FZ` 0/2, `Y8Y6uk` 0/2, `D8bT1a` 0/2, `2dZ1Ji` 0/2 — repeat-fail scenes worth a look.
- `WT4QWw` 4/6 (67%), `HkseAn`/`41FNXL`/`w7QyjJ`/`ZVScmf`/`Uuwwmr` 100% — strong scenes.

## Not a sweep you can throw away

- It is a **resumable partial of the paper's dynagraph local-VLM row** (95/113) plus a classification of *where the remaining 18 questions and the static_graph half sit*. If the paper needs dynagraph-local numbers, resume from here rather than restart (23G of episode cache is on disk for it).
- The type breakdown is directly actionable: count/clock/close-detail questions are the gap the close-look + SigLIP-evidence work targets.
