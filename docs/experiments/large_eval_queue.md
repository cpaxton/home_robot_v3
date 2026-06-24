# Large paper eval queue

Orchestrated overnight / multi-day sweep across SQA3D, OVMM find-phase, and dynamic exploration.

**Script:** [`scripts/run_large_paper_eval.sh`](../../scripts/run_large_paper_eval.sh)

**Full timing and env vars:** [paper_benchmarks.md § Large paper eval queue](../paper_benchmarks.md#large-paper-eval-queue)

## Quick start

```bash
# Full queue: SQA3D val+test → OVMM find → dynamic exploration
./scripts/run_large_paper_eval.sh

# One phase
./scripts/run_large_paper_eval.sh sqa3d-val
./scripts/run_large_paper_eval.sh ovmm
./scripts/run_large_paper_eval.sh dynamic-explore

# Skip phases
SKIP_OVMM=1 ./scripts/run_large_paper_eval.sh
SKIP_DYNAMIC_EXPLORE=1 ./scripts/run_large_paper_eval.sh
```

## Outputs

| Phase | Default location |
|-------|------------------|
| SQA3D | `~/runs/emet/sqa3d/` (`EMET_SQA3D_OUTPUT`) |
| OVMM find | `~/runs/emet/ovmm_find_phase/large_<ts>/` |
| Dynamic explore | `~/runs/emet/dynamic_exploration/large_<ts>/` |
| Logs | `~/runs/emet/large_eval/<phase>.log` |

## Speed knobs (summary)

```bash
# Best throughput: 4 GPUs, full val+test, both methods
SQA3D_GPUS=0,1,2,3 ./scripts/run_large_paper_eval.sh

# Paper iteration: val dynagraph only
SQA3D_METHODS=dynagraph SKIP_SQA3D_TEST=1 SKIP_OVMM=1 SKIP_DYNAMIC_EXPLORE=1 \
  SQA3D_GPUS=0,1,2,3 ./scripts/run_large_paper_eval.sh sqa3d-val

# Overlap: terminal A = SQA3D on GPU; terminal B = OVMM on CPU
OVMM_CPU_ONLY=1 ./scripts/run_large_paper_eval.sh ovmm
```

## Wall-clock estimates (resume on)

| Phase | 1 GPU isolate | 4 GPU isolate |
|-------|---------------|---------------|
| SQA3D val (both methods) | ~10–22 days | ~2.5–6 days |
| SQA3D test (both methods) | ~10–22 days | ~2.5–6 days |
| OVMM find replicates | ~0.5–1.5 days (CPU) | same |
| Dynamic explore | ~1–2 days | same |

Default full queue on 1 GPU: **~22–38 days**. With `SQA3D_GPUS=0,1,2,3`: **~6–11 days**.

## Per-track docs

- [sqa3d.md](sqa3d.md)
- [ovmm_find_phase.md](ovmm_find_phase.md)
- [dynamic_exploration.md](dynamic_exploration.md)
