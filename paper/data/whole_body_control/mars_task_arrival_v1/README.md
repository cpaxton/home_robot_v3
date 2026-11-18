# Mars task-arrival and maximum-margin reset diagnostics

This is compact evidence for twelve explicitly declared development attempts,
with zero full-task successes, no PPO updates and no hardware motion.

- `results.json`: four original-reset outcomes, two distinct reset variants,
  two grasp-region attempts, two hold-fix replays and two physical preparations,
  source/actor/archive identities, stage outcomes, stop/contact/retention evidence,
  exact teacher comparison and private attached-payload witness audit.
- `task-stages.svg`: active hand tracking and loaded teacher lift/rotation,
  ending loaded curves at their first fault.

Raw `.json.gz` archives remain ignored in the two input directories below.
To reproduce into a fresh path on an unoccupied CPU lane:

```bash
env -u WANDB_API_KEY CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/mpl-mars-task-analysis taskset -c 14 \
  .venv/bin/python scripts/summarize_mars_task_arrival.py \
  --arrival logs/eval/mars-task-arrival-v1 \
  --maximum-margin logs/eval/mars-task-maximum-margin-v1 \
  --grasp-region logs/eval/mars-grasp-region-v1 \
  --hold-reacquire logs/eval/mars-hold-reacquire-v1 \
  --physical-stance logs/eval/mars-physical-stance-v1 \
  --output /tmp/mars-task-arrival-analysis-fresh
```

The analysis checks trace hashes and the reconstructed initial state. It uses
private kinematics for the nominal witness audit and never steps physics.
Reconstruct each exact producer source manifest by combining
`source_hashes_common` with that study's `source_hashes_delta`; its canonical
sorted-JSON digest is stored as `source_hashes_sha256`. Shared entries are
stored once to avoid duplicating the same large manifest for every study.
Do not interpret measured geometry or sampled margins as robust certificates.
See the [report](../../../../docs/experiments/mars_task_arrival_20261115.md)
and [protocol](../../../../docs/plans/mars_task_arrival_diagnostic.md).
