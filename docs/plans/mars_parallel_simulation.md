# Parallel Mars simulator training

Use independent native MuJoCo workers immediately. Develop a batched GPU backend
behind matched physics tests before using its rollouts for policy qualification.
Mars remains first; no robot connection is part of this setup.

## Native trainer

`emet.rl.manipulation.mastery train` accepts `--num-envs N`,
`--vec-backend subproc` and `--rollout-steps 256`. Workers use spawn, independent
models/data and one Torch/BLAS thread each. Seeds are assigned independently by
SB3. Rollout telemetry counts every worker. Total steps must be divisible by
`N * rollout_steps`; changing N changes the aggregate PPO batch and should be
recorded as an optimization change, not hidden as a speed-only change.

Teacher preflight and fresh regression receipts remain mandatory. Generate them
in the training host's environment: library versions are part of source identity.
The default remains one environment. On the unstable workstation keep one physics
worker on CPU14, CUDA disabled, and at most one test worker on CPU15. Caliban's
bounded smoke tests use four and eight CPU workers; increase only after measuring memory,
temperature, throughput and the effects on its existing inference service.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH="$PWD/src" python scripts/benchmark_mars_parallel.py \
  --num-envs 1 4 --steps-per-env 128 --ppo-smoke \
  --output logs/parallel/native-probe.json
```

This probe exports no policy and does not bypass a promotion gate. For real PPO,
use the usual `mastery train` command and its gates, adding the worker flags.

## Caliban isolation and recovery

Caliban is an aarch64 Orin with 64 GB shared RAM. Its root disk had only 1.4 GB
free on 2026-10-30. Use `/dev/shm` for isolated environments, source copies and
kernel caches, and copy results back to persistent ignored `logs/` promptly.
These environments disappear on reboot. Do not install into the VLM environment,
upgrade the driver/system Python or stop existing services.

`scripts/setup_mars_parallel_env.sh native NEW_ENV TORCH_SITE_PACKAGES` builds a
fresh Python 3.12 environment. It exposes the existing Torch installation through
a read-only `.pth` and installs simulator dependencies only in NEW_ENV. On this
host Torch is at
`/home/cpaxton/src/home_robot_v4/.venv-vlm/lib/python3.12/site-packages`.
Use `warp` instead of `native` for the separate GPU probe environment. Setup
requires a fresh target; it refuses to overwrite another environment.

For the full controller/TAMP regression gate, also run
`scripts/setup_mars_parallel_test_env.sh ISOLATED_NATIVE_ENV` and provide the
repository's tests, configs, mirrored runtime, robot assets and category TSV.
The minimal rollout environment alone cannot collect these integration suites.
ARM Open3D 0.20 is available here where 0.19 is not; Arrow 19 preserves NumPy
1.26 compatibility. Fresh host-specific tests are mandatory. Text-only LLM
imports now defer microphone/Whisper dependencies until voice is requested.

The tested work area is `/dev/shm/cpaxton-mars-parallel-20261030`, with native
environment `/dev/shm/cpaxton-mars-rl-env` and GPU environment
`/dev/shm/cpaxton-mars-warp-current`. Local persistent reports live under
`logs/recovery/parallel-caliban-20261030/`.

## GPU backend admission

Use direct [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) first.
The tested pair is MuJoCo/MuJoCo Warp 3.17.0 and Warp 1.18.0. The old 3.5 GPU
backend lacks Mars's no-slip solver. Current upstream supports it for the
implicitfast integrator; do not turn it off to obtain an attractive benchmark.
[mjlab](https://mujocolab.github.io/mjlab/v1.6.0/source/installation.html) remains
an option; its current Torch requirement differs from Caliban's existing Torch
2.13, so this prototype keeps the installation isolated.

1. Export the actual initialized native model and meshes with
   `scripts/export_mars_parallel_model.py --output logs/parallel/model`.
2. Generate a reference using the validated MuJoCo 3.5 installation:
   `scripts/benchmark_mars_warp.py --model logs/parallel/model --cpu-reference
   --output logs/parallel/cpu-reference.json`.
3. Copy the model/reference to the GPU workspace. Run the same script with
   `--reference cpu-reference.json --batches 1 8 32 64 --output gpu-probe.json`.
   Physics cadence, model/state digests and no-slip iterations must match.
4. Extend parity beyond the short hold/forward/left/right probe: loaded arm,
   support contact and sliding, grasp/lift/placement, impulses and sustained
   wrenches, reset/goal changes, per-substep peak contact loads, wheel unloading,
   actuator force/velocity limits and overflow handling. Test independent states
   and seeds across worlds, not just replicas.
5. Implement batched controller state, history/preview observations, rewards,
   termination and physical reset admission. Account for every physics substep's
   contact peaks. Use zero-copy Torch/Warp tensors and measure end-to-end PPO,
   including resets and inference. Existing native tests remain the reference.
6. Only then run matched checkpoint/teacher qualification on held-out cases.
   GPU physics stepping throughput alone is not training throughput or evidence
   that manipulation is qualified. Historical CPU policy gates remain separate.
