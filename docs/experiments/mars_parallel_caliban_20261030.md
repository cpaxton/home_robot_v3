# Caliban parallel Mars setup, 2026-10-30

Compute-only experiments; no robot connection or motion. Source snapshot starts
at `98f166a8` plus the review repairs and native multi-worker implementation in
this change. No new trained policy is qualified or exported by these probes.

## Native workers: usable now

The mastery trainer supports spawn-based independent MuJoCo environments with
configurable worker count and rollout length. The callback counts actions,
rewards, case starts and completions from every worker. The original single-worker
default is preserved. Local tests exercise real PPO updates and export/load with
one and two in-process worlds; Caliban tests actual spawn processes.

Caliban uses MuJoCo 3.5.0, NumPy 1.26.4, SB3 2.9.0 and its existing Torch
2.13.0+cu130 with CUDA hidden for native tests. Each worker has one Torch/BLAS
thread. Models have the same fingerprint as the local reference. The isolated
environment does not change the running VLM installation or services.

| Native worlds | Rollout control steps/s, aggregate | Measured rollout | PPO smoke |
| ---: | ---: | ---: | --- |
| 1 | 31.7 | 128 steps / 4.04 s | 64 steps, 2 updates / 3.46 s |
| 4 | 104.9 | 512 steps / 4.88 s | 256 steps, 2 updates / 5.55 s |
| 8 | 193.0 | 1,024 steps / 5.30 s | 512 steps, 2 updates / 8.83 s |

This is a **3.31x / 6.09x** aggregate native-rollout speedup for four/eight workers.
Initialization is excluded (3.54/6.45/6.89 s). Zero-action rollouts intentionally
do not track moving goals; one episode with four workers and three with eight
ended with `mastery_tracking_envelope`. This is not a policy
success evaluation. Tiny PPO updates finish with finite parameters; no actor is
saved and no teacher/regression gate is bypassed for a production launch.

## GPU physics: promising prototype, not PPO-ready

Isolated MuJoCo and MuJoCo Warp 3.17.0 with Warp 1.18.0 run on the Orin GPU.
Mars's real exported geometry, implicitfast integrator, 2 ms physics timestep
and **five no-slip iterations** remain intact. The old 3.5 GPU backend is not
suitable because it lacks the no-slip solver. No driver upgrade is needed for
the tested current stack. First-time kernel compilation took several minutes.

| GPU worlds | Physics steps/s, aggregate | Control steps/s, aggregate |
| ---: | ---: | ---: |
| 1 | 892 | 35.7 |
| 8 | 6,445 | 257.8 |
| 32 | 19,719 | 788.8 |
| 64 | 35,415 | 1,416.6 |

Each control tick groups 25 physics steps (50 Hz). Measurement uses a captured
graph after compilation and warmup, 50 graph launches per batch size, replicated
initial states and fixed hold controls. It excludes controller, observations,
rewards, resets and PPO. GPU and native throughput columns measure different
workloads and cannot be used to claim an end-to-end training speedup. No repeated
trials or service-latency/temperature qualification was performed.

For state parity, native MuJoCo 3.5 generates identical two-second hold, forward,
left and right command sequences. Both Caliban CPU 3.17 and GPU 3.17 are compared
against that reference. Commands ramp over half a second and use only 0.03 m/s
or 0.15 rad/s. All four cases meet the predeclared short-probe limits of 1 mm
base position, 0.01 rad base orientation and 0.02 joint-coordinate disagreement.
Maximum GPU disagreement is **0.0487 mm**, **0.000446 rad** base orientation and
**0.00668** over non-base joint coordinates. The latter includes wheel travel.
Contact allocation has headroom and states remain finite in this probe.

These are short, low-speed open-loop checks. They do not test loaded retention,
substep force peaks, wheel unloading, disturbance recovery, independent resets or
long-horizon divergence. GPU rollouts must not enter policy qualification until
those checks and the batched environment/controller implementation are complete.

## Artifacts and next steps

Compact [GPU](../../paper/data/whole_body_control/parallel_caliban_20261030/gpu-summary.json)
and [native](../../paper/data/whole_body_control/parallel_caliban_20261030/native-summary.json)
summaries are in Git. Full model/reference/results are in ignored persistent
`logs/recovery/parallel-caliban-20261030/`. Caliban's RAM-backed work area is
`/dev/shm/cpaxton-mars-parallel-20261030`; its environments disappear on reboot.
Root storage was nearly full, so no training data was written there.

Follow the [parallel simulation plan](../plans/mars_parallel_simulation.md): use
the native four/eight-worker path after fresh host-specific gates, measure a bounded
jerk-focused run against the same 50 cases, and develop GPU parity/batched
observations/rewards separately. Latest existing policy remains 50/50 base
tracking/holding and 34/50 full mastery; 16 jerk misses remain. These setup
experiments change none of the manipulation or hardware acceptance claims.
