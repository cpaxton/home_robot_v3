# Curriculum videos and target coverage

The Mars curriculum has seven baseline **4×4 videos**, one for each motion family:
ordinary hand/base tracking, base-only tracking, goal changes, three random
arm-spline levels, and unloaded TAMP approach/grasp-pose execution. Level zero is the initial spline training level; medium/full are later stages
shown for inspection, not a claim that the current run samples them. The videos
show actual analytic-teacher simulation. An eighth video, **Larger arm motions |
bounded v2 candidate**, compares a broader arm distribution with shorter holds.
All videos show teacher execution, not learned-policy performance. Playback is
real time; the candidate video explicitly replays shorter successful recordings
with a visible reset instead of leaving their tiles frozen. Replays are not new
trials or continuous trajectories. Failed trials still freeze at their terminal
recorded state.

Recordings are sampled at 20 Hz. Baseline videos were encoded at 8 fps; the
larger-motion candidate now uses 20 fps, without speeding up physical time.
Tiles report finite-interval measured joint speed (deg/s) and hand speed (cm/s).
Tiny floating-point timestamp errors are tolerated when choosing a recorded
frame, without interpolating robot states. The initial sample has no prior
interval and therefore no speed readout.

Each tile has a distinct case/seed identity, split, actual robot, target/measured
hand axes, base goal direction and hand tracking error. Failed teacher cases
and admission failures remain visible. Recordings that end earlier hold their
last recorded state with an explicit END label; stopping is confirmed separately
for general cases. No stopped physics is invented to fill a tile.

## Viewing

The existing SSH-forwarded offline viewer serves:

- `http://localhost:8771/curriculum/`: videos, downloads and coverage preview.
- `http://localhost:8771/curriculum/assets/coverage.html`: interactive 3D paths,
  split/family filters, orientation axes and base paths; drag to orbit/scroll to zoom.

Use the existing tunnel from the client:

```powershell
ssh -N -L 8771:127.0.0.1:8771 cpaxton@olympia
```

The gallery appears when the replay server is started with `--curriculum`:

```bash
PYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -m emet.visualization.herman_rehearsal \
  --rehearsal logs/eval/herman-elbow-wave-v1 --port 8771 \
  --curriculum logs/eval/curriculum-videos-v1
```

Its catalog serves only checksummed media/images/coverage HTML. Raw model,
physics and sensor artifacts are not HTTP assets. Byte ranges support native
video seeking; both H.264 MP4 and VP9 WebM are saved for browser compatibility.
No hardware connections or command endpoints are added.

## Samples and coverage limits

Sixteen cells sample train/audit cases from the existing case pools. The base
and TAMP pools have only eight and six original identities, so the remaining
eight/ten examples are explicitly **visualization-only**, in a separate 480M
seed namespace. They never enter the optimizer. Spline videos each show eight
train/eight audit cases, covering all four reset postures at each level.

Coverage export samples **all 144 baseline spline cases**, including those
outside the videos. Rejected identities remain in `coverage.json`. General and
TAMP paths come from recordings: general paths include the full announced
reference (both stages for goal changes); TAMP coverage includes executed
unloaded-prefix goals only. Neither failed nor unexecuted plans are silently
converted into successful robot motion. With the sixteen v2 candidate paths,
the current gallery contains **224 paths, 208 with active hand targets**. Its
global span still mixes reset postures; use local excursion below to assess
whether an individual episode exercises the arm.

The 3D view and standalone PNG/SVG show hand position paths, sparse orientation
axes, and base paths. Base-only hand masks are false; their passive hand sweeps do not count toward
active hand coverage. Numeric summaries report active-hand axis spans, occupied 2 cm voxels,
posture/split counts and rotation range. These describe sample coverage; there
is no known reachable-workspace denominator and no percentage coverage claim.
The videos also make sparse or nearly stationary parts of the current mixture
visible. Loaded objects, grip closure and carry/place remain absent here.

## Reproduce and inspect

The complete entrypoint is `scripts/render_wbc_curriculum.py --output logs/eval/curriculum-videos-v1`.
It records/renders/publishes each family before proceeding, then exports all
spline targets. `--resume` reuses only completed, source-bound records/media
with unchanged hashes; incomplete families require a fresh output.

Run one numerical worker, numerical threads one, CUDA hidden, EGL rendering.
On the unstable workstation, schedule recording/rendering before PPO, or stop
at an explicit learner/checkpoint boundary and restart afterward. **Do not
freeze a live TAMP IPC worker:** its wall-clock polling deadline ages while
frozen. A stop hook that thaws a service does not prevent that timeout. This
first visualization run exposed that issue; the original interrupted run is
preserved and the unchanged frozen training configuration was restarted in a
fresh directory. See the linked experiment report below. Rendering reads recorded states, calls `mj_forward`, and never steps
physics. Video intermediates are deleted after each grid; models are compressed
and content-deduplicated within each family.

```bash
export PYTHONPATH=src EMET_UV_RUN=1 CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Repeat these two commands for each --family listed by --help.
taskset -c 15 .venv/bin/python -m emet.visualization.curriculum_record \
  --output logs/eval/curriculum-videos-v1 --family arm_splines_l0
taskset -c 15 .venv/bin/python -m emet.visualization.curriculum_video \
  --output logs/eval/curriculum-videos-v1 --family arm_splines_l0 --fps 20

# Export all declared spline targets, then publish completed media.
taskset -c 15 .venv/bin/python -m emet.visualization.curriculum_coverage \
  --output logs/eval/curriculum-videos-v1
.venv/bin/python -c 'from emet.visualization.curriculum_gallery import publish; publish("logs/eval/curriculum-videos-v1")'
```

Use fresh family output directories. Recordings include copied qpos, targets,
measured hand/base/joints, model/source hashes, case identities and outcomes.
Receipts include frame count, 1× time scale, resolution, media hashes, source
report hashes and measured motion spans. Check decoded frames and the browser
player, not only an encoder return code. Keep all videos, binary models,
recordings and compressed traces under ignored `logs/`; commit scripts, docs
and compact numeric evidence only.

Generate the additional candidate without replacing baseline recordings:

```bash
taskset -c 15 .venv/bin/python scripts/render_wbc_curriculum.py \
  --output logs/eval/curriculum-videos-v1 --family dense_arm --loop-shorter
```

Its sixteen teacher trials all succeed. Median local hand excursion increases
from 33.5 mm in baseline small splines to **80.5 mm**, and physical time with a
moving reference increases from 66.5% to **86.2%**. This is a different
distribution, not a timing-only comparison or a new actor result. Conservative
derivative bounds can still make larger paths slow; the speed, acceleration
and jerk caps have not been raised.

Measured candidate median peak speeds are **0.229 rad/s (13.1 deg/s)** for the
fastest joint and **3.92 cm/s** for the hand. Level zero deliberately uses 40%
of the configured 0.6 rad/s arm profile, or 0.24 rad/s. Its reference reaches
96% of that selected cap; the slowness is substantially real, not only 8 fps
presentation or an overly conservative jerk estimate. This profile is not a
new measurement of Herman's hardware capabilities.

Audit physical speed coverage separately from video timing:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_wbc_motion_speeds.py \
  --recordings logs/eval/curriculum-videos-v1 \
  --output logs/eval/motion-speed-audit-v1
```

The report includes per-joint pose ranges and speed extrema, plus time-weighted
speed-bin exposure relative to configured profile caps. Quiet time stays visible;
base-only passive arm motion is labelled. Neither sparse sample counts nor
replayed video loops count as extra physical exposure.

[Generated videos, measured coverage and validation results](../experiments/mars_curriculum_visualization_20261117.md).

## Per-trajectory spread and dead-time audit

The global coverage box mixes reset postures. Inspect **within-episode** motion
as well: maximum excursion and path length, orientation excursion/travel, and
hand spread relative to the commanded SE(2) base. A moving base must not count
as arm movement; a return-to-start path must not count as zero movement.

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_wbc_trajectory_spread.py \
  --recordings logs/eval/curriculum-videos-v1 \
  --output logs/eval/trajectory-spread-v1
```

The existing gallery now shows these statistics. The audit preserves failed
trials and distinguishes physical reference motion from extra frozen video
padding. Raw reports remain ignored under `logs/`.

| Family | Physical time with moving target | Median base-relative hand excursion |
| --- | ---: | ---: |
| Ordinary hand | 16.5% | 0.4 mm |
| Base-only | 15.6% | Inactive hand |
| Goal changes | 31.0% | 7.6 mm |
| Small splines | 66.5% | 33.5 mm |
| Medium splines | 76.9% | 77.9 mm |
| Full splines | 77.6% | 135.0 mm |
| Unloaded TAMP prefix | 53.1% | 24.5 mm |

Ordinary cases move for approximately 3.9 of 25 seconds. A normalized phase
retimer also imposed a 3.9-second floor even for tiny arm paths. These are setup
issues, not merely slow video playback; they plausibly underweight arm tracking
but do not prove the sole cause of policy failure.

Opt-in `bounded_v2` timing derives duration directly from the same conservative
velocity/acceleration/jerk bounds. `finish_after_hold` ends a general episode
after its unchanged two-second measured quiet hold; goal-change episodes cannot
finish before their successor. The v2 general candidate declares a six-second
post-reference ceiling, including that quiet hold. It ends immediately after
successful settling rather than padding every episode to the ceiling. Spline
cases keep their three-second ceiling. Legacy reference timing remains available
for comparison.

The candidate `mars_arm_splines_v2.json` samples larger arm excursions, uses
independent seeds, a 0.2-second initial hold, and rejects paths below both 5 cm
base-relative hand excursion and 0.35 rad orientation excursion. This allows
intentional orientation-only motion without accepting no-ops. It is a candidate,
not a newly qualified policy. V2 also replaces the ordinary hand and goal-change
pools with seeded, larger joint offsets whose hand targets are generated by FK;
base-only and unloaded TAMP cases stay explicit. Both nominal handoff stages are
geometry-admitted before execution, and the hand-active references must meet the
same local spread requirement. Fine TAMP insertions and explicit hold tasks
remain separate. Per-family moving/quiet/unknown physical ticks now appear in
the training ledger and W&B. The preflight requires exact canonical identities
and source binding for every family before a v2 training launch.

The first expanded preflight retained all 70 cases and confirmed all stops, but
only 4/16 ordinary hand, 1/8 base and 5/16 goal-change cases succeeded (splines
24/24, unloaded TAMP 6/6). A three-second ceiling starved some two-second holds.
Diagnostics also exposed a stale 5 mm fixture tolerance in the base servo after
the task switched to a 5 cm region. Correcting that mismatch alone can leave
3–4 mm hand residuals: a stopped base removes a direction the five-joint arm
needs. The v2 general teacher therefore opts into a joint base/arm Jacobian
solve even with all base coordinates specified. Active base coordinates remain
lower-weight objectives; the precise hand contract and native motion guards
remain unchanged. This is an explicit analytic baseline option, not an actor
fallback. Default TAMP/runtime teacher configuration retains its existing solve.

The corrected full preflight passes **70/70**, with 70 confirmed stops: 24 arm
splines, 16 ordinary hand cases, 8 base cases, 16 handoffs and 6 unloaded TAMP
commands. Its canonical source/identity readiness check passes. The report is
kept under `logs/eval/motion-density-v2-coupled-preflight/`; the earlier failed
preflight remains under `logs/eval/motion-density-v2-full-preflight/`. This
validates the teacher distribution, not the learned policy or hardware safety.

Reproduce it with one numerical worker:

```bash
taskset -c 15 .venv/bin/python -m emet.rl.manipulation.arm_spline_benchmark \
  --manifest configs/benchmarks/mars_arm_splines_v2.json --level 0 --teacher \
  --output logs/eval/motion-density-v2-coupled-preflight
```
