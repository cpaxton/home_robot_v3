# Curriculum videos and target coverage

The Mars curriculum now has seven **4×4 videos**, one for each motion family:
ordinary hand/base tracking, base-only tracking, goal changes, three random
arm-spline levels, and unloaded TAMP approach/grasp-pose execution. Level zero is the initial spline training level; medium/full are later stages
shown for inspection, not a claim that the current run samples them. The videos
show actual analytic-teacher simulation, not joint-witness animation, target
markers alone, or a learned-policy performance claim. Playback is real time.

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

Coverage export samples **all 144 declared spline cases**, including those
outside the videos. Rejected identities remain in `coverage.json`. General and
TAMP paths come from recordings: general paths include the full announced
reference (both stages for goal changes); TAMP coverage includes executed
unloaded-prefix goals only. Neither failed nor unexecuted plans are silently
converted into successful robot motion.

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
  --output logs/eval/curriculum-videos-v1 --family arm_splines_l0 --fps 8

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
after its unchanged two-second measured quiet hold, with a three-second
post-reference settling deadline; goal-change episodes cannot finish before
their successor. V1 behavior remains available for comparison.

The candidate `mars_arm_splines_v2.json` samples larger arm excursions, uses
independent seeds, a 0.2-second initial hold, and rejects paths below both 5 cm
base-relative hand excursion and 0.35 rad orientation excursion. This allows
intentional orientation-only motion without accepting no-ops. It is a candidate,
not a newly qualified policy. Ordinary hand cases still need broader arm motion;
fine TAMP insertions and explicit hold tasks should remain separate. Per-family
moving/quiet/unknown physical ticks now appear in the training ledger and W&B.
