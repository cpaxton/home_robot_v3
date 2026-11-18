# Curriculum videos and target coverage

The Mars curriculum now has seven **4×4 videos**, one for each motion family:
ordinary hand/base tracking, base-only tracking, goal changes, three random
arm-spline levels, and unloaded TAMP approach/grasp-pose execution. The videos
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
axes, and base paths. Numeric summaries report axis spans, occupied 2 cm voxels,
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
On the unstable workstation, pause the owned simulator-training service while
recording/rendering and always resume it afterward. The managed video job has
an independent stop hook that resumes that service even if video generation
fails. Rendering reads recorded states, calls `mj_forward`, and never steps
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
