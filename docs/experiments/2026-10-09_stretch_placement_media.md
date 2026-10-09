# Stretch table-transfer examples

These recordings isolate the native Stretch arm/gripper model from the ongoing
Sourccey model work. They are **scripted simulator examples, not full TAMP or
agent acceptance**. Coupled-joint pose IK generates actuator targets. MuJoCo
advances the live state; the script does not attach/weld the object or write
runtime object/robot poses. It does not certify arm/payload collision-free paths.

The two fixtures use the existing blue block and red cylinder in the default
table scene. Both start at the reachable near edge and target a point 16 cm
deeper on the same table. For the cylinder fixture, the neighboring block is
moved sideways **before simulation starts** to provide open-gripper clearance.
These are intentionally simple solvable examples, not clutter robustness tests.

Acceptance requires at least 8 cm of measured object lift, both fingertip
contacts and no table contact during the final second of lift and transfer,
less than 2.5 cm relative translation drift against the lifted grasp reference,
final XY error below 4 cm, final height error below 1.5 cm, and two seconds of
released table support with speed below 1 cm/s and no fingertip contact.
This does not check rotational retention or complete robot collision safety.
JSON results and frame-sampled contact/pose traces accompany the videos.

Initial trials are retained outside the repository: a center-height grasp
contacted the table and did not lift; a higher grasp lifted only 7.2 cm with the
smaller lift command. The final command gives more lift clearance. The cylinder
with the neighboring block close by also failed to grasp; increasing fixture
clearance fixes this simple case without changing the robot model or contacts.
These tuning attempts are not an unbiased success-rate experiment.

## Reproduction

From this worktree, with the project environment and robot assets installed:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
.venv/bin/python scripts/render_stretch_pick_place.py --render \
  --object object1 --output /tmp/stretch-examples/blue-block
.venv/bin/python scripts/render_stretch_pick_place.py --render \
  --object object2 --output /tmp/stretch-examples/red-cylinder
```

Omit `--render` for the same dynamics and checks without rasterization. Rendering
uses CPU Mesa; no shared GPU jobs are preempted. Each clip is about 27 seconds,
with the task, active stage, and assistance scope in an external border. Original
scene pixels are retained. Numbered clean stills correspond to the result events.

## Recorded results

| Example | Measured lift | Net table transfer | Final XY target error |
| --- | ---: | ---: | ---: |
| [Blue block video](media/stretch_placement/blue_block.mp4) | 13.1 cm | 14.3 cm | 1.7 cm |
| [Red cylinder video](media/stretch_placement/red_cylinder.mp4) | 13.8 cm | 17.3 cm | 1.3 cm |

Both recorded runs pass the checks above. Result JSONs are beside the videos.
The paper appendix includes `paper/figs/stretch_pick_place.pdf`; clean frames,
contact traces, and a local HTML gallery are in `/tmp/stretch-examples/`.
The recordings were executed directly in-process on CPU, not through the ZMQ
agent path. They therefore do not validate stored-plan execution or live-client
state synchronization. Source base: `b4430601`; media script was uncommitted
when recorded. The cylinder result includes the executed script hash; the blue
recording predates that provenance field (its object1 dynamics/checks are the same).
