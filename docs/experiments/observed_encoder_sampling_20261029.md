# Observed-input sampling and encoder geometry — October 29

Implemented simulator co-sampling, a distinct eight-channel motor packet and
motor-only rigid EE FK. The final live development smoke passed **5/5 motor/FK
checks, geometry fits and observed refresh calls**. **No full observed transfer
has passed yet.** Default execution remains unavailable without a qualified source.

![Reported capture ages and fitting times](figures/observed_encoder_sampling_20261029.png)

## What changed

Previously RGB/depth acquisition intervals were preserved, but the published
camera pose and normal joint state could describe a later robot posture. The
opt-in renderer now copies `MjData` under the physics lock, then renders all
cameras from that private sample. Physics continues while rendering. Sampling
bounds describe the original copy, not rendering or IPC delivery. Camera world
poses follow the pixels. The real client also preserves its simulation identity.

Full simulated robot configuration includes passive fingertip angles and
individual telescope lengths that hardware does not measure. That packet is now
explicitly `simulator_robot_kinematics_debug`. Observed effect admission rejects
its `RobotPoseSample` records with `robot_feedback_unqualified`, including when a
caller supplies zero alignment errors. The first prototype's generic source name
is retained in its raw artifact and is retrospectively classified as debug truth.
It must not be counted as qualified observed robot feedback.

A separate `motor_sample` contains the same eight motor-position channels as the
Stretch driver: lift, total extension, wrist yaw/pitch/roll, gripper slider and
head pan/tilt, in the declared actuator-length calibration. Transmissions and
gearing are checked. Passive angles and per-link telescope values are absent.
The consumer rejects stale, asynchronous, other-clock, malformed, missing and
extra-channel packets, and does not read the full-qpos debug packet.

`estimate_stretch_ee_from_encoders` uses those motors plus the localized camera
pose. Dividing total extension among collinear slides preserves rigid EE FK; it
does not determine each link's collision geometry. The estimator rejects changed
robot/camera/transmission identity and unmeasured joints on the EE/camera ancestry.
It explicitly leaves gripper geometry and execution unqualified. Tests alter
passive fingers and telescope split while preserving motor readings: EE FK stays
correct while those unobserved geometry coordinates remain ambiguous.

## All live attempts

| Managed job | Refresh / geometry | Robot sampling debug | Motor-only FK | Combined diagnostic |
| --- | --- | --- | --- | --- |
| `20261029_004156_b485e5` | 5/5 each | 5/5 | Not collected | Passed; prototype robot packet is debug-only |
| `20261029_004936_9c83b3` | 5/5 each | 4/5 | Not collected | Failed: one capture expired during fitting |
| `20261029_005357_2ae4f7` | 5/5 each | 5/5 | Not collected | Passed after redundant seed removal |
| `20261029_010537_303f19` | 5/5 each | 5/5 | 5/5 | Passed with separate motor/evaluator channels |

The stale frame remains rejected under the original two-second limit. The solver
now removes only starts equivalent under a declared exact shape symmetry: the
24 cube rotations yield three unoriented cylinder-axis starts, with a separate
measured warm start on later frames. Measured models retain undeclared rotational
alternatives. Distinct centers, tilt alternatives, original-point surface checks,
nullspaces and work limits remain. The earlier fast repeat fit in **0.40–0.47 s**
and checked capture ages of **0.93–1.20 s**. These are development-run observations,
not a matched timing ablation or hardware uncertainty measurement.

The final motor-only EE estimates matched the separate robot debug reference to
numerical precision (maximum position difference below `4e-15 m`). This validates
the channel/projection/kinematic invariant in the simulator, not independently
measured physical pose accuracy. Both use the declared robot calibration model.
The image diagnostic uses a declared cylinder prior and the same development
fixture/seed. Caliban made three semantic calls per run; updates added no VLM
inference. Object-only refreshes return an empty task list; support is measured
separately and these results do not establish executable task discovery.

## Contact controls and hidden geometry

Privileged native controls passed **2/2** in each of two calibration audits. These
are repeated controls of the two development fixtures, not four independent tasks
or observed task successes. The second audit separates initial fixture settling
from execution and retains peak command/time context.

During execution with target contact, the largest finger-opening deviation from
the nominal `10 × slider` relation was **0.00827 rad**; rubber joint angle reached
**0.11877 rad**. The large **0.35816 rad** peak without target contact occurred at
**0.002 s during fixture initialization**, before task commands. During execution
without target contact, opening deviation peaked at **0.00622 rad**. Ground-truth
contacts classify these debug samples only; no contact channel is supplied to the
observed runtime. Physics-step samples are correlated, not independent trials.

These measurements motivate a gripper-geometry uncertainty model and settling
admission. Two controls do not certify a global error bound. Motor-only EE FK can
be correct while rubber/finger geometry varies; runtime collision/clearance checks
must bound that variation rather than read passive joints as fictitious sensors.

## Reproduction and validation

Requires the project simulation environment, working OSMesa and remote Caliban
access. If the software renderer is installed in a non-system path, configure
`LD_LIBRARY_PATH` as in the author's run scripts. The fixture is now provisioned
from packaged assets rather than depending on an undocumented checkout-local XML:

```bash
export EMET_UV_RUN=1 PYTHONPATH="$PWD/src:$PWD/scripts"
uv run --no-sync python scripts/prepare_observed_stretch_fixture.py \
  --output .tamp-runs/observed-review/fixture --port-offset 3180

MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES= \
EMET_STRETCH_HEAD_CAMERA_SCALE=2 EMET_SIM_HEAD_RENDER_ROBOT=1 \
EMET_SIM_SYNCHRONIZED_RGBD=1 \
NO_PROXY=192.168.1.55,localhost,127.0.0.1 \
no_proxy=192.168.1.55,localhost,127.0.0.1 \
uv run --no-sync emet jobs run --name observed-encoder-review \
  --cpu-safe --no-gpu-exclusive --out-dir "$PWD/.tamp-runs/observed-review/job" -- \
  python scripts/eval_observed_capture_binding.py \
  --sim .tamp-runs/observed-review/fixture/sim.json \
  --output .tamp-runs/observed-review/live \
  --base-url http://192.168.1.55:8000/v1 --model Qwen/Qwen3-VL-8B-Instruct \
  --samples 5 --execution-readiness --support-query tabletop \
  --geometry-model src/test/fixtures/geometry/cylinder_prior.json \
  --synchronized-robot-model .tamp-runs/observed-review/fixture/stretch.xml
```

Expect separate component outcomes and `execution_qualified=false`; requested
missing/stale components fail the diagnostic. The JSON fixture manifest identifies
source assets and explicitly labels this a development fixture, not held-out data.
For contact calibration use `scripts/eval_stretch_encoder_geometry.py --output DIR`
through `emet jobs`, with `MUJOCO_GL=disable` and no local model/GPU workload.

Managed CPU regression `20261029_013616_c49114` passed **787 checks**, zero failures,
errors or skips, including all previous motion/TAMP/agent targets and the new
motor/sampling/fixture cases. The focused motor/source/fixture suite passed
**27 tests**, overlapping that regression. Ruff and whitespace checks pass; the
documented PR review command also passed **93 tests** (overlapping). The paper
builds at 52 pages. [JSON evidence](observed_encoder_sampling_20261029.json)
preserves all attempts, script hashes, control audits, source-asset manifest and
validation counts. Raw data: `.tamp-runs/observed-alignment-20261029/`.

## First-transfer gate

The remaining work is concrete source/contract integration:

1. Consume primitive shape symmetry as geometry equivalence in payload planning,
   retention and effects. Keep unknown yaw explicit; do not forge a unique pose.
2. Bound gripper/telescope geometry from real motor inputs and declared uncertainty.
   Validate bounds against held-out debug geometry, including startup, transitions
   and loaded contact. The two audit episodes alone cannot qualify those bounds.
3. Associate moving object masks from current images and measured tracking,
   rejecting ambiguity/occlusion. The live harness still uses stationary reference
   reassociation; fresh point-set registration tests do not qualify moving masks.
4. Assemble that evidence with support identity, persistent geometry provenance and
   phase-specific contact/unknown-space policy in the existing observed adapter.
   Keep original map observation times; a new image does not resurvey occluded
   support or background cells. State any static-scene/model assumptions explicitly.
5. Run the first stationary transfer through the public tools, then freeze held-out
   transfers, a buffered swap and failure/recovery cases. Navigation/exploration
   follows those gates. Hardware clock/noise qualification remains separate.
