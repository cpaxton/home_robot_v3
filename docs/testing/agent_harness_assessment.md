# General-purpose agent harness assessment

The first milestone is a capability audit across Stretch/MuJoCo, Molmo rby1,
and Habitat. Start from the main revision recorded in the
[case manifest](../../configs/eval/harness_assessment_20261119.json). Commit
the assessment driver and freeze its SHA before launching a scored cohort.
Use a clean checkout, a new artifact directory per attempt, and the existing
managed jobs policy. Keep the checkout, raw outputs, and job logs on persistent
storage so a reboot preserves completed cases. Validate one frame on the actual
renderer before resuming an interrupted live cohort. Keep simulator, replay,
and hardware results distinct.

## Assessment order

| Gate | Entry point and acceptance evidence |
| --- | --- |
| Contracts | Run the registry catalog, CHAT dispatch/camera/history, measured motion/floor, authored geometry, and [TAMP workflow](../TESTING.md#tamp-contracts). Record the JUnit output and skips. |
| Recorded views | Build all 12 development cases with [the existing builder](../../scripts/build_harness_capability_dev.py). Render only when a valid renderer is available. Run the [six replay cases](../../scripts/replay_harness_capability_evidence.py) with both history arms. Independently check the image roles and exact answers/abstentions. |
| First CLI transfer | Follow [the native control](../plans/2026-11-18_tamp_agent_closeout.md#first-agent-test). The existing 2-goal fixture helper in `run_shared_harness_integration.py` supplies the calibrated stationary reset and placement regions. Use `eval_native_chat_session.py --single-transfer --native-policy bounded_release` with the approved remote endpoint. The script scores only the cylinder and retains the two-goal default for existing callers. Require a successful receipt, stable independent geometry, and zero forbidden commands. |
| Stop and explicit retry | Use [the contextual recovery protocol](../../configs/eval/tamp_contextual_recovery_20261116.json) with `eval_tamp_partial_agent.py`. Run stop-only on `two-goal-control`. Run `two-goal-shift` with `--recover --recovery-style contextual`. Keep the original stop and recovery checks and record both outcomes. |
| Active observation | Run [the authored Stretch CHAT binding](../../scripts/run_harness_capability_agent.py) one case per job so a failed process cannot suppress later assignments. Start with `cap-dev-01` and `cap-dev-02` after a bridge/camera smoke. Score exact request images, acquisition, measured turns, and final answers. This profile uses geometry-only memory. Qualify learned grounding separately through the shared find runner. |
| Cross-simulator | Use `run_shared_room_pairs.py` with the named Stretch and Molmo find cases, model-final, seed 0, and the existing high-budget 12 rounds/8 nav calls. Use the matched scoped HM-EQA/OVMM entry points for Habitat q2/q6/q12 and the manifest's nearest-v2 proxy. Retain how each runner supplies depth, semantics, navigation, and other assistance. |
| Observed composition | Qualify [#245](https://github.com/cpaxton/home_robot_v3/issues/245) stationary transfer and evidence-loss controls, then repeat the agent transfer with observed input. Follow with the stopped base handoff, protected multigoal, swap and authorized clutter tests. Freeze held-out layouts before any tuning. |

Start with one read-only remote request on
`Qwen/Qwen3-VL-8B-Instruct` using the configured Caliban endpoint. Label its
effective quantization and decode settings unknown wherever the endpoint has
no authoritative receipt. Brief reproducible CPU geometry and scoped contract
tests do not reserve the GPU. Use the existing exclusive lease for GPU model
or renderer work. Run simulator and inference trials serially on their declared
resources. Retain infrastructure errors, timeouts, losses, and interrupted
attempts. When a repair changes the source, repeat the positive and negative
control under a new manifest or cohort.

Interpret actor history, fresh visual presence, admitted measured geometry,
actual motion, and final goal predicates individually. A refusal can pass its
own negative case while the task itself remains incomplete. Keep current and
historical images labeled and scorer-only object state upstream of scoring only.

## Native controllers across embodiments

Use the same public tools, evidence requirements and independent success
checks across robots. Each adapter sends admitted motion to its embodiment's
native controller and reports the joint, gripper and base feedback that the
controller exposes. Track simulated kinematic execution as a distinct test
condition. Add a skill and embodiment to the controller training track when
repeated native trials identify a control gap. Hold the fixture split, sensing,
admission rules and success criteria fixed when evaluating the learned policy.

### Port a benchmark arm

1. Add the benchmark, native robot, runner when available, frozen cases and asset IDs to
   the [transfer roster](../../configs/eval/observed_transfer_ports.json).
   Use the benchmark's supported embodiment. The first two bindings are
   PandaOmron for RoboCasa365 and Stretch for the in-house tabletop fixture.
   MolmoSpaces/rby1 is the next simulated arm to evaluate with this contract.
   Its native asset uses different joint, actuator and camera names from the
   current Galaxea-based backend, and drives its coupled fingers with motors.
   Qualify position feedback and the finger command loop on the native model.
   Select the robot for each additional benchmark when its task and
   controller requirements are known.
2. Implement `RobotBackend.bind_observed_arm` with stopped-base feedback,
   a robot-only kinematic model, and native joint and finger commands.
   Record controller limits, model fingerprint, synchronized camera/encoder
   timing, model-to-feedback pose residuals, and measured gripper arrival.
   Run a bridge smoke before connecting the agent. The
   [PandaOmron smoke](../../scripts/smoke_pandaomron_bridge.py) is a worked
   example. Mark unsupported commands independently of task success.
3. Feed `ObservedTransferSource` original RGB-D/encoder captures, a tracked
   payload, distinct pickup and destination supports, and calibrated robot
   collision material. Preserve unknown and conflicting volume. If a shape
   prior is needed, [calibrate it separately](../../scripts/calibrate_transfer_payload.py)
   from an isolated object asset or independent measurements. Check fit,
   pose uncertainty and tracking through occlusion. Record any excluded
   annotation geometry. Use
   `candidates_from_supports` with calibrated object-to-tool transforms
   and native finger targets. Bind the source and candidates through
   `observed_arm_factory`.
4. Use production CHAT `scene_tasks`, `plan_pick_place` and
   `execute_pick_place_plan`. Run the scripted routing first, then repeat with a
   real model and a held-out fixture. Keep scores for sensing, admission,
   pickup, carry, release and settled placement. Include the stale-frame
   and evidence-loss controls, measured motion, timing and resource use.
   The benchmark scorer owns simulator object and contact state. Keep
   native reference replay, live preflight and observed transfer as
   distinct assessment stages. Include the roster with
   `summarize_harness_assessment.py --transfer-ports configs/eval/observed_transfer_ports.json`
   to audit and export those assignments alongside the capability matrix.

### PandaOmron sink bridge

Launch the stopped-base RoboCasa365 profile with a compatible RoboCasa and
RoboSuite installation.

```sh
emet serve robocasa --robot pandaomron --headless \
  --robocasa-task PickPlaceCounterToSink --robocasa-layout 11 --robocasa-style 11
```

Connect through the shared robot client as `pandaomron`. Full observations carry
front and wrist RGB-D with camera calibration, capture time and measured joint
state. State packets include independent joint feedback and the gripper pose.
Arm and torso targets use seven angles in radians followed by the torso height
in meters. The bridge accepts binary and paired native finger targets and
exposes joint limits in the session. A successful command receipt acknowledges a target.
Use measured feedback to verify physical arrival.

Run the bridge smoke on dedicated loopback ports.

```sh
PYTHONPATH=src python scripts/smoke_pandaomron_bridge.py --output /tmp/pandaomron-bridge
```

Use `--sim-pythonpath` to point the simulator subprocess at a separately
pinned RoboCasa365 environment. The normal agent client can stay in the Emet
environment. This profile supports stopped-base arm, torso and gripper control.

To run production CHAT on the same archived counter-to-sink scene, pass the
preflight fixture directory and a fresh output directory.

```sh
PYTHONPATH=src python scripts/eval_pandaomron_sink_agent.py \
  "$archived_fixture" /tmp/pandaomron-sink-agent
```

The default scripted tool selection exercises the public photo, discovery and
planning gates. Supply `--model-endpoint` and `--model` to test model behavior.
Use `--scene-camera left` or `--scene-camera center` to compare RoboCasa's
other native scene cameras with the default right view. Each run exports the
scene camera and wrist RGB-D, calibration, and the encoder-derived robot-only
configuration. Use `eval_pandaomron_sink_observation.py` with recorded pixel
selections to compare measured support footprints and isolated IK.
The robot process publishes sensor packets while the scorer alone reads contact
and object state. A transfer pass requires the RoboCasa predicate, full basin
footprint, a measured two-pad lift and settling. The native arm and the
shared refiner are available. Supply `--observed-profile` to attach the live
RGB-D provider to production CHAT. The profile declares visual labels and
camera roles, isolated surface and payload geometry, grasp transforms,
robot geometry calibration, and the observed workspace. Explicit pixel
regions are recorded as diagnostic grounding. The runner exports measured
support bounds, payload fit, and the first evidence failure in its report.

The simulation-only contact fallback requires a matched `--probe-calibration`
file and a guarded placement profile. The bridge checks the commanded descent
and the virtual wrist wrench locally and holds on any terminal condition.
Calibrate and score the loaded wrist separately from the empty-wrist stop
smokes. The ordinary RGB-D placement path remains available without a probe
profile.

The native reference evaluator can replay sensor-derived shaker goals with
simulator-state collision supervision. It accepts the diagnostic JSON via
`--observed-shaker-goals`. Record the camera selection, approach variants,
lift height, and torso planning flags with the report. Label that run as
simulator-assisted and keep it outside the observed CHAT score.

## Discord in simulation

Use a dedicated test bot installed only in the private test server. Set
`EMET_DISCORD_CHANNEL_ID` to the operator-confirmed text-channel ID and keep
`DISCORD_TOKEN` outside the artifacts. Run `model-discord` with the surveyed
Stretch runner, model endpoint and dedicated ports described below. It uses
the production interactive CHAT loop and the read-only tool scope. Check
the startup log for the simulator identity and matching channel ID before
the operator takes a turn. Exact channel mode suppresses fallback and ignores
mentions from other channels. Ctrl-C ends the session and releases the simulator.

Start with a scene description, a photo, a visible-object query, and a
follow-up. Qualify bounded requests and cancellation in a subsequent session.
Save the operator transcript, actual tool calls, image delivery,
measured motion, refusals, and turn latency. Mark any message sent while a
tool is running as queued or handled according to the timestamped trace.
Treat the session as a supervised usability check.

For the bounded session, use `model-discord-drive` with the same private
channel and model settings. `look around` requests measured left and right
head views. `explore` or `drive 0.45 m` authorizes one guarded straight
waypoint. An exact `stop` message bypasses the model queue and disables
further drive for that session. Confirm `cancelled` and `stop_confirmed`
on the original navigation receipt. Run `operator-stop` locally without
Discord for the matched scripted stop control. It requests cancellation
after the encoders measure more than 7 cm of travel. Use the independent
chassis trace to check travel after stop.

Use `model-discord-full` for the general operator session, with a Stretch
scene that has collision geometry and movable objects. It uses the production
CHAT registry with its normal capability checks, DynaMem, instance perception
and visual servoing. The surveyed simulator supplies synchronized RGB-D and
encoder measurements. `show me your map` exercises `send_map_snapshot` using
the observed voxel map. Check the delivered PNG separately from the tool's
queued-upload receipt.

## Reproduce the surveyed Stretch navigation controls

```sh
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
  PYTHONPATH=src python scripts/eval_stretch_navigation_chat.py \
  --case clear --output /tmp/emet-stretch-clear --port-offset 2040
```

Run `clear`, `occupied`, `occupied-recovery`, `turn` and `stale`
sequentially with a fresh directory for each case. The runner builds the
authored fixture from hydrated Stretch assets, captures the launch survey,
and drives production CHAT with local scripted HTTP replies. Its measured
floor profile allows one targeted head view after an unknown-footprint
refusal, followed by the same route check. The scorer saves
`audit.json` beside the exact model requests, survey frames, command
receipts and held-out measurements. A zero process exit requires scoring
to pass. Review the source SHA and dirty-worktree flag before aggregating
results.

Run `witness-clear`, `slip` and `witness-loss` for the instrumented
live waypoint gate. These cases add a fixed RGB-D camera that observes
physical chassis markers in the simulator image. The runner checks paired
wheel and camera measurements from the same snapshot. The slip case
changes floor traction, and the witness-loss case withholds the independent
pose after motion starts. Inspect `motion-witness.jsonl`, the terminal
command receipt and the scorer's stop evidence.

## Real-model terminal controls

Run `model-full-chat` to exercise the full CHAT registry on the table scene
with terminal input. With hydrated Stretch assets, a configured model endpoint,
and the cached CLIP, YOLOE and MobileCLIP weights, run the same command used
for the surveyed controls with `--case model-full-chat --model-endpoint URL`.
This case runs the scene description, a spatial follow-up, an absent-object
query and a map request in one conversation. The normal map tool writes its
PNG to the local artifact directory through an image sink. Scoring checks
the answers, an actual populated map, current camera input and zero
actuation during the turns. The commands and images stay local when using
the loopback model server.

Run `model-object-recovery` on the same table fixture to verify memory and
reacquisition. The initial measured table view populates semantic memory.
The agent looks away, queries the earlier location while the object is out
of view, and returns to the table view. Scoring checks the remembered XYZ
against the fixture and uses RGB-D at the turn boundaries to verify that
the block disappears and reappears. The base stays stationary.

Run `model-table-transfer` for the first household task on the same full CHAT
registry. The instruction requests moving the red cylinder to the green region
while protecting the blue cube and keeping the base stationary. The scorer
reads simulator poses separately from the agent packets and checks lift,
footprint containment, stable support, protected-object motion and the
terminal receipt. A typed refusal with no actuation counts as a safe stop
and a failed transfer.

Run `model-read-only` with the surveyed Stretch runner first, then use
`model-drive` after the read-only audit passes. Supply `--model-endpoint`
with the OpenAI-compatible base URL including `/v1`. `--model` defaults to
`Qwen/Qwen3-VL-8B-Instruct`. Run serially with the same software-rendered
simulator settings above. If hosting the model locally, use the exclusive
GPU lease and record the checkpoint revision, precision and decoding.

The read-only turn exposes `send_image` and checks the model's answer against
the known visible object, along with zero actuation during the turn. The
motion turn exposes one 0.45 m request through the normal guarded waypoint
handler. Check the measured endpoint, typed command receipt and the final
model response. The scorer records the exact model calls and verifies that
the model received the motion result. Use a new output directory for each
attempt and preserve both HTTP packets and scorer results.

For tabletop admission, run `model-table-transfer --table-preflight-only
--table-sweep` without a model endpoint. It measures the green support patch
and red surface from calibrated depth, estimates the arm frame from the
synchronized motor sample, and rejects a deliberately expired capture.
The optional head sweep records a static multiview enclosure diagnostic.
Whole-payload bounds, pose tracking, self-collision geometry, and the
post-grasp stop remain separate execution requirements.

For changed-object recovery, `model-object-recovery-changed` moves the fixture
object while the camera faces away. The physics worker owns the intervention.
The robot client receives measured observations and the scorer checks the
hidden move from its independent trace. `model-object-recovery-revisit`
supplies the return head angles to isolate visual reacquisition from view
selection and metric grounding.

Run `scripts/smoke_sourccey_bridge.py --table-view --model-endpoint URL` to
exercise the Sourccey camera and turn control followed by read-only CHAT.
Score camera input, explored cells in a delivered map, and actuation during
the two CHAT turns. Retain the bridge's simulator-localization label.

## Stationary hardware

Schedule Stretch and Innate Mars only after the simulator review. Use explicit
robot profiles, local receive-only `emet preview-cameras --source zmq
--no-discord` and passive bridge observations. Audit progression, valid RGB,
available metric depth and corresponding calibration, pose/TF and sensor
dropouts. Keep unavailable streams visible in the results.
Use `scripts/probe_stationary_bridge.py --robot stretch` or
`--robot innate_mars` with the authorized observation host and port. The
probe opens a subscriber socket, retains the latest sensor packet, and reports
receive gaps and available acquisition-timestamp progression.
Run read-only model questions on captured packets and record these as hardware
image replay. Review the stationary runs and independent stop path before
scoping a motion session. See the [Mars operator guide](../robots/innate_mars_hardware.md).

## External benchmarks to add

1. Start with RoboCasa365 `PickPlaceCounterToSink` and `OpenDrawer`, then
   add a held-out kitchen and `CategorizeCondiments`. Use the
   benchmark's official success predicate and label any Stretch embodiment
   adaptation in the result. The benchmark accepts agentic submissions.
   [Atomic tasks](https://robocasa.ai/docs/build/html/tasks/atomic_tasks.html) · [Composite tasks](https://robocasa.ai/docs/build/html/tasks/composite_tasks.html) · [Leaderboard](https://robocasa.ai/leaderboard)
2. Borrow the partial-progress approach from Robocurve for our household
   cases. Record independently verified grasp, lift, delivery, stable release,
   and whether the neighboring object was disturbed. Report full completion,
   wall time, model calls and stop outcome per episode. The RoboDojo-RC
   results use adapted tasks on YAM arms, so use those results for protocol
   design and compare scores on a matched robot and fixture.
   [Robocurve task rubrics](https://robocurve.org/opus-5-5-robodojo-rc-tier-1/)
3. Consider RoboArena after hardware qualification. It runs pairwise trials
   on DROID and takes a remotely served robot policy. A DROID-compatible
   policy adapter would be separate from the Stretch operator session.
   [RoboArena](https://robo-arena.github.io/) · [Policy interface](https://github.com/robo-arena/roboarena)

The [Stretch pilot manifest](../../configs/eval/robocasa_stretch_pilot.json)
freezes the three tasks, the source revision, one development layout, one
held-out layout, and seeds. Run `scripts/eval_robocasa_stretch_preflight.py`
with the pinned RoboCasa source checkout and an isolated dependency overlay
using MuJoCo 3.3.1 and NumPy 2.2.5. Provision the corresponding RoboCasa assets.
The script renders each fixture, preserves the model and joint initialization,
and checks the upstream predicate against an initial state and a constructed
success state in scorer scratch data. Start execution trials after measured
reachability and the task's observed action requirements are verified.

## Evidence and paper figures

Run the analysis-only exporter from the frozen checkout:

```sh
PYTHONPATH=src python scripts/summarize_harness_assessment.py \
  --results /path/to/run/assessment-results.jsonl \
  --output /path/to/run/figure-inputs --plot
```

The exporter also accepts no results to materialize the assignments and
capability matrix. Pass `--recorded-geometry` and `--recorded-replay` with the
corresponding raw directories to render an example history success and a retained
abstention failure. The exporter cross-checks the replay and image hashes.
Pass `--active-trace` a completed authored CHAT case directory for the
sampled XY/yaw and turn-boundary camera panel.
Supply one JSONL row per evaluated assignment, using
`stage`, `id`, `status`, `source_sha`, and an `artifact` path relative to the
JSONL file or absolute. Include the independently checked result for
`passed` and `failed` rows as `independent_check`. Put the blocker in `reason`
for blocked rows. Leave timing unavailable when the run has no corresponding
clock. Optional timing keys are `wall_s`, `simulation_s`, `model_s`,
`planning_s`, `collision_s`, and `execution_s`. A `milestones` field maps
an independently scored milestone to true, false, or null for unmeasured.
The exporter writes milestone counts, failure reasons, and per-phase latency
figures. Retain attempt artifacts and export retries in distinct cohorts.

Retain the actual request packets with ordered images, scorer references,
source and fixture hashes, receipts, and short synchronized base/head traces
in run artifacts. Export the machine-readable coverage and outcomes first.
When the inputs exist, generate a top-down route with requested and achieved
poses and synchronized images, followed by a two-case recovery panel. Select
the lowest-ID eligible successful recovery and retained failure in each
frozen cohort. Keep the compact CSV, hashes, and figures under paper evidence
with a link to the raw run directory.

For observed native pick-and-place, the [agent-stack change map and arm transfer
recipe](observed_manipulation_stack.md) identify the shared planning, sensing,
contact, execution and CHAT changes, the Panda-specific calibration, and the
explicit simulation assistance. Keep full-task success separate from the
invalid-start recovery handoff regression.
