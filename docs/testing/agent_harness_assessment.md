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

## Stationary hardware

Schedule Stretch and Innate Mars only after the simulator review. Use explicit
robot profiles, local receive-only `emet preview-cameras --source zmq
--no-discord` and passive bridge observations. Audit progression, valid RGB,
available metric depth and corresponding calibration, pose/TF and sensor
dropouts. Keep unavailable streams visible in the results.
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
clock. Optional timing keys are `wall_s`, `simulation_s`, `model_s`, and
`planning_s`. Export retries separately so one case cannot overwrite another.

Retain the actual request packets with ordered images, scorer references,
source and fixture hashes, receipts, and short synchronized base/head traces
in run artifacts. Export the machine-readable coverage and outcomes first.
When the inputs exist, generate a top-down route with requested and achieved
poses and synchronized images, followed by a two-case recovery panel. Select
the lowest-ID eligible successful recovery and retained failure in each
frozen cohort. Keep the compact CSV, hashes, and figures under paper evidence
with a link to the raw run directory.
