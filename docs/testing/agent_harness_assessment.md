# General-purpose agent harness assessment

The first milestone is a capability audit across Stretch/MuJoCo, Molmo rby1,
and Habitat. Start from the main revision recorded in the
[case manifest](../../configs/eval/harness_assessment_20261119.json). Commit
the assessment driver and freeze its SHA before launching a scored cohort.
Use a clean checkout, a new artifact directory per attempt, and the existing
managed jobs policy. Keep simulator, replay, and hardware results distinct.

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

Use the Stretch Discord preset with an explicit loopback connection and
dedicated nondefault ZMQ ports. Launch the default-table simulator with a
fresh seed-0 reset, use the production `emet run agent` in interactive mode,
and check the bridge's simulator identity before handing off to the operator.
Use a dedicated test bot installed only in a private test server with the
expected channel. Confirm guild membership and the exact `EMET_DISCORD_CHANNEL`
before launch. The current bot can fall back to the first channel of a guild
when the requested channel is missing. Keep tokens outside the artifacts.

Start with a scene description, a photo/map, a visible-object query, and a
follow-up. Then exercise bounded requests and cancellation within the
simulator. Save the operator transcript, actual tool calls, image delivery,
measured motion, refusals, and turn latency. Mark any message sent while a
tool is running as queued or handled according to the timestamped trace.
Treat the session as a supervised usability check.

## Stationary hardware

Schedule Stretch and Innate Mars only after the simulator review. Use explicit
robot profiles, local receive-only `emet preview-cameras --source zmq
--no-discord` and passive bridge observations. Audit progression, valid RGB,
available metric depth and corresponding calibration, pose/TF and sensor
dropouts. Keep unavailable streams visible in the results.
Run read-only model questions on captured packets and record these as hardware
image replay. Review the stationary runs and independent stop path before
scoping a motion session. See the [Mars operator guide](../robots/innate_mars_hardware.md).

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
