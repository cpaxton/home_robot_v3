# TAMP agent qualification

## Current status

| Work | Evidence and disposition |
|---|---|
| [#250](https://github.com/cpaxton/home_robot_v3/pull/250), assisted transfer and diagnostics | Merged. Includes capture-bound collision maps, fresh-source path validation, bounded lateral release, checkpoints, and failed-tool bundles. The original expanded cohort completed **2/7**. The current release-focused suite retains four layouts, with the original seven outcomes preserved. |
| Cube-back placement | The pre-descent pilot stopped safely after grasp and carry. All 11 candidate destinations failed. It remains on the local `experiment/tamp-pre-descent-admission` branch at `8c8d03d3` pending a passing transfer and positive controls. |
| [#251](https://github.com/cpaxton/home_robot_v3/pull/251), collision-check cost | Merged. The assisted cylinder-right replay matched 50,000 admission decisions. Admission took 16.06 s on the base and 15.69 s with the shortcut. Both full searches expired after about 275 s. Time to a certified plan remains open. |
| [#245](https://github.com/cpaxton/home_robot_v3/issues/245), observed execution | Capture, geometry, and contact components have evidence. A complete live observed-input transfer remains open. |
| Remote CHAT startup | Preflight the frozen source with a read-only remote turn. If startup fails, validate the local `fix/tamp-remote-chat-startup` repair at `27dded16` separately and record the source actually used. |

The [transfer blocker catalogue](../experiments/mobile_transfer_blockers_20261118.md)
and [hard-case index](../../configs/eval/mobile_transfer_hard_cases_20261118.json)
give exact cases and replay inputs. The [observed admission development history](../experiments/observed_admission_development_20261116.md)
preserves capture, geometry, and prior integration evidence.

## First agent test

Use a clean, frozen main revision and the existing
[prospective agent protocol](2026-11-16_tamp_prospective_evaluation.md) and
[contextual recovery control](../experiments/tamp_contextual_recovery_20261116.md)
as references. Pin the selected fixture and reset, exact remote model and
endpoint, tool-call and wall-time budgets, and artifact directory before
launch. Run the [TAMP contract checks](../TESTING.md#tamp-contracts) on that
source and verify one read-only remote CHAT turn before enabling motion.

1. In a reference-certified Stretch/MuJoCo table scene, have the model use
   `emet run agent` and the public discovery, planning and execution tools for
   one stationary transfer. Require a successful tool receipt and independently
   verify the object is supported, inside the goal region and settled.
2. Run the stop-only and explicit retry controls with the declared preflight
   refusal. Require no commands from the refused step and rejection of the
   consumed handle. In the retry case, send a separate authorization and check
   that fresh discovery and planning complete only the remaining goal.
3. Save actual tool selections and receipts, independent goal checks, command
   ranges, source and fixture identifiers, assistance mode, and failures.
   Record elapsed wall and simulator time when available, plus model and
   planning time. Label this native simulated control with its privileged inputs.

The native control can run while placement and observed-admission work
continues. Repeat the single-transfer agent test with observed inputs after the
stationary observed public-tool controls pass.

## Remaining execution qualification

1. **Reduce time to a certified plan.** Replay cylinder right and time solid
   creation, map admission, arm paths, base sweeps, and release separately.
   Require collision-decision parity before timing comparisons. Rerun both
   cylinder cases with the same 299-second source lifetime and sweep spacing.
   Cache validated local geometry if the profile shows it dominates. Existing
   code already caches hull planes.
2. **Resolve placement and source coverage.** Replay cube-back with the saved
   destination scan. Identify the solid and height that first fail during the
   sampled fall and retreat. Test a different certified destination or a
   calibrated view that covers the missing volume. Compare candidate ranking
   by support footprint and hand/retreat clearance on frozen inputs. Require
   cube-back and a previously passing control to complete before enabling the
   stricter preflight. For cube left and front, replay the starting payload
   check with a second view and expose the geometry rejection in the tool
   receipt. For cylinder left, distinguish unknown support from expiry.
3. **Qualify live observed input.** Bind synchronized simulated Stretch RGBD,
   measured robot joints, calibrated robot and target envelopes, support and
   background evidence, and the motion refiner. Require a stationary
   public-tool transfer, a stale-source preflight stop, and a post-grasp
   evidence-loss control. Then add a stopped base handoff with fresh
   destination evidence. Use independently measured outcomes for scoring.
4. **Expand agent composition after the single-transfer gate.** Test protected
   multi-object goals, swaps, authorized clutter clearing, and exploration or
   navigation for missing evidence. Report full goals, partial progress, tool
   and model calls, search time, and failures on held-out layouts before
   updating paper claims.

The Stretch fingerprint cache and grounding-range feedback are already on
main. Capture the range rejection at the observed-agent gate if it occurs.
Controller tracking (#241) and the interaction-memory plan (#230) are parallel
work. Use the interaction-memory plan for matched evaluation once execution
is stable.

## Transfer rules and replay

| Condition | Tool contract and recovery |
|---|---|
| Unknown starting or swept volume | Keep the volume blocked and report the evidence failure at the affected phase. Seek a calibrated view from a checked viewpoint, then replan. |
| Shared start rejected | Check it once and skip dependent candidates. Independently checked actions remain available. |
| Placement exit rejected | Check descent, opening, fall, and retreat before selecting the destination. Try another candidate inside the goal region and budget. Recheck measured state before opening. If no candidate passes after grasp, report partial progress and inspect. |
| Stale evidence | Retire the old authority. Validate a retained candidate against a fresh capture or search again at a checked, stopped state. Track expiry separately from numerical and geometry failures. |
| Occupied destination | Search other permitted destinations. Clearing a blocker is a separate authorized task step. |
| Uncertain command outcome | Consume the handle, preserve verified progress, and inspect measured state before a new command. |

Pair each regression with a passing control. Preserve the relevant simulator
checkpoint, capture and map generation, first rejection, and command range.
Full agent restore still needs conversation, memory, controller state, goal,
verified progress, and random state saved at a quiet tool boundary. Restored
execution acquires fresh evidence and new handles.

Main's [agent architecture](../agent_architecture.md),
[tool contracts](../agent_tool_contracts.md), [tool index](../agent_tools.md),
and [operator guide](../AGENT_RUN.md) define the shared interface.
[Observed TAMP API](../api/observed_native_tamp.md) owns native evidence and
receipt details. The [household benchmark plan](2026-10-22_mujoco_tamp_benchmarks.md)
owns the broader experiment design.
