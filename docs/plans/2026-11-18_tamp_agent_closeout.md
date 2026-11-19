# TAMP agent qualification

## Current status

| Work | Evidence and disposition |
|---|---|
| [#250](https://github.com/cpaxton/home_robot_v3/pull/250), assisted transfer and diagnostics | Ready for review as a development slice. Includes capture-bound collision maps, fresh-source path validation, bounded lateral release, checkpoints, and failed-tool bundles. The original expanded cohort completed **2/7**. The current release-focused suite retains four layouts, with the original seven outcomes preserved. |
| Cube-back placement | The pre-descent pilot stopped safely after grasp and carry. All 11 candidate destinations failed. It remains on the local `experiment/tamp-pre-descent-admission` branch at `8c8d03d3` pending a passing transfer and positive controls. |
| [#251](https://github.com/cpaxton/home_robot_v3/pull/251), collision-check cost | Draft stacked on #250. It adds an all-fresh/free AABB shortcut and reports 121 passing tests. Runtime benefit on the transfer workload still needs a matched comparison. |
| [#245](https://github.com/cpaxton/home_robot_v3/issues/245), observed execution | Capture, geometry, and contact components have evidence. A complete live observed-input transfer remains open. |
| Remote CHAT startup | Local `fix/tamp-remote-chat-startup` preserves `27dded16` for separate review and validation. |

The [transfer blocker catalogue](../experiments/mobile_transfer_blockers_20261118.md)
and [hard-case index](../../configs/eval/mobile_transfer_hard_cases_20261118.json)
give exact cases and replay inputs. The [observed admission development history](../experiments/observed_admission_development_20261116.md)
preserves capture, geometry, and prior integration evidence.

## Work in dependency order

1. **Review #250 and land the scoped changes.** Review partial receipts,
   unknown-space rejection, and the evidence boundaries. Use the two successful
   layouts as positive controls. Preserve the historical failures and keep the
   unqualified pre-descent pilot separate.
2. **Measure and qualify #251.** Replay cylinder right and time solid creation,
   map admission, arm paths, base sweeps, and release separately. Require
   collision-decision parity before timing comparisons. Rerun both cylinder
   cases with the same 299-second source lifetime and sweep spacing. Cache
   validated local geometry if the profile shows it dominates. Existing code
   already caches hull planes.
3. **Resolve placement and source coverage.** Replay cube-back with the saved
   destination scan. Identify the solid and height that first fail during the
   sampled fall and retreat. Test a different certified destination or a
   calibrated view that covers the missing volume. Compare candidate ranking
   by support footprint and hand/retreat clearance on frozen inputs. Require
   cube-back and a previously passing control to complete before enabling the
   stricter preflight. For cube left and front, replay the starting payload
   check with a second view and expose the geometry rejection in the tool
   receipt. For cylinder left, distinguish unknown support from expiry.
4. **Qualify live observed input.** Bind synchronized simulated Stretch RGBD,
   measured robot joints, calibrated robot and target envelopes, support and
   background evidence, and the motion refiner. Require a stationary
   public-tool transfer, a stale-source preflight stop, and a post-grasp
   evidence-loss control. Then add a stopped base handoff with fresh
   destination evidence. Use independently measured outcomes for scoring.
5. **Qualify agent composition.** Use the same `run agent` entry point and remote
   model configuration on reference-certified MuJoCo episodes. Start with one
   transfer and an explicit recovery turn. Then test protected multi-object
   goals, swaps, authorized clutter clearing, and exploration or navigation
   for missing evidence. Report full goals, partial progress, tool and model
   calls, search time, and failures on held-out layouts before updating
   paper claims.

Parallel PRs are #228 (Stretch model fingerprint and clock), #229 (grounding
feedback), #241 (draft controller tracking), and #230 (interaction memory and
matched model evaluation). Review those in their own scope. Test #229 feedback
at the observed-agent gate and use #230 for matched evaluation once execution
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
