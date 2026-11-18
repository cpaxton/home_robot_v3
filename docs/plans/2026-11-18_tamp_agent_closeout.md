# TAMP agent integration and qualification

Main at `f1b7bb5b` includes #247–249. Use [agent architecture](../agent_architecture.md), [tool contracts](../agent_tool_contracts.md), [tool index](../agent_tools.md), and [run agent](../AGENT_RUN.md) as the current interface documentation. The TAMP branch is already based on that revision. Its explicit gaze invalidation fix is documented there. Keep tool behavior in those pages, native input details in [observed TAMP](../api/observed_native_tamp.md), and measured outcomes in experiment records and the paper appendix.

1. **PR #250, placement and observed admission.** Clarify the expanded layouts as nominal footprint tests with intentional boundary contact. Run all seven layouts on the same revision with fresh-source validation, and retain all attempts. Score full completion, measured settled support, failure stage, partial receipts, and map lifetime. Previous post-refactor successes include clutter, shifted cylinder and larger cylinder. They do not constitute the new seven-case cohort. Review and merge #250 first.
2. **Stacked search-cost PR, proposed by DeepSeek.** Profile arm, base sweep, and release queries on a frozen case, with timing for solid construction and admission. Cache validated local geometry only if measurements support it. Hull planes are already cached. Add a conservative fresh/free AABB shortcut after evidence validation and preserve the detailed query on a miss. Check stale cells, boundary contact, unknown, occupied, support rays and outside-map behavior. Compare decisions on identical replay inputs, then report wall time and expiry counts on the same layout cohort. Keep existing sweep spacing. Base this PR on #250 until it lands.
3. **Observed execution.** Connect the live simulated Stretch RGBD and measured motor stream to capture-bound robot/target material, independent support/background evidence, and the contact refiner. Verify a stationary public-tool transfer first, with stale/no-command and post-grasp loss controls. Then qualify a stopped base handoff and observed goal grounding. Record the actual sensor and simulator assistance used.
4. **Agent composition.** Freeze solvable MuJoCo tasks with independently verified reference executions. Run the same `run agent` entry point and remote model configuration across native assisted and qualified observed inputs. Start with single transfers and explicit recovery, then protected two-object goals, swaps, clutter clearing when required, and exploration/navigation to acquire missing evidence. Require observed goal verification, correct partial feedback, and a new handle before an authorized retry. Report task and step completion, failure stage, unnecessary moves, model calls, planning time and evidence expiry. Keep reference construction failures visible. Use fixed cases and held-out layouts before updating paper claims.

Other open dependencies and parallel work:

- **#229** preserves query-grounding rejection reasons. Review and exercise the feedback in the observed agent gate.
- **#228** caches the Stretch native model fingerprint. Review its clock and model identity tests independently of the collision-query optimization.
- **#241** is draft controller work for coupled tracking and native feedback. Retest against measured failures when it lands. It is a parallel track for assisted agent qualification.
- **#230** plans interaction-grounded memory and matched model evaluation. Review separately and use the matched evaluation design once execution gates are stable.

The two DeepSeek work items are #250's layout expansion and the proposed search-cost follow-up. The latter has no open PR in the current inventory.

## Completed expanded layout baseline

The seven provided-map runs finished at **2/7**. Cube right and far completed.
Cube left and front both stopped before IK when the map could not certify the
starting payload volume. The public budget code obscures that rejection.
Cylinder right stopped on stale evidence and cylinder left reported unknown
support geometry. Both consumed about 299 s across map construction and search.
The left-cylinder trace also includes a rejected destination with finger contact
against the distractor. Resolve the support rejection independently of timing.
Cube back completed grasp and stopped at release with an invalid measured start
for the adjustment path.

Use separate follow-ups for these three mechanisms:

- Profile and accelerate collision checks on cylinder right. Replay identical
  checks for parity and rerun both cylinder layouts with the original lifetime.
- Replay the left/front source captures and isolate the unknown payload volume.
  Test a second view at the same starting state, and expose an actionable public
  preflight cause if the evidence remains insufficient.
- Replay the cube-back saved release state. Identify the first rejecting solid
  and evidence source before changing placement selection. If the terminal pose
  leaves no certified correction, select an alternative earlier while the load
  is still at the checked preplace pose.

Artifacts are under `.tamp-runs/expanded-clutter-20261118/`. This baseline
is separate from earlier controls. Keep all seven cases in the next comparison.

## Checkpoints and simpler candidate selection

The mobile harness saves MuJoCo integration state alongside scans and release
diagnostics. Restore into a private simulator using the original fixture and
runtime. Retain the resolved fixture, command prefix, decision phase, and capture
artifacts together. Before relying on a replay, compare the next measured
states and tool receipts against the original.

For an agent checkpoint, save at a quiet tool boundary. Bundle the simulator
and controller state, agent conversation and model settings, memory checkpoint,
task goal and verified progress, and RNG state. Keep pending commands and their
outcomes in the trace. Recreate the session on restore, acquire fresh evidence,
and rediscover/replan before sending commands. Historical captures and tool
receipts may be replayed for offline comparisons at their recorded times.
Executable handles from the prior session cannot authorize restored motion.

Use the source and release checkpoints to compare candidate samplers on the
same state. Precompute robot-frame geometry and deterministic sample templates.
Rank destination candidates by payload footprint and conservative room for the
opening hand and retreat before spending IK on them. Certify the selected path
with the existing detailed geometry and evidence checks. Compare candidate
count, time to the first certified path, and final decisions. Begin with failed
cube-back release and cylinder-right source planning, then rerun all seven
layouts.

The [transfer blocker catalogue](../experiments/mobile_transfer_blockers_20261118.md) indexes each failure and its replay inputs.
