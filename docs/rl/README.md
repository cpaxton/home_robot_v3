# Reinforcement learning and whole-body control

For current ownership, the linear review stack, training status and the next
milestone, use [Active Mars controller work](ACTIVE_WORK.md). Continue work in
the canonical thread/worktree listed there rather than an older parallel thread.

Start with [whole-body control philosophy](whole_body_control.md) for the task
contract, controller boundaries, training approach and evidence needed to adopt
a learned controller. Mars is the first target; the same pose contract motivates
other embodiments, with their own actions, models and qualification.

## Reading map

| Need | Document |
| --- | --- |
| Current owner, branches, training gate and next milestone | [Active Mars controller work](ACTIVE_WORK.md) |
| Architecture, historical results and research comparison | [2026-11-14/15 snapshot](status_20261114.md) |
| Ordered changes, decision criteria and bounded next experiments | [Mars WBC next steps](../plans/mars_wbc_next_steps.md) |
| WBC design principles and lessons | [Whole-body control philosophy](whole_body_control.md) |
| Simulator setup, CLI and runtime details | [Mars reactive RL runbook](../mars_reactive_rl.md) |
| General goals, changing references and recovery | [General whole-body tracking](../plans/general_whole_body_tracking.md) |
| Finite tracking problem and per-case acceptance | [Tracking mastery](../plans/mars_tracking_mastery.md) |
| Component initialization and one-actor export | [Arm-to-whole-body transfer](../plans/mars_arm_to_whole_body.md) |
| Refinement hypotheses and follow-up protocols | [Refinement plan](../plans/mars_refinement_hillclimb.md) |
| GRU / multi-critic training prototype | [Whole-body v2 plan](../plans/mars_whole_body_v2.md) |
| Public tools, agent integration and hardware gates | [Controller integration](../plans/mars_controller_integration.md) |
| CPU reservations, Caliban and simulator parity | [Parallel simulation](../plans/mars_parallel_simulation.md) |
| Preview velocity, orientation features and frozen-base pilot | [Experiment report](../experiments/mars_reference_velocity_prototype.md) |
| Completed frozen-base/context pair | [2026-11-15 results](../experiments/mars_frozen_base_arm_20261115.md) |
| Completed GRU credit comparison and base interference | [C1 results](../experiments/mars_c1_normalization_recovery_20261115.md), [command diagnostic](../experiments/mars_c1_zero_base_diagnostic_20261115.md) |
| Base exploration and actor-active settling | [Bounded GRU comparison](../experiments/mars_reference_base_exploration_20261115.md) |
| Actual grasp-stage progression and reactive hold recovery | [Task-level diagnosis](../experiments/mars_task_arrival_20261115.md), [bounded protocol](../plans/mars_task_arrival_diagnostic.md) |
| Unloaded grasp-pose training boundary and teacher checks | [Stage-admission protocol](../plans/mars_unloaded_grasp_admission.md), [5/6 readout and planning gap](../experiments/mars_unloaded_grasp_admission_20261116.md) |
| Near-term work | [RL TODOs](../../TODO.md#rl-controller-integration-and-acceptance-mars-first) |

Use this directory for enduring design guidance. Keep proposed changes and
bounded protocols in `docs/plans/`, measured outcomes in `docs/experiments/`,
and compact evidence in `paper/data/whole_body_control/`. Detailed traces and
weights stay in ignored artifact storage. Link those records rather than
turning this index into another experiment log.

Run status and passing software tests are separate from measured policy
improvement. The experiment report records which options were implemented,
which were trained, failed gates and whether a candidate was retained.
