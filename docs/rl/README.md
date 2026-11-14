# Reinforcement learning and whole-body control

Start with [whole-body control philosophy](whole_body_control.md) for the task
contract, controller boundaries, training approach and evidence needed to adopt
a learned controller. Mars is the first target; the same pose contract motivates
other embodiments, with their own actions, models and qualification.

## Reading map

| Need | Document |
| --- | --- |
| WBC design principles and lessons | [Whole-body control philosophy](whole_body_control.md) |
| Simulator setup, CLI and runtime details | [Mars reactive RL runbook](../mars_reactive_rl.md) |
| General goals, changing references and recovery | [General whole-body tracking](../plans/general_whole_body_tracking.md) |
| Finite tracking problem and per-case acceptance | [Tracking mastery](../plans/mars_tracking_mastery.md) |
| Component initialization and one-actor export | [Arm-to-whole-body transfer](../plans/mars_arm_to_whole_body.md) |
| Refinement hypotheses and follow-up protocols | [Refinement plan](../plans/mars_refinement_hillclimb.md) |
| Separate GRU / multi-critic prototype on the parent branch | [Whole-body v2 plan](https://github.com/cpaxton/home_robot_v3/blob/feat/mars-base-isolation/docs/plans/mars_whole_body_v2.md) |
| Public tools, agent integration and hardware gates | [Controller integration](../plans/mars_controller_integration.md) |
| CPU reservations, Caliban and simulator parity | [Parallel simulation](../plans/mars_parallel_simulation.md) |
| Preview velocity, orientation features and frozen-base pilot | [Experiment report](../experiments/mars_reference_velocity_prototype.md) |
| Near-term work | [RL TODOs](../../TODO.md#rl-controller-integration-and-acceptance-mars-first) |

Use this directory for enduring design guidance. Keep proposed changes and
bounded protocols in `docs/plans/`, measured outcomes in `docs/experiments/`,
and compact evidence in `paper/data/whole_body_control/`. Detailed traces and
weights stay in ignored artifact storage. Link those records rather than
turning this index into another experiment log.

Run status and passing software tests are separate from measured policy
improvement. The experiment report records which options were implemented,
which were trained, failed gates and whether a candidate was retained.
