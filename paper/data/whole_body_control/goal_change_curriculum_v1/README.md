# Goal-change curriculum v1 launch evidence

Frozen controller source: `2a2ce5c1`. The archived runtime matches the canonical
controller/test input digest. The source archive and raw checkpoints stay ignored.
These compact files pin exact generation cases, PPO protocol and executable
producers; no hardware executor is enabled.

The pipeline runs the fresh full regression suite on CPU15 and 16-case teacher
and source-policy nominal-path admission on CPU14. PPO starts only after both
complete successfully. The admission requires nominal geometry, confirmed stops
and valid physics, not perfect teacher task success; safety/task failures remain
recorded. Budget: 131,072 additional decisions, 30/10/10/50 hand/base/goal/TAMP
physical ticks, actor/critics/Adam restored with new physical/RNG state.

At launch neither full-gate nor curriculum PPO results exist. Do not infer
success, qualification or promotion from a launch record. The live pipeline is
`logs/recovery/goal-change-curriculum-v1/pipeline.json`; native results and weights
are under `logs/manipulation/goal-change-curriculum-v1/`. Retained simulator viewer
is paused during the run and restored by the supervisor afterward.

See `docs/experiments/goal_change_curriculum_20261117.md` and the maintained
training-mixture/hardware-readiness documents. Herman's passive audit and offline
feedback patch are separate from this simulator-only experiment.
