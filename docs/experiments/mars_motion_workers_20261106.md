# Whole-body training supports independent native workers

The current motion trainer now accepts `--num-envs 1..8` and
`--vec-backend dummy|subproc`. Each subprocess constructs its own MuJoCo
world and frozen source policy; live simulator pointers and loaded parent
policies are never sent to child processes. Inference still exports one
ordinary eight-output actor. The default remains one worker.

Each worker retains a 256-tick rollout horizon. Thus eight workers collect
2,048 transitions per PPO rollout, with batch size 256 unchanged. Total steps,
evaluation interval and critic warmup must be multiples of `256 * num_envs`;
invalid boundaries fail before evidence/model/output access. Sampling and
reward diagnostics count every worker and each worker's episode starts,
including automatic resets. Contract curriculum advancement counts all
transitions. More workers change the experience/update distribution, so
multi-worker training is a separate trial, not an identical-seed comparison.

All 68 focused tests pass. Two native dummy worlds reproduce the old serial
reset and three control steps exactly, maintain separate physics data, update
both the base and whole-body policies through PPO, and export matching actions.
The fresh full source-bound regression gate passes all 1,095 tests.

Two bounded Caliban spawn probes then execute the actual worker factory,
context-enabled PPO and export: two workers contribute 32 transitions each;
eight workers contribute 64 each. Every worker contributes, exported motion
actions match exactly, and no recorded initialization/unsafe-contact/wheel-lift/
invalid-physics/retention flag trips. These use a new synthetic zero-mean source
created on ARM, **not** the retained trained checkpoint. The reported times
(11.93 s and 16.68 s including setup) are not a speedup benchmark or evidence
of learned tracking quality. [Compact reports and source hashes](../../paper/data/whole_body_control/mars_motion_workers_v1/results.json)
retain both outcomes.

This clears worker/PPO plumbing, not Caliban training qualification. Its two
[frozen public-tool fixture failures](mars_caliban_portability_20261106.md)
remain unresolved. Do not bypass the full regression/teacher/admission gates or
relax fixture tolerances to launch an eight-worker refinement. The workstation
remains one CPU14 simulation/training worker and one CPU15 diagnostic worker;
active policy trials run in immutable serial checkouts. No hardware was used.
