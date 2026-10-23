# Stack integration checkpoint, 2026-10-23

## PR #199 against merged main

Frozen main `a2872951` includes #213 and #222. Candidate `245b80ec` includes
that main and the reviewed evidence-packet repairs. Job
`20261023_183204_08caf7` ran serially with the same local Qwen model, legacy
decision policy, seed 0, and budgets on both arms.

- EQA q12/14/6/56: main 1/4, candidate 1/4; identical predicted answers.
- S0 FIND: main unsuccessful, candidate successful.
- Molmo rby1-proxy FIND: both unsuccessful.
- All requested artifacts completed. The job exited 1 because the room driver
  requires all task cases to succeed; this is a task-acceptance failure, not an
  infrastructure crash. These checks do not qualify manipulation or prove
  general accuracy superiority.

Artifacts: `~/runs/emet/pr199-post-tamp-{main,candidate}-eqa-20261023`
and `~/runs/emet/pr199-post-tamp-room-pair-20261023`.

## PR #202 integration

Merge #199 without dropping image acquisition timing, received-depth freshness,
or sensor/estimated depth provenance. Keep selector-supplied labels and roles;
append admitted FIND/crop roles with the actual image positions. Preserve
episode-local packet and metadata retention together.

170 focused selector, acquisition-history, captured-view, graph-answer and VLM
assessment tests pass after resolving the integration conflicts. Added a
regression for labeled selector packets combined with explicit FIND and crop.
Broader map/provenance checks and live comparison are still pending.

Keep history ranking opt-in. Advance #216 and #217 only after refreshing their
dependency and rerunning their own answer/dialogue contract tests. Passing this
small EQA comparison is a non-regression signal, not proof of recovered overall
agent performance or a claim that all acceptance gates are green.
