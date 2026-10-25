# Stack integration checkpoint, 2026-10-23

## PR #202 landing checks, 2026-10-25

PR #199 landed as `495f7097`. PR #202's history-only merge `dc77214d`
has the same source tree as tested `2d2f2fae`.

- EQA q12/14/6/56, seed 0, same local model and legacy answer policy:
  #199/main 1/4, #202 2/4. Only q56 changes from wrong to correct.
- Paired S0 FIND: both succeed. Paired Molmo FIND: both fail.
  Job `20261024_134747_ade824` produced all artifacts; its nonzero exit
  reflects the shared task failure, not missing output or a runtime crash.
- Extra-history ranking stayed opt-in; this does not establish superiority
  of any ranking policy. The shared Molmo failure remains follow-up work.
- Review repairs reject boolean history timestamps, uniquely number captures
  sharing a timestamp using view ID order, and evaluate candidate features once
  per selection round. Duplicate view IDs remain rejected at the API boundary.
- 183 focused selector/history/packet/map tests pass. No default policy flip,
  real-robot qualification, or full manipulation acceptance is claimed.

Artifacts: `~/runs/emet/pr202-refreshed-eqa4-20261023` and
`~/runs/emet/pr202-landed-main-room-pair-20261024`.

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
