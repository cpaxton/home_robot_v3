# Non-RL integration results, 2026-10-26

Frozen candidate `35bdab14`, main `61da55cb`, managed job
`20261026_090114_e985ce`. Artifacts:
`~/runs/emet/nonrl-harness-integration-20261026`. All workload phases ran serially.

| Check | Main | Candidate |
| --- | --- | --- |
| EQA12, seed 0, same local Qwen/budgets, native answer policies | 7/12 | 9/12 |
| S0 FIND | Pass | Pass |
| RoboCasa FIND | Infrastructure: missing worktree asset links | Same |
| Molmo FIND | Object and receptacle unlocalized | Object localized, receptacle unlocalized |
| Scripted physical two-goal TAMP | Pass | Pass |
| Live CLI multi-goal TAMP | Invalid instruction, stopped before manipulation | Same |

Candidate contracts: 272 passed, zero failures/errors/skips. Candidate floor-on
S0 FIND: pass. The combined gate is red. EQA is a small development result with
different native answer/history policies, not an isolated ablation or paper claim.

## Correcting the test setup

The RoboCasa asset precheck resolved paths relative to each isolated checkout,
where ignored external asset directories were absent. Both worktrees now link
only `third_party/robocasa/robocasa/models/assets` to the existing installed pack.
Basic fixtures, registry layout, LightWheel registry and Objaverse bounds checks
all pass. No download or runtime source change is needed. The integration driver
now checks this prerequisite before the long EQA phase.

The live TAMP instruction used a natural-language conjunction and articles:
`Put the red cylinder in cylinder goal and the blue cube in cube goal.` Both
models copied that string into `plan_task`, whose compiler advertises semicolon-
separated clauses and exact discovered roles. Both returned `unknown_task_role`;
both obeyed the failure-stop boundary with zero forbidden commands. This is not
physical manipulation failure, but neither trial qualifies physical execution.
The corrected control uses `move red cylinder to cylinder goal; move blue cube
to cube goal`, verified against the actual compiler in a regression test. This
changes the control input identically on both arms, not the planner, tolerances
or model answer. Preserve the original failures and score reruns separately.

## Remaining Molmo blocker

The candidate sees the microwave and describes it above the stove, but has no
admitted localized geometry. Measured rotation is rejected before execution due
to unknown floor under the swept footprint. First floor recovery observes 315
new cells and reduces the unknown footprint from 18 cells to 7. A second recovery
observes 8 new cells but leaves all 7 unknown; the targeted cell remains below
the camera (projected v≈527 in a 480-pixel image) at the measured tilt limit.
No navigation step executes. This is a real coverage/motion deadlock, not evidence
that the final model answer should be replaced or that missing floor is free.

Do not waive it into a green all-stack gate. Measured footprint support and a
qualified viewing/motion recovery remain follow-up work; no larger visited disk,
unknown-space exemption or widened head range is introduced here.

## Evidence reuse

Setup repairs affect the orchestration script, its tests and documentation only.
EQA/contract/runtime evidence remains tied to the original frozen source. Retry
RoboCasa and live TAMP in fresh output directories with explicit source and input
provenance. A partial rerun does not overwrite or relabel the original red gate.
