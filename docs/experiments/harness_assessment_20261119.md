# First general-purpose harness assessment

The [assessment protocol](../testing/agent_harness_assessment.md) inventories
44 registry entries and freezes 57 case assignments by adapter and evidence
tier. Its frozen runtime is `c9a8cf413aa181be61d2538359a68e11805aec6f`.
The terminal-feedback evaluator recheck uses
`91304072380277f00115c16d3705dfbe983da2a1`.

| Check | Result |
| --- | --- |
| Focused CPU contracts | 490 passed. |
| TAMP workflow's CPU test list | 421 passed. |
| Recorded geometry | 12/12 controls passed and 24 views rendered. |
| Remote model replay | Full history 3/6, current only 1/6. Zero inference errors. Both arms guessed on the absent and ambiguous controls. |
| Privileged stationary Stretch agent | 0/1 task. The model called discovery, skipped planning and supplied an invented plan handle. The executor rejected it with `unknown_plan` and stopped. The cylinder stayed supported outside the goal. Seven audit entries, zero forbidden commands. |
| Live Stretch seen-recall case | Three turns completed and five HTTP requests were captured. The answer was green, and the box remained visible. |
| Live Stretch unseen-recall case | Three turns completed. The model correctly declined to name a color for the absent box. |
| Measured turn in both active cases | Both 90-degree requests were rejected on an unobserved footprint. Floor sensing reported `floor_target_unobservable`. Measured yaw stayed within 0.001 degrees of the earlier sample in both cases. |

The first privileged run took 72.43 wall seconds and 36.73 measured simulator
seconds. The replay uses remote Qwen3-VL-8B-Instruct with unverified server
precision and decoding. The replay corpus is development-only and so is the
active Stretch fixture. The state-only trial exercises the CLI and guarded
physics boundary with privileged object inputs. It gives no observation or
manipulation success credit.

The original stop-only run also reached a typed `unknown_plan` refusal with a
forced final response. Its evaluator was still waiting for the expected partial
result, so the job was cancelled after retaining the logs. The evaluator now
ends on unexpected terminal feedback, and a regression checks this path before
any retry authorization. The recovery recheck is a separate cohort. Both trials finished with
`unexpected_terminal_feedback:execute_task_plan:unknown_plan`. Neither
reached the specified partial-transfer boundary or received a recovery turn.

See the [coverage and per-case figure inputs](../../paper/data/harness_assessment_20261119/README.md).
The raw run root is `/tmp/emet-harness-run-20261119` on the assessment host.
Managed job receipts preserve the commands and timestamps. Discord awaits an
explicit private test channel. Stationary hardware and observed execution
remain gated by the simulator review and #245 respectively. Follow the scoped
Stretch/Molmo find and Habitat jobs through `20261119_170502_21b235`.

The active cases give evidence for camera transport, absent-target handling,
and no-dispatch motion refusal. The changed-view recall gate remains open.
The next isolated navigation comparison is a scripted feasible rotation
with qualified footprint coverage versus the fixed unknown-footprint
refusal, followed by the same agent requests. Keep broad downward
inspection and targeted floor projection separately measurable.
