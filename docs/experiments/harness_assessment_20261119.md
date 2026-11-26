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
Managed job receipts preserve the commands and timestamps. Discord awaits an
explicit private test channel. Stationary hardware and observed execution
remain gated by the simulator review and #245 respectively.

The active cases give evidence for camera transport, absent-target handling,
and no-dispatch motion refusal. The changed-view recall gate remains open.
The next isolated navigation comparison is a scripted feasible rotation
with qualified footprint coverage versus the fixed unknown-footprint
refusal, followed by the same agent requests. Keep broad downward
inspection and targeted floor projection separately measurable.

## Recovered cross-simulator assessment

Job `20261119_170502_21b235` recorded five completed verdicts before the host
reboot at 17:50 EST. The available journal does not establish the cause.

| Case | Recovered result | Evidence |
| --- | --- | --- |
| Stretch S0 find | Object and destination gate passed | Supervisor log |
| Molmo rby1 find | Finished and missed the joint object and destination gate | Supervisor log |
| Habitat HM-EQA q2/q6/q12 | 2/3 correct. q6 and q12 passed. | Original episode bundles and model-final metrics |
| Habitat nearest-v2 find | Interrupted, no task verdict | Missing final metric after reboot |

The reboot cleared the original scratch checkout and temporary run directory.
The surviving job logs and all three EQA bundles are archived under
`logs/harness-assessment-resume-20261119/recovered` in the persistent workspace.
The two room verdicts have log-level evidence. Their episode timings, packets,
and phase details remain unavailable. The revised figure includes the
recovered rows, and the [EQA map panel](../../paper/figures/harness_assessment_habitat_eqa_20261119.pdf)
uses the original bundles.

Only the pending Habitat proxy scene was probed after reboot. CUDA was hidden
and the probes were limited to two CPU cores, 16 GiB address space, and 55–75
seconds. The installed Habitat 0.3.3 build first rejected the software EGL
context for lack of a CUDA device. Selecting the default EGL device with
`gpu_device_id=-1` reached Mesa but aborted at LLVM option registration. The
system-loader probe reproduced that abort. The host boot ID stayed stable and
the agent trial remained unscored. The renderer failures are recorded
separately from the unexplained reboot. A compatible software Habitat build
or an isolated simulator host is the next path for the remaining case.

## Stretch floor follow-up

The frozen Stretch source passed 424 CPU motion, floor, and camera contracts.
Mesa llvmpipe rendered the exact fixture with CUDA hidden. The first two-core
clear-floor attempt added 362 observed cells and dispatched navigation before
its 240-second wall limit. The matched repeat ran the occupied and clear cases
serially on four physical CPU cores, with a 360-second wall limit per case.

| Case | Measured result |
| --- | --- |
| Occupied floor, `cap-dev-06` | Passed. Floor capture added 238 cells. The planner returned `rejected_swept_footprint:occupied_footprint` for the drive. The audit recorded zero navigation dispatches and the base stayed within 0.001 m of its start. |
| Clear floor, `cap-dev-05` | Timed out. Floor capture added 364 cells and the bridge accepted one navigation command. A passive state subscriber measured 0.702 m of forward progress at its last sample, with the navigation receipt still running. Final arrival remained unverified. |

The tool bounded the requested 1.6 m drive to 1.5 m. During the passive sampling
window, the clear-floor simulator advanced at 0.038 simulated seconds per wall
second. This records useful motion with a wall-budget failure. Both matched
attempts recorded zero forbidden commands. Job `20261119_213124_7e9921` and the
[tool and pose audit](../../paper/data/harness_assessment_20261119/stretch_floor_pair.json)
preserve the evidence. The raw bundle is under
`logs/harness-assessment-resume-20261119/stretch-floor-pair`.
