# Merged TAMP review corrections: validation

Runtime `5a891c59`, PR [#246](https://github.com/cpaxton/home_robot_v3/pull/246).
The [review ledger](../plans/2026-11-16_tamp_stack_review.md) records all findings
from #226/#231/#234/#242/#243/#244, including suggestions deliberately retained
or rejected on evidence. [Raw artifact hashes and attempts](tamp_stack_review_20261116.json)
are compact; verbose traces remain in `.tamp-runs/stack-review-20261116/`.

| Gate | Result |
| --- | --- |
| Full motion, approach and camera contracts | 502 passed, zero failures/errors/skips |
| Full task, CHAT, VLM and evaluator contracts | 417 passed, zero failures/errors/skips |
| Final affected contracts | 163 and 92 passed; overlap with broader suites |
| Bounded-release live two-goal control | 2/2 transfers, 482 audited commands, zero forbidden |
| Bounded-approach live shifted swap | 3/3 transfers, 787 commands, one completed checked base approach, zero forbidden |
| Actual Caliban CHAT contextual recovery | Correct partial/stop, then exactly one remaining transfer; 8 model calls, 514 audited commands, zero forbidden/shutdown warnings |
| Separate observed-branch integration | 390 affected checks pass; observed transfer remains unqualified |

No totals combine overlapping suites or separate trial populations. Native
controls used a working tree later committed as `5a891c59`; each corrected
control's 17 frozen source hashes matches the final runtime. The actual-agent
trial used that clean commit and independently measured both original goals.
Inference used remote Caliban (`Qwen/Qwen3-VL-8B-Instruct`); simulation had no
cameras or local GPU. No hardware was used.

The partial receipt names the completed cylinder placement and unfinished cube,
reports second-step `preflight`, overall dispatch true but failed-step dispatch
false, consumes its handle and stops. Only after the separate user request
“Please retry the remaining work” does the agent call `scene_tasks` → `plan_task`
→ `execute_task_plan`, using a new one-step handle. It moves only the remaining
cube, verifies both final goals and exits cleanly. This is an evaluator-injected
admission refusal, not actual solver exhaustion, contact loss or unknown-outcome
recovery. It is one familiar development case, not a reliability estimate.

Retain earlier failures. The first contract run caught an array truthiness error,
camera sensor/render metadata disagreement, and removal of a recovery-ledger
check that is actually needed because its callback mutates the ledger. Later
fixtures needed a collision-free retreat and explicit radian convention for
their independent placement oracle. The first expanded task run passed 416/417;
the remaining fixture expected the pre-change `ReleaseLimits` serialization.
Its corrected full rerun passes 417/417.

Both first live controls failed at release. Measured anchoring alone omitted
the 18.7 mm command/FK shortfall from candidate construction, leaving too little
opening clearance. Adding the same measured reserve used by the lowered probe
restores search coverage inside the original 40 mm global raise bound. Actual
sampled FK is checked against that bound. Support penetration, drop, retention
and final-goal thresholds are unchanged. Strict 1 mm support contact and unknown
space remain rejection conditions; there is no support exemption.

Review agents inspected the code read-only and are not human approvers. Ruff
and whitespace checks pass. Historical cohorts are not rescored; observed
grounding/exploration, hardware, continuous collision and dynamics guarantees
remain outside these development controls.

Disk maintenance pruned regenerable package caches and removed only a verified
duplicate CUDA installer, preserving its identical original. Model weights,
installed environments and experiment evidence were retained. Free space was
observed at 1.7 GB before and about 14–15 GB after cleanup; concurrent filesystem
activity prevents exact attribution of that difference to these removals.
