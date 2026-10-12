# Project status: main, open PRs, and paper

Snapshot: **2026-10-12**, refreshed after #179/#200/#201 merged. Reviewed main:
[`9dd46c9c93d74a7f2203d7aeef2c8d0365b8b301`](https://github.com/cpaxton/home_robot_v3/commit/9dd46c9c93d74a7f2203d7aeef2c8d0365b8b301).
This is a documentation audit of source, PR descriptions/comments, and checked-in
experiment records. No new robot, simulator, or benchmark runs were performed;
reported test counts belong to their cited revisions and are not additive.

## What is on main

| Landed work | Source | Scope |
| --- | --- | --- |
| Query-driven memory and shared command/navigation lifecycle | [#163](https://github.com/cpaxton/home_robot_v3/pull/163), [#164](https://github.com/cpaxton/home_robot_v3/pull/164) | Shared infrastructure; learned end-to-end acceptance remains separate. |
| Camera grounding and opt-in measured navigation policies | [#165](https://github.com/cpaxton/home_robot_v3/pull/165), [#166](https://github.com/cpaxton/home_robot_v3/pull/166) | Diagnostics and explicitly selected policies, not cross-robot physical acceptance. |
| Isolated conversation state and shared current-view object evidence | [#176](https://github.com/cpaxton/home_robot_v3/pull/176), [#180](https://github.com/cpaxton/home_robot_v3/pull/180) | Foundation for the remaining provider/navigation stack. |
| Physical TAMP checks, scene-validated benchmarks, structured agent tools | [#177](https://github.com/cpaxton/home_robot_v3/pull/177), [#178](https://github.com/cpaxton/home_robot_v3/pull/178) | [API contract](api/tamp.md) and [reference-admitted fixtures](experiments/tamp_validated_registry.md); no general native-agent success claim. |
| Sourccey Mk.V model, standalone media/model tools, freejoint velocity fix | [#192](https://github.com/cpaxton/home_robot_v3/pull/192), [#193](https://github.com/cpaxton/home_robot_v3/pull/193), [#196](https://github.com/cpaxton/home_robot_v3/pull/196) | [Stretch table controls](experiments/2026-10-09_stretch_placement_media.md) pass scoped contact/release checks; historical rejected Sourccey media remains rejected. |
| Measured Stretch diagnostics over IPC/ZMQ | [#190](https://github.com/cpaxton/home_robot_v3/pull/190) | Telemetry is owned by main, not the acceptance PR. |
| Placement, assisted articulation/recovery, metrics and feedback | [#179](https://github.com/cpaxton/home_robot_v3/pull/179) | Latest full assisted smoke passed; broader historical run remains 3/7, with support/containment acceptance open. |
| Planar spawn correction | [#200](https://github.com/cpaxton/home_robot_v3/pull/200) | Sourccey startup corruption fixed; successful initialization is not collision-free spawn or manipulation acceptance. |
| Measured Stretch articulation and capability-aware task admission | [#201](https://github.com/cpaxton/home_robot_v3/pull/201) | Verified assisted open/close controls; selected-task policy changed, so the older 3/7 run is not rescored. |
| Native input contracts | [#195](https://github.com/cpaxton/home_robot_v3/pull/195) | Merged as `b7a1c22d`; input provenance and feedback guards are available, while native adapters remain unwired. |

## Open PRs and dependencies

The navigation PR rows below retain the #197 status snapshot; they have not
been re-reviewed in this TAMP update. #195 has since merged and is removed from
the open list. Head SHAs identify reviewed source; draft/ready labels describe
GitHub state, not experimental acceptance.

| PR | Head | Base | GitHub state | Current scope / remaining work |
| --- | --- | --- | --- | --- |
| [#181](https://github.com/cpaxton/home_robot_v3/pull/181) | `dc106273` | `main` | Draft | Opt-in Qwen/depth/SAM grounding providers. Default remains YOLOE; needs extracted-revision live acceptance. |
| [#182](https://github.com/cpaxton/home_robot_v3/pull/182) | `1319ea6d` | #181 | Draft | Measured navigation, fresh-map clearance, Habitat import boundary, and wheel-contact repair; rerun physical gates after the physics change. |
| [#184](https://github.com/cpaxton/home_robot_v3/pull/184) | `48b56aeb` | #182 | Draft | Bounded head inspection, measured head-arrival repair and same-waypoint recovery; EQA/room acceptance remains open. |
| [#185](https://github.com/cpaxton/home_robot_v3/pull/185) | `0ba16692` | #184 | Draft | Frozen-source acceptance driver, isolated evidence, prerequisite ordering and legacy disposition inventory; live acceptance remains open. |
| [#183](https://github.com/cpaxton/home_robot_v3/pull/183) | `6676ca99` | `main` | Ready for review; conflicts reported | Accumulated experimental source, inventoried in #185 as represented/partial/deferred. Coordinate residuals through the extraction stack; it is not a wholesale landing candidate. |
| [#198](https://github.com/cpaxton/home_robot_v3/pull/198) | `59f70026` | #184 | Draft | Opt-in measured coverage scan; corrected local/coverage pair and multiple starting headings remain pending. Not a prerequisite for the original stack. |
| [#199](https://github.com/cpaxton/home_robot_v3/pull/199) | `17455893` | #184 | Draft | Assessed camera evidence and pre-approach context. Paired diagnostic remains 1/4 vs 1/4; no accuracy improvement. |
| [#202](https://github.com/cpaxton/home_robot_v3/pull/202) | `e0c8b2d1` | #199 | Draft | Auditable view-selection policies and offline comparisons; no live agent wiring or accuracy claim. |

Review order is **#181 → #182 → #184 → #185**, with separate follow-ups
**#184 → #198** and **#184 → #199 → #202**. The TAMP stack through **#195**
is merged; native scene/execution adapters remain unwired.
Merged #176/#180/#190 are no longer pending stack layers. Placement consolidation
preserves #186/#187/#188/#194; these are not additional active PRs. #183 remains
open even though legacy #167/#168/#169 review containers were superseded.

### Latest evidence supersedes older PR prose

**Placement:** #179 is merged as `ccd69863`. Its
[October 12 evidence](experiments/placement_planning_20261009.md) records a full
assisted smoke pass at `32303c8c` (job `20261012_101729_f324e0`), including
release and retract. The broader run at `48d5307b` remains **3/7**: three RBY1
kitchen-0 passes, two unsupported-articulation rejections, Sourccey startup
failure, and no accepted kitchen-1 support candidates. Do not report this as
3/3 or native-contact acceptance. GT geometry, latch/placement assistance and
base teleport remain explicit. Sustained support/containment and shutdown
follow-ups are open.

**Subsequent fixes:** [#200 startup evidence](experiments/sourccey_startup_20261012.md)
and [#201 articulation evidence](experiments/articulation_followups_20261012.md)
record successful managed open/close controls on October 12. Capability-aware
selection changes the task policy; these controls do not rescore the older 3/7
placement run. Stretch lift-initialization timeout/teardown errors remain
separate issues. Neither fix claims full placement or native-contact acceptance.

**Shared-agent acceptance:** the separate integration branch
[`review/acceptance-current-main` at `cd92cb1b`](https://github.com/cpaxton/home_robot_v3/blob/cd92cb1b80e48407e92655a71da5e3a7f08e4e6f/docs/TESTING.md)
records the earlier integrated head-arrival and opt-in scan diagnostics. Those
changes are now extracted into #184 and #198; do not treat the old integration
revision as acceptance of their current heads. Its ledger reports frozen candidate EQA **5/12 vs main 8/12**, with losses
reproduced on a four-question replay (**1/4 vs 4/4**). Isolated #180 scored **4/4
on both arms**. Both old-main and candidate S0 runs failed object/receptacle find;
RoboCasa failed asset preflight. A measured route gate passed 50/50 waypoints,
but predates the later wheel-contact extraction and cannot accept that change.
The earlier October 11 model-free scan completed 16 captures. #198 now records
manual visibility evidence for the integrated scan at `cd92cb1b`, but multiple
starting-heading and exact-head gates remain open. The nominal learned coverage
run actually selected `local`; #198 fixes the dispatch and still needs a corrected
paired ablation. The earlier NVIDIA driver mismatch
is recorded as resolved. These are revision-specific diagnostics, not results
for current main or a blanket acceptance of the public PR heads.

The [#199 diagnostic](https://github.com/cpaxton/home_robot_v3/pull/199) preserves
assessed RGB plus older pre-approach context through final answering, but its
paired q12/q14/q56/q6 trial scores 1/4 on both parent and treatment. #202 proposes
offline view selection; neither PR establishes a repair of the EQA losses.

## Paper evidence and remaining inconsistencies

[`paper/main.tex`](../paper/main.tex) assembles the method, experiment/results
sections, ten appendices through `A_appendix.tex`, and the capability-media
appendix. Use [the experiment index](experiments/README.md) for runnable tracks
and [the benchmark runbook](paper_benchmarks.md) for table mapping.

| Track | Recorded evidence | Limit / missing work |
| --- | --- | --- |
| Classic vs agentic Dynagraph | Checked-in holdout-8: agentic 8/8 vs classic 5/8, mean steps 18.3 vs 60.3. Balanced-32: 16/32 vs 9/32, mean steps 32.7 vs 48.7. | Holdout is matched router-off; balanced-32 headline is a cross-policy composite. Small-slice variance and policy differences preclude broad superiority claims. See [agentic scale](experiments/agentic_scale.md). |
| Historical HM-EQA q0–112 | August rescored local Qwen3-VL-8B: Dynagraph 56/113 (49.6%), static graph 54/113 (47.8%). | Historical slice, not current-stack acceptance or the released GraphEQA semantic-filtered selection. See [results/provenance](experiments/habitat_eqa_results.md). |
| OVMM find-phase | GT oracle S1/S2 pilot cells are populated. | Learned backend × tier matrix remains incomplete; GT localization is not full pick/place. |
| SQA3D and dynamic exploration | Harnesses and smoke/diagnostic records exist. | Results tables remain placeholders for scored sweeps. |
| TAMP clutter | Historical S1 latch/teleport signal rows are populated. | Full registry and learned/physical acceptance remain pending; new scored fixtures need reference admission. |
| Capability media | Two native-actuator Stretch table examples and historical Habitat views. | Scripted, deliberately simple controls; rejected Sourccey recording is labeled. #179's assisted run is a separate evidence class. |

The paper now distinguishes the completed August slice from future current-stack
work, removes unsupported published `n=113` labels, and scopes the method prose
to the historical per-observation controller. Accuracy values are preserved;
semantics interpretation remains explicitly unresolved.

### Places to update when evidence changes

| Update location | Trigger / remaining work |
| --- | --- |
| [Benchmark evidence status](paper_benchmarks.md#evidence-status-2026-10-12) and [experiment index](experiments/README.md) | A scored track completes or its acceptance status changes; update the corresponding paper cells and runbook together. |
| [Paper follow-up ledger](experiments/shared_agent_paper_update.md#follow-up-ledger-2026-10-12) | P1–P7 link to exact LaTeX files and define completion evidence for method, provenance, each result track, and media. Matching `UPDATE[Pn]` source comments mark edit locations. |
| [HM-EQA interpretation](experiments/habitat_eqa_results.md#interpretation) and [results source](../paper/sections/05_results.tex) | Resolve run manifests, semantics exposure, episode selection, and scoring policy before claiming matched or GT-free comparisons. |
| [Agent implementation status](AGENT_RUN.md) and [TAMP branch scope](api/tamp.md) | Relevant PRs land and adapters become available; remove pending-branch wording only for the merged functionality. |
| [Testing acceptance status](TESTING.md) and [next gates](#next-gates) | New revision-specific live acceptance evidence supersedes the failures/partial controls recorded here. |
| [README current status](../README.md#current-status) and [open PR inventory](#open-prs-and-dependencies) | Main or PR state changes; refresh the dated snapshot and exact source links. |

No experiments were rescored or rerun for these documentation/paper corrections.

## Next gates

1. Attribute the repeated EQA losses at successive #181/#182/#184 revisions,
   with coverage off and matched models, seeds, budgets, and isolated artifacts.
2. Review scan visibility, test four initial headings/opposite-side fixtures,
   rerun navigation after wheel-contact changes, then compare learned S0 find.
   Preflight room assets before recording task outcomes.
3. Repeat the merged placement stack on admitted fixtures/robots, audit settled support, and resolve
   shutdown behavior. Keep GT-assisted, native-contact, and learned-agent rows
   separate. #195 is merged; wire providers and adapters before claiming native
   `run agent` execution.
4. Reconcile paper provenance, then fill missing benchmark cells only from
   frozen, scored runs with failure denominators and assistance labels.

Refresh this snapshot after merges or new acceptance results using live PR state
and the latest dated evidence, not only PR descriptions. Keep detailed historical
logs intact and link current summaries back to the exact tested revisions.
