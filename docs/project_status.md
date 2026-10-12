# Project status: main, open PRs, and paper

Snapshot: **2026-10-12**. Reviewed main:
[`7f65dbeff6ef81fc824da25497b420b57dc1cc16`](https://github.com/cpaxton/home_robot_v3/commit/7f65dbeff6ef81fc824da25497b420b57dc1cc16).
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

## Open PRs and dependencies

All seven open PRs at the snapshot are listed below. Head SHAs identify reviewed
source; draft/ready labels describe GitHub state, not experimental acceptance.

| PR | Head | Base | GitHub state | Current scope / remaining work |
| --- | --- | --- | --- | --- |
| [#181](https://github.com/cpaxton/home_robot_v3/pull/181) | `dc106273` | `main` | Draft | Opt-in Qwen/depth/SAM grounding providers. Default remains YOLOE; needs extracted-revision live acceptance. |
| [#182](https://github.com/cpaxton/home_robot_v3/pull/182) | `1319ea6d` | #181 | Draft | Measured navigation, fresh-map clearance, Habitat import boundary, and wheel-contact repair; rerun physical gates after the physics change. |
| [#184](https://github.com/cpaxton/home_robot_v3/pull/184) | `b5d04a01` | #182 | Draft | Bounded head inspection and same-waypoint floor recovery; EQA/room acceptance remains open. |
| [#185](https://github.com/cpaxton/home_robot_v3/pull/185) | `76ce7f8c` | #184 | Draft | Frozen-source acceptance driver and legacy disposition inventory. Reconciled stack reports 165 focused passes, without live task acceptance. |
| [#179](https://github.com/cpaxton/home_robot_v3/pull/179) | `48d5307b` | `main` | Ready for review | Consolidated placement, assisted articulation/recovery, metrics and feedback. One assisted full-task pass; repeatability, settled support, and shutdown follow-up remain. |
| [#195](https://github.com/cpaxton/home_robot_v3/pull/195) | `ceaec939` | #179 | Ready for review | Observed/privileged input boundaries and immutable measured-joint evidence; native scene/execution adapters and observed geometry providers remain unwired. |
| [#183](https://github.com/cpaxton/home_robot_v3/pull/183) | `6676ca99` | `main` | Ready for review; conflicts reported | Accumulated experimental source, inventoried in #185 as represented/partial/deferred. Coordinate residuals through the extraction stack; it is not a wholesale landing candidate. |

Review order is **#181 → #182 → #184 → #185** and **#179 → #195**.
Merged #176/#180/#190 are no longer pending stack layers. Placement consolidation
preserves #186/#187/#188/#194; these are not additional active PRs. #183 remains
open even though legacy #167/#168/#169 review containers were superseded.

### Latest evidence supersedes older PR prose

**Placement:** #179's description still says the live task times out and should
remain draft. Its October 11 follow-up and
[experiment log at the reviewed head](https://github.com/cpaxton/home_robot_v3/blob/48d5307be2269f7efe04a9a6ef3df997afe84721/docs/experiments/placement_planning_20261009.md#october-11-rigid-attachment-and-coherent-payload-geometry)
report GPU job `20261011_133133_a84009`, source `10076be1`, completing RBY1
bowl → dishwasher through public CHAT tools: pickup, transport, placement,
verification, and retract. It recorded 575 successful acknowledgements without
execution timeouts. The final `48d5307b` adds error-path hardening after that run.
The run used GT geometry, object latch/placement assistance, assisted opening,
and base teleport. Its final scorer checks proximity, not sustained support.
Shutdown then rejected `cancel_navigation` with “cancellation target has no
confirmed stop”. One assisted pass does not establish repeatability or native
contact success. The draft wording in #179/#195 is stale relative to GitHub state.

**Shared-agent acceptance:** the separate integration branch
[`review/acceptance-current-main` at `cd92cb1b`](https://github.com/cpaxton/home_robot_v3/blob/cd92cb1b80e48407e92655a71da5e3a7f08e4e6f/docs/TESTING.md)
contains additional head-arrival and opt-in coverage-scan work beyond the public
stack. Its ledger reports frozen candidate EQA **5/12 vs main 8/12**, with losses
reproduced on a four-question replay (**1/4 vs 4/4**). Isolated #180 scored **4/4
on both arms**. Both old-main and candidate S0 runs failed object/receptacle find;
RoboCasa failed asset preflight. A measured route gate passed 50/50 waypoints,
but predates the later wheel-contact extraction and cannot accept that change.
The October 11 model-free scan completed 16 captures; manual visibility review
and four-heading coverage gates remain open. The earlier NVIDIA driver mismatch
is recorded as resolved. These are revision-specific diagnostics, not results
for current main or a blanket acceptance of the public PR heads.

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
3. Repeat #179 on admitted fixtures/robots, audit settled support, and resolve
   shutdown behavior. Keep GT-assisted, native-contact, and learned-agent rows
   separate. Restack/review #195 against the accepted parent; wire providers and
   adapters before claiming native `run agent` execution.
4. Reconcile paper provenance, then fill missing benchmark cells only from
   frozen, scored runs with failure denominators and assistance labels.

Refresh this snapshot after merges or new acceptance results using live PR state
and the latest dated evidence, not only PR descriptions. Keep detailed historical
logs intact and link current summaries back to the exact tested revisions.
