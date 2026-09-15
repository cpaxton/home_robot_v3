# PR #167: review and landing map

The goal is one learned harness for EQA, OVMM and multistep manipulation, not
an independently tuned policy per benchmark. Keep PR #167 in review/draft until
the [bounded acceptance gates](../experiments/manipulation_acceptance.md) pass.
Current evidence and failed trials live in the
[shared grounding report](../experiments/shared_grounding_pilot.md).

Late September 14 follow-up: RoboCasa inertia fix is isolated in
[dependency PR #1](https://github.com/cpaxton/robocasa/pull/1). The first native
room case loses its payload; `1d257800` now detects missing post-lift evidence
and stops placement, while its original-tabletop control passes T/T. Molmo
stops at its first turn; `65574600` separately fixes contact-margin evidence
recording. Broad tests: 589 passed / 4 skipped. A fixed-turn diagnostic supports
correcting wheel contact-profile mixing, but that production change and exact
room retry are pending. These fixes do not establish room acceptance or replace
the earlier frozen six-case panel.

Current evidence: Stage B passes **6/6** physical pick/place cases on frozen
`4f78ae62` with tracked-narrow and explicit NoSlip=10 scenes, all final views
inspected. This separately repeats the older `ef533ed3` success after fixing the
failed `787acb6f` mirrored placement; no earlier runs are pooled. Keep production
contact defaults unchanged. The later launch-only Molmo interpreter-discovery
fix `b5fd55ef` passes its symlink and real-environment checks; broad tests now
pass 570 / 4 skip, and focused Molmo config/CLI checks pass 35 / 1 skip.
Both Molmo geometry archives compile and render in `20260914_224024_b8970e`;
the exterior views are ceiling-occluded, so interior task/start inspection is
still pending. This is launch preflight, not a room-policy success.

Room OVMM, learned TAMP and paired EQA/find still block promotion. The failed
sink remains documented; general end-effector clearance planning is deferred.
New RoboCasa fixtures additionally require a dependency review: the installed
fork `3d0bd42` rewrites unspecified mesh inertia to shell before native model
creation. EMET preserves those masses, but that equality does not establish
authored physics. Do not admit these fixtures by tuning density or hiding the
dependency issue. Keep the unchanged gates on predeclared accessible tasks.

## Review in dependency order

| Review unit | Main code | Required evidence |
| --- | --- | --- |
| Command and motion contracts | `core/command_*`, lightweight `emet_core` copies, ZMQ client, native simulator, velocity control, IK | Command identity/writeback tests; wheel gearing, braking, grasp frame and RGB-D registration checks; physical carry controls; measured route completion |
| Scene conversion and reproducibility | RoboCasa generation and XML adaptation | Source-to-adapted mass/COM/inertia equality (including massless markers), frozen expanded XML, task-identity and actual-start checks; corrected dynamics reported separately |
| Grounded visual evidence | `memory/{query_grounding,vlm_region_grounding,surface_candidates,grounded_target}`, SAM2 proposal support | Cached positive/absent/wrong-surface controls; raw VLM responses and masks; no spatial promotion from unlocalized labels |
| Shared task loop | `agent/{loop,prompt,tools}`, query manipulation task, grasp/place and navigation operations | Observation→action→action tests, stop on tool failure, bounded recovery, fresh post-motion views; independently scored learned manipulation |
| Evaluation and diagnostics | `eval/manipulation_trace.py`, dataset/ablation tools, pilot driver and replay renderer | Private GT only; physical versus process/tool results separated; failed and unrun cases retained; frozen settings and artifact hashes |
| Configs, docs and paper handoff | Named `configs/emet/query_*_pilot.yaml` rows, environment guides, experiment reports | Resolved-config tests; explicit control/candidate differences; claims limited to demonstrated tasks and embodiments |

These are review units, not guaranteed cherry-pick boundaries: the development
history touches the same files repeatedly. If splitting into stacked PRs,
extract and test each dependency layer on a fresh branch without rewriting the
running evaluation checkout. Do not blindly cherry-pick a list of commit names
or squash away failed-trial provenance. Push review branches only, never main.

## Config and scope checks

- Retain detector-only, Qwen-box and recovery-only controls. The near-support
  release row is experimental and must not silently replace those settings.
- Treat calibrated robot-frame grasp offsets, native wheel limits and joint
  profiles as adapter parameters, not universal object-localization corrections.
- Do not claim that measured surface support is full object geometry or that
  near-support placement is contact/force verification.
- Keep oracle scene enumeration, attachment and teleport controls out of
  learned query-mode acceptance. Door/drawer articulation remains unsupported;
  use open receptacles and say so.
- Leave experimental onboard perception/streaming and deferred hardware work
  out of this promotion. Habitat long-range OVMM remains a documented follow-up.

## Landing sequence

1. Freeze one candidate and complete the six-case basic physical gate. Earlier
   development passes and different-source panels are not pooled.
2. Freeze actual room task instructions, simulator bodies/supports, seeds and
   starts before the bounded RoboCasa/Molmo OVMM and learned TAMP runs. The
   RoboCasa adapter must preserve source body dynamics; earlier shell-inertia
   conversion produced overweight objects and invalidated payload assumptions.
   A rendering-only check does not establish a physically equivalent scene. The
   current tabletop driver and oracle TAMP script are **not** those room gates.
   Ordered multistep physical scoring must be connected before claiming TAMP.
3. Run the paired EQA/find regression rows with model and budgets held fixed.
   Investigate discordant cases; a three-question smoke is not equivalence.
4. Update the paper from the scored results: include top-down/trajectory views,
   agent RGB/masks and labeled simulator reconstructions, including failures.
   Separate complete-system comparisons from ingestion/fusion ablations.
5. Review the units above, preserve the frozen controls and only then decide
   promotion/merge. Passing unit tests or a single tabletop task is insufficient.
