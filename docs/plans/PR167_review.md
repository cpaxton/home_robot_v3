# PR #167: review and landing map

The goal is one learned harness for EQA, OVMM and multistep manipulation, not
an independently tuned policy per benchmark. Keep PR #167 in review/draft until
the [bounded acceptance gates](../experiments/manipulation_acceptance.md) pass.
Current evidence and failed trials live in the
[shared grounding report](../experiments/shared_grounding_pilot.md).

Current evidence: Stage B passed **6/6** physical pick/place cases on frozen
`ef533ed3` with tracked-narrow and explicit NoSlip=10 scenes. Room OVMM, learned
TAMP and paired EQA/find gates still block promotion. Subsequent private sequence
scoring, open-sink fixture and timestamp transport are separately tested changes,
not part of the frozen panel. Keep production contact defaults unchanged.

The latest code candidate is `787acb6f`: the repaired separated-tabletop control
passes 2/2, and `20260914_214506_a2a467` runs the remaining original/mirrored
four cases serially. Do not claim the old panel covers this source. The failed
sink remains documented. General end-effector clearance planning is deferred;
this PR still needs the unchanged bounded gates on predeclared accessible
open-support room tasks, learned two-step TAMP and paired EQA/find.

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
