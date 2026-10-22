# Shared-agent stack readiness

## Restacked review closeout — 2026-10-22 (current)

All eleven initially open PRs received an exact-head focused test attempt, and
all received review/inline-comment inspection. Failures were retained, repaired
on owners where appropriate, and rerun. The new independent #218 is also tested.
No main merge, default-model/policy promotion or heavy parallel experiment ran.
User's dirty main checkout is untouched. Only one extra ~700 MB audit checkout
is reused, with a 12 GiB free-space floor; no eleven-worktree disk expansion.

### Landing/review status

| PR | Current head | Evidence | Status |
| --- | --- | --- | --- |
| #218 | 754c0a81 | 39 parity/command/tracker/arrival tests | New ready main-based repair; land first after review. |
| #181 | 604227a3 | 91 provider/parity tests; provider source comparison against successful S0 component path | Ready; base #218, opt-in providers, no task-superiority claim. |
| #182 | 3da30084 | 120 full owner-suite tests; 77 navigation/TAMP conflict-resolution checks, 24 sweep/mode checks | Ready; base #181. |
| #184 | efed2fc5 | 308 owner-suite tests; 44 scan/recovery/base-pose checks | Ready; base #182. |
| #185 | c29891b8 before this report-only update | 100 tooling/parity tests on refreshed parent; runtime fixes inherited | Already ready; tooling not task-performance acceptance. |
| #199 | b2b51d2c | 646 broad regression tests | Ready; base #184. |
| #202 | f7b73bae | 188 selector/history/parity tests | Ready; base #199, ranking defaults remain opt-in. |
| #216 | 665215b4 | 422 authority/inherited/TAMP/parity tests | Ready; base #202, preserve final model authority. |
| #217 | 1a3d65c5 | 124 dialogue/transport/recovery/parity tests; successful S0 repeat evidence | Ready; base #216, not robust long-horizon acceptance. |
| #198 | 0b0ed8e6 | 32 scan/parity tests before ownership propagation; owner scan fixes separately verified | Keep draft/optional, not a blocker for the critical stack. |
| #213 | 57bc97b0 | 244 TAMP/observed/timing/command/parity tests after refresh | Keep draft for new motion/pregrasp review; base #218. |
| #214 | 0795f470 | 97 pass; contact_retreat recovery fails, repeated in isolation | Independent merge blocker; not a dependency of this stack. |

Counts are overlapping suites, not unique tests or independent task trials.
For #199/#202/#216/#217, the last ownership-only merges preserve both src/emet
and src/emet_core tree hashes exactly relative to final tested heads
85431dfd/a216222c/210b3bb4/e0d915b5 respectively. This is code-equivalence
evidence, not a fictional rerun on every documentation/merge SHA.

### What was repaired/reorganized

- Current main itself had four stale emet_core runtime copies. #218 uses the
  existing sync script to copy canonical command_client/runtime/tracker and
  navigation_result without changing canonical policy. #181/#213 are now
  children of #218; the rest of the chain is refreshed on current main.
- SAM2 PIL normalization and its regression moved from #199 to owner #181.
  The new PIL case fails before the two-line repair and passes after.
- Sweep failure reset/removal of duplicate validation moved to owner #182.
- Full-circle scan spacing and rejecting False/None turn receipts moved to #184.
  Four captures now cover four headings rather than a hard-coded 45-degree arc;
  table-specific scan behavior and safety checks are preserved.
- Fixed an outdated telemetry fixture (actual actuator_names contract) and a
  voxel test that incorrectly borrowed evidence from a same-numbered graph ID.
  The latter now requires an explicit captured-frame binding and tests rejection
  without it. No evidence threshold was weakened; no excluded failure hidden.
- Resolved real refresh conflicts preserving main's articulation-access metadata,
  shared check_base_poses helper, articulation model, measured navigation progress
  and bounded head capabilities. #213 also inherits main #215 acquisition timing.

#214's earlier review concerns about core parity and stop/release handling are
already addressed in its current code. Its original collection errors came from
missing stable_baselines3 in the shared venv, not code failure. Installed only
SB3 2.9.0 --no-deps into /tmp/emet-rl-audit-deps-20261021; shared environment
unchanged. With that dependency the actual seeded contact-retreat teacher
recovery fails at seed 71000 / 15 simulated seconds (~0.409 m approach error).
It repeats in isolation; no threshold, horizon or physical condition was changed
to make it pass. Owner PR has the exact failure comment. Keep it independent.

### Evidence and next gate

Manifest/log/JUnit roots: /tmp/emet-stack-audit-20261021; original,
repair-, propagated-, final-, and owner-final- runs retained separately.
Jobs: 20261021_235143_06ae20, 20261021_235849_d10b29,
20261022_000329_c1e317, 20261022_001148_ec7cf8,
20261022_002003_518a6a. Exact frozen inputs are named in invocation JSONs.
Readiness is for scoped review, not automatic main merge or a performance claim.

Next freeze a combined #217 + #213 sandbox, excluding #214 and default-off #198.
First test shared dispatch/state/grounding/freshness contracts. Then serially run
paired EQA and small observed OVMM plus a privileged native two-goal TAMP control
and a failure-stop probe. Keep privileged feasibility separate from observed
agent performance; find-only S0 success is not pick/place success. Native TAMP's
new motion delta still needs review even with green unit tests. Do not describe
its design-only household registry as an executable completed benchmark.

TODO.md on #217 now tracks real sim-clock/wall-clock phase reporting and a later
mjlab/mjenv-style parallel-backend parity study. Neither is implemented by this
test closeout; keep heavy workloads serial and avoid simulator migration detours.
Paper-scale promotion still requires inspected end-to-end outcomes, exact
configs/SHAs, wall + actual simulator time, image/command provenance, task-level
failure breakdowns and figure-ready evidence. No full sweep requested.

### Combined next-stage sandbox

Pushed `test/shared-agent-e2e-20261022` at b094bcfd: #217 1a3d65c5 plus #213
57bc97b0, excluding optional scan #198 and RL #214. Reuses the audit-owned
checkout; per-PR source SHAs are preserved in invocation JSON, not its current
working-tree HEAD. No extra PR and no main merge.

The merge exposed real integration conflicts. Sandbox resolution retains both
camera acquisition image_timing/map-scope metadata and TAMP received_monotonic/
depth provenance arguments, rather than dropping one interface. SAM2 retains
explicit device selection, PIL normalization and one shared model. These are
testing-only merge resolutions, not silently pushed back into owner PRs.

Serial one-core shared-contract check `20261022_002629_479092` is running;
JUnit `/tmp/emet-combined-agent-contracts-20261022.xml`. Gate covers discovery,
TAMP dispatch/session/physical contracts, EQA dialogue/transport/recovery,
captured views/history, grounding, namespace and acquisition timing. No live
end-to-end task result is implied by starting this unit gate.

## Exact-head closeout — 2026-10-21 (historical)

User priority: test every open PR, resolve comments, and converge on independently
mergeable slices before end-to-end complex-task assessment. Main remains 801dce82.
Do not merge an experimental umbrella branch or silently promote a new policy.

Serial exact-head regression job `20261021_235143_06ae20` covers all eleven open
PRs: #181, #182, #184, #185, #198, #199, #202, #213, #214, #216, #217.
Manifest `/tmp/emet-stack-audit-manifest-20261021.json`; results/logs/JUnit under
`/tmp/emet-stack-audit-20261021`. Changed test files plus explicit parity/transport
regressions run one core at a time. One reused audit checkout, not eleven copies;
stop below 12 GiB free. Test-file success is not full-suite or live-task acceptance.
Recheck remote SHA drift before marking any PR ready. Documentation-only follow-ups
may move heads; record those explicitly rather than attributing new runtime tests.

### Review/stack disposition

All scoped stack PRs' top-level comments/reviews and inline-comment APIs inspected;
no inline comments exist on #181/#182/#184/#185/#199/#202/#216/#217. Older review
findings have owner commits, but readiness still requires the following checks.

| Slice | Remaining closeout / action |
| --- | --- |
| #181 providers | Focused exact-head run: 89 pass. Verify opt-in provider admission/live evidence, then clear independently; no default promotion or accuracy claim. |
| #182 navigation | Audit found real robot/runtime parity drift: navigation policies/progress and base-pose query differ across distributions. Sync on owner branch and rerun parity + command/arrival/navigation tests before propagation. |
| #184 recovery | Focused exact-head run passed. Inherit tested #182 parity repair; preserve two-observation bound and structured head failures. Live successful recovery evidence exists in descendant S0; do not call it an exact-parent task comparison. |
| #185 acceptance | Focused exact-head run passed; already non-draft. Land tooling independently of task-score improvements after repaired parent. Missing/error artifacts remain hard infrastructure failures. |
| #199 evidence | Audit broad changed-file regressions; merged-main noise means some tests are inherited. q14 final packet retains fan as historical slots 3/4: no new pinning rule justified. Clear capture/handoff contracts, not semantic accuracy superiority. |
| #202 selector | Verify scope/current-capture/pin-budget regressions and opt-in defaults. Separate ranking policies/experiments from mandatory timestamp/evidence correctness if policy performance blocks review. |
| #216 authority | Keep narrow final-model semantic authority contract independent of accuracy gains; preserve abstentions and physical admission gates. Paired 9/12 vs 9/12 historical pilot is not current-main non-inferiority proof. |
| #217 dialogue | Public-facade recovery mismatch fixed a0a168a9; S0 and timing-only repeat both ground both targets. EQA4 2/4 clean; Molmo bowl only. Context budgeting remains TODO, not silent eviction. Documentation-only later heads retain tested runtime. |
| #198 optional scan | Keep optional, not on critical landing chain. Test measured scan/receipt contract independently; do not require a default scan promotion for other repairs. |
| #213 TAMP orchestration | Read review + merge-scope document; discovery-only/failure batch boundaries remain mandatory. Latest household protocol is explicitly design_only with no registered episodes: not executable benchmark evidence. Test current head and distinguish privileged control from observed discovery. |
| #214 reactive RL | Independent main child, not a dependency for shared-harness merge. Review explicitly requests runtime parity and gripper-stop semantics; test these, plus deployment/loopback safety. No learned-controller promotion needed for this stack. |

### Completed first-pass audit and owner repairs

| PR | Frozen first-pass result | Follow-up |
| --- | --- | --- |
| #181 | 89 pass | Scoped provider live gate still distinct. |
| #182 | 118 pass, 1 parity failure | cd062c49 syncs runtime copies; full 119-test retest passes. |
| #184 | 302 pass | Inherited tested parity fix at 80838224; rerun new head. |
| #185 | 99 pass | Parent propagation and ledger refresh pending. |
| #199 | 571 pass, 1 outdated telemetry fixture failure | ff470223 adds actual actuator-name fixture field/assertion; all 572 pass. |
| #202 | 187 pass | Parent repairs propagated at a9298eee; rerun new head. |
| #216 | 421 pass | Parent repairs propagated at 30b79000; rerun new head. |
| #217 | 122 pass, 1 obsolete graph/voxel-ID test assumption | 3f03175e binds a real captured frame and asserts unbound graph IDs cannot borrow voxel evidence; all 123 pass. Inherited parent repairs at ca3e6d38; rerun new head. |
| #198 | 31 pass | Optional scan remains off the critical landing path. |
| #213 | 199 pass | Current scope adds motion/pregrasp changes beyond original orchestration review; those need review, not just old approval. |
| #214 | Collection blocked: missing declared stable_baselines3 | Installed SB3 2.9.0 --no-deps into isolated /tmp test target, not shared venv. Retest running; current code already synchronizes core copies and revokes/freezes release on stop. |

Repair job `20261021_235849_d10b29`; original and repair logs/JUnit are retained
separately. No threshold/semantic safety change was made to make tests pass.
The formerly excluded voxel fixture now exercises the captured-view namespace
contract instead of incorrectly treating a graph ID as a voxel frame. It has
been moved to the evidence owner #199 as 75187c76 and propagated upward, not
left as dialogue-only test churn. Original failures remain visible in artifacts.

### Reorganize only where it removes a real blocker

Keep parent chain #181 -> #182 -> #184. #185 and #199 are separate children;
#199 -> #202 -> #216 -> #217. Propagate tested runtime fixes bottom-up, without
rebasing active frozen experiments or merging #185's report/replay history into
runtime PRs. If selector policy evidence remains contested, split mandatory
scope/timestamp/pixel-handoff repairs from optional ranking policies, then rebase
authority/dialogue onto the correctness slice. Do not discard tested fixes merely
because a policy experiment has no accuracy gain. #198/#213/#214 stay independent
workstreams with explicit gates. No unnecessary new mega-PR.

### Runtime reporting and next end-to-end gate

S0 profiling succeeded again (509 s wall in that repeat): all nine base commands
succeeded, base-call total 440 s including initial scan, endpoint sampling 0.26 s,
A* 0.06 s; simulation runs ~0.15–0.18x realtime. Nested timings overlap and must
not be added. Do not equate wall duration with algorithm compute or estimate
elapsed sim time from a sampled realtime factor. Add actual per-phase sim-clock
start/end plus boot/domain identity, reset/missing-clock handling and wall time
to artifacts. TODO tracks this and a later mjlab/mjenv-style parallel-backend
parity/throughput study; no migration and no parallel heavy runs now.

After regression and review gates, freeze one combined runtime/config candidate
and use the same model/tool interfaces, budgets, physics and paired seeds:

1. EQA12 development check versus frozen main, preserving final raw answer and
   provenance. Inspect changed cases; small sample cannot certify superiority.
2. Small OVMM progression: visible S0 find -> physical pick/place -> one-room
   Molmo/RoboCasa search/pick/place. Score current-view grounding, measured motion,
   grasp retention and destination placement separately. Find-only passes are
   not manipulation passes; microwave recognition alone is not geometry.
3. Minimal native TAMP sequence with at least two dependent manipulation actions,
   plus one partial/failure-stop case. Privileged world-state control verifies
   feasibility/tooling; observed-agent mode is a distinct gate, not a GT-free row.
4. Use an actually registered compound household case once available; the #213
   design-only registry must be implemented/admitted before quoting those rows.
   Do not launch the proposed full 144-episode condition sweep for this pilot.

Retain exact configs/SHAs/seeds, task-level outcomes, wall/sim time, per-tool errors,
trajectory/command receipts, model requests, image provenance and figure-ready
key views/top-down maps. Paper must report partial capabilities and failure modes;
no claim yet of robust long-horizon manipulation or cross-task superiority.
Habitat OVMM and hardware remain explicit follow-ups, not stealth prerequisites.

## Closeout audit — 2026-10-16 (historical)

The [October 17 context-payload audit](context_payload_audit_20261017.md) records
the completed pilot, inspected image examples, history/freshness gaps and the
controlled offline + trace-only live diagnostics. It supersedes any suggestion
that final-model authority alone cleared task performance.

### Requested main-integrated pilot

Job `20261016_132245_d318eb`, source `0873430d` on
`test/final-model-pilots-20261016`: 12 EQA questions (seed 0), then serial S0
Stretch, RoboCasa Stretch and MolmoSpaces rby1 find checks. Fetched main remains
`801dce82`; merged into #216 at `52464a8e`, preserving main's shared TAMP
base-pose query helper. Integrated authority/selector/driver/TAMP suite: 262 pass.

EQA uses `configs/emet/pilot_model_final.yaml`: model_final, pose selection,
spatial labels, maximum four images, no experimental approach-history module.
Room driver explicitly writes model_final in config and manifest (4 driver
tests pass); VLM/depth-candidate/SAM grounding, physical navigation, 30 s step
timeout, four mapping headings. Local Qwen3-VL-8B int4 throughout. This is one
candidate, not a paired accuracy result, complete manipulation or a full sweep.

Artifacts: `/tmp/emet-final-model-pilot-results-20261016`; job log:
`/home/cpaxton/runs/emet/jobs_runs/final-model-eqa12-room3/job.log`.
At scheduling, exclusive GPU lock acquired and admission stability check pending.
No new score yet. Keep source frozen and inspect actual images, measured motion,
answer provenance and admitted geometry before interpreting aggregate metrics.

Main is `801dce82`. Do not merge the old monolithic experiment into main.
Land the owned slices bottom-up, with defaults unchanged. No full sweep or
hardware deployment was run for this closeout. There are no GitHub inline
review threads to resolve; review dispositions are recorded in PR comments.

### Evidence and limits

The completed serial same-context answer-authority pilot
`20261015_230408_fca6ac`, frozen source `9093b0c9`, scored **9/12 legacy and
9/12 model_final** (24/24 completed). New-policy final answers survived scoring
unchanged in all 12 cases. q14 improves; q65 loses a legacy voting win. Main was
not rerun in that comparison. This is development-set parity, not superiority
or a statistical non-inferiority result. Some correct answers still contain
unsound absence reasoning (q15).

The subsequent exact packet/label repair was NOT included in that live pilot.
Its integrated regression suite passes 97 tests. The clean owner extraction
passes 80 evidence/selector tests; the narrow authority extraction passes 174
answer/config/graph/selector tests. Extraction changes source and must receive
a small exact-tip live handoff check before inheriting any acceptance claim.

OVMM is **not cleared**: the configured-budget S0 diagnostic localizes the
destination with grounded object geometry but misses the pickup object.
The corrected-resource RoboCasa/Molmo find pair did not localize the targets.
These are find diagnostics, not completed pick-and-place trials. The 30 s S0
timeout repair changes its effective budget and is not an equal-budget gain.

### Review disposition and landing order

| Slice | Resolved findings / evidence | Remaining gate or disposition |
| --- | --- | --- |
| #181 | `da1a1a39`: validate tracking mask/region configuration before model setup; remove unused `select_surface`. 25 focused tests pass. No provider/default flip. | Small live provider gate; earlier exact-head EQA attribution was 4/4, not proof of provider success. |
| #182 | `e9cdcaee`: missing-score diagnostics and off-map view fingerprints fixed; 10 focused tests. Earlier mode propagation, real-room wheel-contact fixture and measured timeout distance repairs retained. | Short measured Molmo drive gate. Whole-route rejection and no diagonal corner cutting are intentional safety contracts, not partial-motion promises. |
| #184 | `f0e66a4f`: bridge/sensor errors return failed head observations; motor-actuator abstention and sim tilt limits documented; latest #182 inherited. 250 focused tests passed before parent propagation. Earlier two-capture limit/shared status repair retained. | Exact-tip live recovery check. Motor control ranges are not angular limits; unsupported adapters must supply calibrated capabilities. |
| #185 | Provenance, missing/invalid artifacts, JSON/JSONL, shell exit handling, config timeout precedence and phase errors repaired. 70 acceptance/script/queue/OVMM metrics tests pass after updated parent propagation. Already ready for tooling review. | Tooling readiness does not clear task performance. Do not count infrastructure failures as task failures or passes. |
| #199 | `f4d4c94f`: best unverified packet retained without claiming verification; partial/invalid depth trace explicit; missing scan receipt rejected; reviewed parents inherited. 73 focused tests. | Exact-tip assessed-to-final image check. Semantic answer authority is a separate narrow repair, not hidden in this PR. |
| #202 | `8a590cb7`: prior map-epoch/current-capture/pin-budget repairs retained; optional sequence/pose labels now survive final handoff. 80 focused tests. | Keep opt-in; run the same exact-tip handoff check. No claim that one selection policy is universally best. |
| #216 | Narrow final-model authority extraction on #202, `8e350eb6`; 174 focused tests plus one saved-response audit test. Profile changes only answer authority relative to default. | Draft pending exact-tip live check. Does not import experimental approach-history or sweep code; do not assign the frozen integration's score to this head. |
| #198 | Measured coverage scan remains independent and opt-in. | Defer from this landing sequence; corrected local/coverage ablation and heading fixtures still pending. No default promotion. |
| #183 | Historical monolithic experiment, superseded by the owned slices; closed without deleting its branch. | Archived rather than merged. Experimental residue is not implicitly approved or required for this harness. |

Low-priority review nits are explicitly deferred rather than expanded into new
behavior: unused translation-only controller scaffolding, physical-map `to_xy`
sentinel validation, the graph selector's wasted work before explicit evidence
replacement, and unrelated formatting noise. None is evidence of a task pass.
The physical off-map planning path already rejects invalid cells; do not call
its inverse conversion with the rejection sentinel. Future cleanup must keep
that fail-closed contract and receive focused tests.

### Current shared flow (shipping default versus candidate)

The shared loop is `AgenticEQAExecutor`; OVMM FindObj/FindRec are questions into
that loop, not a second exploration agent. RGB-D, calibration and measured pose
feed voxel memory. Detector/voxel matches are search hypotheses, not verified
objects. Query-driven/lazy grounding promotes an object only after current-view
mask/depth/identity admission. Legacy memory modes remain separate baselines.

1. Qwen receives the task and available memory/navigation tools. It chooses a
   candidate investigation or exploration action within explicit budgets.
2. Navigation validates known-floor/swept-footprint clearance and measures the
   outcome. A command acknowledgement is not arrival. Unknown floor can trigger
   at most two automatic stationary observations and replanning, never an
   assumption that the area is free. Head aiming respects capabilities and
   requires fresh imagery; failures are exposed to the agent.
3. The reached camera view is captured and assessed. Optional history selection
   pins current/pre-turn evidence when valid, rejects cross-map/stale frames,
   deduplicates pixels, and bounds context. Labels distinguish historical camera
   poses from object geometry. The exact assessed packet is retained for final
   reasoning; it is not silently replaced by a nearby graph image.
4. **EQA:** a final Qwen call reasons over the packet and task context. Under
   opt-in `model_final`, its semantic answer/confidence/abstention survives
   submission and Habitat scoring. Earlier assessments guide evidence gathering,
   not answer replacement. **Default/legacy still retains old overrides** until
   an explicit promotion decision; do not describe model_final as merged main.
5. **OVMM:** visual recognition is not manipulation authorization. In
   query-driven mode the same loop requires grounded object evidence; VLM/depth/
   SAM or the configured detector provider supplies object-specific geometry.
   Grounded object IDs take precedence over camera/viewpoint anchors. Fresh
   target geometry is required at manipulation boundaries; navigation, pick,
   place and merged TAMP tools retain their own physical validity checks.

YOLOE may still cheaply propose candidates or serve the configured legacy mask
backend. It is not a requirement of the opt-in VLM/SAM final grounding provider.
Provider, memory policy and selector are explicit configuration axes; do not
change several at once and call the result an isolated ablation.

### Before scaling

First run one exact-tip packet/authority diagnostic and the small serial S0,
RoboCasa Stretch and Molmo rby1 find/navigation checks. Require actual pose
progress, current RGB evidence and admitted geometry, not merely clean exits.
Then run small complete pick/place and the minimal merged TAMP regression gate.
Only after those results are inspected should we freeze a broader paired
EQA + small-OVMM assessment. Habitat OVMM, hardware, coverage-scan promotion and
large sweeps are not prerequisites for closing the correctly scoped repairs.

The candidate is a more auditable tool contract, not a demonstrated stronger
policy or a claim that the harness is optimal. Preserve common models, budgets,
seeds, physics, failure artifacts and raw final answers in the next assessment.

Clear slices independently. Non-draft means evidence is ready for review, not
permission to merge. Keep defaults unchanged; no full sweep or hardware gate.

## Acceptance review repairs — 2026-10-15

### Answer-authority audit continuation (23:08 ET)

The apparent scored regression mixes context quality with answer substitutions.
Fixed-response audit of the same EQA-12 artifacts: main's recorded final model
answers score 5/12 while evidence profile scores 8/12, despite reported totals
9/12 versus 8/12. Targeted seeds 1/2: evidence profile's final responses score
5/6, but its substituted answers score 1/6. This is a counterfactual on saved
responses, not a live accuracy claim; targeted examples are not representative.

`fix/final-model-answer-authority` at `9093b0c9` adds an isolated `model_final`
policy using the existing selector. Assessment guides acquisition, while the
final model owns semantic answer/confidence. No assessment/router substitution,
MCQ voting, uniform guess, geometric answer override or salvage inference.
Physical geometry admission and safety checks are unchanged. 177 focused tests
pass, plus the saved-response audit regression. Reproducible script and report:
`scripts/audit_final_answer_authority.py` and
`docs/plans/final_answer_authority_audit.md` on that branch.

Live paired 12-question control/new-policy pilot: `20261015_230408_fca6ac`,
`/tmp/emet-final-model-authority-eqa12-20261015`, 24 serial episodes at the same
model, seed, image selection and navigation settings. q14 reproduces a legacy
loss and a model-final success; trace confirms final model answer survives.
q15 model-final also scores correctly, but still reasons incorrectly from
absence in limited views. Keep model reasoning/coverage limitations visible.
Legacy summaries currently label the new policy's metrics `scored_policy:
no_salvage`; actual `decision_policy` and `answer_provenance` are `model_final`.
This is a metadata-label debt, not an observed semantic override.

Separately tested context repair: `fix/final-evidence-packet` at `be14f2ed`,
84 focused tests passed. Forward the best assessed packet even when unverified;
retain its full sequence/world-pose labels, not just generic context roles.
This is NOT in the running authority A/B. After it finishes, cherry-pick
`be14f2ed` onto the authority branch and test packet handoff separately. Do not
modify `/tmp/emet-reviewed-evidence-context` while job `230408_fca6ac` is active.

Configured-budget S0 completed on `2bf77670`: object find failed, destination
find succeeded with `grounded_object` geometry near the blue cube. Effective
timeout is recorded as 30 s and no phase exception occurred. This is a partial
proof of life, not full OVMM acceptance or a manipulation success.

### Current continuation checkpoint (22:08 ET)

- #185 is **ready for tooling review**, not merged; runtime/task gates remain
  open. Main and the shared dirty workspace have not been modified.
- Active clean-weight S0 control: `20261015_213514_ca66a7`, frozen `63016107`.
- Queued EQA discordance repeats: `20261015_215827_91edf0`, frozen `7c1bba8e`.
- Queued repaired S0: `20261015_220801_2bc43b`, frozen `2bf77670`, output
  `/tmp/emet-reviewed-s0-configured-budget-20261015`. Only two repairs differ
  from `63016107`: configured-timeout precedence and measured timeout travel.
  Roots stay frozen until their jobs finish; all use one exclusive GPU lock.

The clean-weight S0 control exposed an actual harness bug: YAML requests 30 s,
but the find runner overwrites it with its S0 default of 15 s. A command receipt
reports `motion deadline exceeded`; measured base motion is about 0.56 m while
the timeout attempt publishes `dist_m=0`. #185 `872e319a` resolves timeout after
loading parameters (explicit run override > YAML > existing defaults), and
records the effective budget in metrics. #182 `9a5aec0b` retains measured
displacement on timeout without changing failure/success/arrival semantics or
any tolerance. 28 timeout/config/metrics tests and 19 navigation tests pass;
the isolated integration passes all 47. Real-loader check confirms 30 s is used.
This repair changes effective budget, so its S0 diagnostic is **not** an
equal-budget accuracy ablation. Earlier S0 controls used 15 s on both arms;
RoboCasa/Molmo defaults were already 30 s.

Read-only published native TAMP snapshots confirm S0's objects remain on the
table (~0.52 m Z) and the robot is upright (~0.99999 body-up dot world-Z).
Published intrinsics imply a 43.1-degree horizontal FOV. No simulator truth was
sent to the agent or used for navigation. The first failed motion did not
establish a completed arrival, so do not attribute its subsequent empty view
to a VLM/SAM grounding failure. Diagnostic script: `/tmp/inspect-s0-framing.py`.

### Latest gate outcomes (evening)

- Repaired default integration `63016107` versus main `801dce82`, paired
  EQA-12 seed 0: **7/12 versus 9/12**, all 24 episodes completed. This does
  **not** clear the EQA gate. Artifacts:
  `/tmp/emet-reviewed-default-eqa12-s0-20261015`.
- Corrected-resource RoboCasa Stretch and MolmoSpaces rby1 retry completed
  all four episodes, but neither arm localized either requested target:
  `/tmp/emet-reviewed-room-resource-retry-20261015`. No room acceptance.
- Eight-view S0 control also failed both arms. Candidate Qwen recognized the
  red cylinder in its arrival RGB, but YOLOE grounding returned zero detections.
  This is a localization bottleneck, not evidence that Qwen missed that object:
  `/tmp/emet-reviewed-s0-scan8-20261015`.
- Existing VLM/depth-candidate/SAM2 diagnostic
  `20261015_211654_2470ed` encountered a missing checkout-local SAM2 checkpoint;
  construction attempted a network download during grounding. A partial
  download remains as evidence. Treat this run as dependency-contaminated,
  not a provider-quality comparison. The room driver now requires local
  checkpoint weights before starting and records their resolved path/hash.
  Cancelled that run and explicitly reaped its surviving pilot-owned children;
  original artifacts and `diagnostic_disposition.json` remain. Local-asset retry
  `20261015_213514_ca66a7` is queued behind EQA at
  `/tmp/emet-reviewed-s0-vlm-local-assets-20261015`. 62 focused acceptance,
  discovery, prerequisite-order and registry tests pass.
- Combined reviewed evidence branch `7c1bba8e` preserves repaired map epochs,
  actual-capture timing, bounded approach views and spatial labels. Its normal
  opt-in `configs/emet/evidence_context.yaml` completed EQA-12 seed 0:
  **8/12 versus main 9/12**, all 24 episodes completed without infrastructure
  errors. Job `20261015_212434_f4fcff`, output
  `/tmp/emet-reviewed-evidence-profile-eqa12-s0-20261015`. Defaults unchanged.
  Candidate loses q14/q15 and wins q65; q12/q56 recover the repaired-default
  misses. Summed episode wall time is 714 s versus 528 s (includes candidate's
  first cold asset download; not a pure steady-state latency comparison).
  Only the three disagreements are repeated at seeds 1/2, job
  `20261015_215827_91edf0`, output
  `/tmp/emet-reviewed-evidence-disagreements-s12-20261015`, queued behind S0.
  Complete portable gallery (29 packets, original PNGs, exact prompt labels):
  `/tmp/emet-reviewed-profile-gallery-20261015/index.html`.

Early checks on that live pilot confirm actual spatial prompt labels and four
retained images (current, pre-turn, two context). q12 now passes with both beds
and bedside tables visible in approach context; q14 still fails. Its final
model answer is correctly "In the living room", but the confirmed assessment's
"Next to the TV" wins the existing precedence rule. The context images do show
a TV elsewhere in the room; co-occurrence is not proof of the stated relation.
Do not patch the rule for this question: an offline audit of the previous
36-episode candidate finds nine confident final/scored disagreements; blindly
preferring the final model would lose five net correct answers. This is an
answer-policy/model-consistency limitation, not an established general repair.
Both final and assessed responses remain available for diagnosis.

Room metric validation now rejects `agentic_find_error` and
`agentic_find_error_recep` even when the CLI exits zero and emits boolean false
outcomes. The paired driver shares that validator and records its hash;
35 driver/metric tests pass. Ordinary scored misses remain completed episodes.

The linked SAM2 checkpoint constructs successfully on CPU both in a fresh
process and after loading the real default config. This validates dependency
availability, not mask quality. The integrated TAMP API/bridge suite passes
57 tests; the evidence/selection/history/capture suite passes 81 tests.

Do not substitute the earlier development result (26/36 versus 25/36) for
these exact-head gates. Navigation, recovery and evidence contracts have unit
coverage, but the shared-agent task acceptance remains open. Native TAMP tools
from main are retained; this work does not replace their interfaces.

The driver now hashes itself and `check_habitat_runtime.py` alongside its other
inputs. EQA captures the child exit status, reports `failed.txt`, validates each
requested artifact, then exits with failure; `set -e` no longer skips the checks.
Metric validation accepts one JSON object or one JSONL episode for either phase,
and reports missing/malformed artifacts as infrastructure errors without raw
tracebacks. The actual room writer emits indented JSON objects. Question/episode
lists use shell arrays, and the empty evidence-tag reservation is removed on
exit while actual episode evidence remains intact. 30 offline driver/metric
tests pass, with shell syntax checks. This does not turn a wrong scored episode
into an infrastructure error or claim a task success.

Related reviewed integration `63016107` includes main's native multi-object TAMP
tools (#212), timestamp repair (#215), navigation/recovery review fixes, and
#202's map-scope/current-capture repairs. 392 combined tests pass. Paired room
job `20261015_083001_7e7aef` completed against main `801dce82`: both S0 arms
failed their find targets; RoboCasa and Molmo arms hit resource-discovery errors
(checkout-local Robocasa assets and Molmo venv absent). No task pass.
Both roots now link the existing shared Robocasa package and Molmo environment;
asset completeness and wrapper discovery checks pass. Corrected-resource retry
`20261015_084043_140730` retains original failures and reruns those four cases.
S0's fast four-view preset samples 0/45/90/135 degrees while the targets are at
negative Y. Candidate upright telemetry and measured head tilt do not indicate
a fall; absence of an object from dark RGB alone cannot establish the cause.
Standard eight-view S0 diagnostic `20261015_083813_207ff7` completed on both
arms without a task pass. Do not infer perception failure from an unobserved
target direction. All jobs use the serial GPU lock.
Final-owner default EQA gate `20261015_084243_b8f836` completed: main `801dce82`
9/12 versus integration `63016107` 7/12, same 12 questions, seed 0, no
experimental selection overrides. Final task acceptance remains open.
The earlier candidate's EQA 26/36 versus main 25/36 is a modest development
result on different frozen revisions; prioritize reliable shared tooling and
do not present this as a meaningful accuracy gain or this PR's exact-head gate.

## Ownership and exact heads

| PR | Ownership | Current gate |
| --- | --- | --- |
| #181 `dc106273` | Opt-in grounding providers, shared assessment dispatch | Exact-head q12/14/56/6 attribution completed 4/4 at `slice181-eqa4-retry-20261012`; 118 grounding/conversation plus 11 assessment tests previously passed. Live/cached provider audit remains open. |
| #182 `9a5aec0b` | Measured navigation, clearance, wheel contacts | Reviewed exploration-mode propagation fixed; 42 route/sweep/arrival tests and 5 real-scene wheel-contact tests passed. New timeout travel repair passes 19 navigation tests. Short Molmo rby1 drive and final task gate remain open. |
| #184 `d2e922a0` | Bounded head/floor recovery, measured head-arrival fix, executor delegation | Shared status contract and two-total-capture bound repaired; integration suite passes. Intended live recovery demonstration remains open. |
| #185 | Frozen drivers, evidence isolation, queue ordering, environment discovery | 62 acceptance/discovery/queue/registry tests and 24 OVMM result/exception tests passed. Completed room failures are correctly reported, not task passes. Local-asset VLM retry pending. |
| #199 `9df2a382` | Preserve assessed camera pixels through final submission | Exact pixel identity is protected; no accuracy-gain claim. Joint repaired-head EQA gate is running. |
| #202 `914aa1c5` | Optional bounded view selection | Map-epoch, actual-current-capture and pin-budget repairs pushed; 87 owner tests passed. Four-image labeled packets validated live at integration descendant `7c1bba8e`, not a standalone owner-head accuracy gate. |
| #198 `59f70026` | Separate opt-in coverage scan, probe, exact-frame ingestion | 281 extracted coverage/recovery tests; 31 coverage/config-dispatch tests after repair. Corrected live ablation and multi-heading visibility gate pending. |

The measured-head fix moved from integration to #184; evidence isolation and
prerequisite-before-GPU-lock ordering moved to #185. #185 no longer owns the
executor behavior cleanup. #198 is based on #184 and does not block #185.

## Frozen pilot findings

- Baseline `7f65dbef` versus integration `cd92cb1b`, Qwen3-VL-8B-Instruct int4,
  seed 0: EQA q12/q14/q56/q6 scored **4/4 versus 1/4**. Losses on q12/q14/q56
  reproduced. No claim of EQA acceptance or statistical equivalence.
- Both arms failed object and destination find in S0 and RoboCasa. This is not a
  success gate. Molmo returned a wrapper-discovery infrastructure error in both
  arms and is excluded from task accuracy.
- Integration post-wheel route: 50/50 waypoints, maximum measured error
  0.00834215 m / 0.01457790 rad, unchanged 0.02 m / 0.03 rad limits.
- Integration model-free scan: 16 views / 69.19 s. All frames manually inspected;
  views 12/13 clearly show both objects with valid depth. Not a GT-mask benchmark.
- The nominal learned coverage arm recorded `mapping_scan_profile: local`:
  consumers incorrectly read `mapping/scan_profile` after config flattening.
  Fixed only in #198, with a real-loader dispatch regression. Discard that run
  as coverage-effect evidence; preserve it as the bug reproducer.
- Explicit `MOLMOSPACES_PYTHON` discovery resolved a venv symlink to the system
  interpreter. #185 now preserves the venv path; installed venv imports pass.
- Old room driver accepted zero-exit error JSON as job completion. #185 now
  requires a scored episode with boolean outcomes and no infrastructure error.

Artifacts: `/home/cpaxton/runs/emet/acceptance-plan-20261012/` records the initial
protocol. Frozen outputs are `pair-eqa4-{main,candidate}-20261012`,
`pair-room-{main,candidate}-20261012`, `pair-s0-coverage-20261012`,
`postwheel-route-20261012`, and `postwheel-coverage-20261012` under the same run root.
Do not modify their source trees or overwrite results.

## Integration diagnosis update (2026-10-12)

Manual review of arrival RGB, decision images, and agentic traces narrows the
failure mode; prefix attribution is still pending, not a passed acceptance gate.

| Question | Main versus integration | Interpretation |
| --- | --- | --- |
| q14 fan location | Same arrival XYZ; integration turns 30 degrees. Main image includes sofa/window; integration centers fan/plant and loses sofa. | Integration assessment guesses "next to the TV" without a visible TV. Object visibility is not sufficient room/relationship evidence. |
| q56 bookshelf room | Same arrival XYZ; integration turns 130 degrees. Main arrival RGB includes blue curtains; integration faces shelf without curtains. | Assessment infers white curtains from white walls. This is contextual evidence loss plus model overconfidence, not failure to translate to the goal. |
| q12 bedside-table count | Same arrival XYZ; integration turns 50 degrees and sees a partial bedroom instead of mostly wall. | Main's correct answer came from `mcq_debias`, not a successful visual count. Integration commits to one visible table; that does not establish the room total. |

Main's fourth success, q6, came from `uniform_prior`. Thus 4/4 versus 1/4 is
the scored outcome, not four strong visual successes versus one. These four
diagnostic questions do not estimate general benchmark accuracy.

There is also a pre-existing evidence handoff gap: q12/q56 assess the retained
voxel view `1099511627790`, but `select_obs_ids_for_verified_answer` only accepts
graph observations. Submission records `force_obs_ids=null` and
`last_eqa_obs_ids=[]`; saved final-answer inputs are older navigation views.
The final model abstains or disagrees, while the existing answer resolver
prioritizes the earlier confirmed assessment. q14 does attach graph obs16;
its salvaged living-room answer still loses to the confirmed assessment.
Do not conflate final-call reasoning with the scored answer's provenance.

Recommended repair boundary: preserve captured-view RGB through final answer
selection without manufacturing object nodes, and supply bounded context views
alongside object-facing views. Test exact evidence identity first; then paired
EQA and small OVMM. Do not revert measured heading correctness or introduce
question-specific answer rules based on these examples.

Diagnostic export also deletes `images/` during recorder flush, leaving stale
query-image links in traces. This review used retained decision images and
trajectory-matched arrival frames instead; `frames/rgb_0016.png` is a simulator
step, not graph observation16. Preserve per-query evidence in a separate repair.

The original prefix jobs never ran episodes: after changing into SOURCE_DIR,
the driver copied itself via a now-invalid relative path. Fixed and pushed in
#185 commit `f2281e35`; 27 acceptance-driver tests pass, including a new real
cross-checkout relative-invocation regression. Failed artifacts are preserved.
Retry jobs (exclusive GPU, seed0, q12/14/56/6):

- #181: `20261012_101709_3947df` (`slice181-eqa4-retry-1012`); EGL preflight passed,
  episode loop started.
- #182: `20261012_101732_85b60d` (`slice182-eqa4-retry-1012`).
- #184: `20261012_101755_169aec` (`slice184-eqa4-retry-1012`).

Exact #182 route probe completed with 51 frames and no reported failures at
`/home/cpaxton/runs/emet/slice182-route-20261012/summary.json`; native Molmo
startup/stationary render also completed under #185. Neither clears EQA or
learned room-find acceptance. Original #198 paired jobs hit the same driver
failure and remain pending a rerun; they produced no scan comparison.

## Run and disposition protocol

Use `scripts/run_extracted_acceptance.sh` through CPU-safe, GPU-exclusive jobs.
`SOURCE_DIR` selects the frozen checkout; `SOURCE_SHA` must match; `OUT_DIR` must
be new. `EMET_PY` and `HABITAT_BIN` select installed environments.
`ACCEPTANCE_CONFIG` supplies a frozen matched-provider room config; its default
is the separately advertised opt-in provider configuration, not a matched-main
ablation. `DEV_EQA_IDS`, `DEV_SEED`, and `FIND_EPISODES` select the pilot subset.
Record source and driver hashes. Keep model/seed/budgets identical across pairs.

1. Attribute q12/q14/q56/q6 across main → #181 → #182 → #184. Then complete the
   existing EQA-12 set on passing prefixes. Repeat disagreeing pairs at seeds 1/2;
   a loss on two of three matched seeds blocks the owning slice.
2. For #181, audit all advertised providers on preserved Molmo/RoboCasa RGB-D
   fixtures: accepted masks/geometry, missing/ambiguous targets, invalid depth,
   malformed model responses. Shared dispatch must preserve agent conversation.
3. For #182, run the five-repeat 50-waypoint gate on its head, minimal existing
   TAMP contracts, and a short native Molmo rby1 route. No relaxed tolerances,
   unknown-free assumptions, or navigation teleports.
4. For #184, demonstrate fresh downward evidence enabling same-waypoint recovery.
   Occupied floor remains blocked; impossible gaze and stale observations fail
   explicitly. Require bounded retries and successful intended recovery, not
   merely equally poor aggregate find scores.
5. For #185, validate source/output guards and missing-assets/native-crash/timeout/
   empty/error-metrics handling, then one live EQA pair and one live room pair.
   Correctly reported task failure does not itself block evaluation tooling.
6. For #198, run corrected same-provider local/coverage S0 plus four initial
   headings and an opposite-side fixture with offline GT mask/depth scoring.
   Report image/time cost. Do not condition policy on simulator truth.

For each slice: update its PR with exact parent/head, commands, artifacts and
limitations; mark ready only when its gates pass; request focused review.
After parent merges, retarget and rerun affected gates. No automatic main merge.

## October 12 evening: evidence selection and S0 coverage

- #199 main integration at `0fb37763`: EQA q12/14/56/6 remains **1/4** with
  identical predictions and no episode errors. q14's final Qwen answer is
  correct but loses to earlier confirmed-assessment precedence. This policy
  disagreement is separate from missing context on q56 and partial count on q12.
- Review repairs: SAM2 PIL input, lost first-sweep diagnostics, invalid-depth
  obstruction, and SAM2/Hydra controller initialization fixed. Combined suite
  passed 300 tests before subsequent integration work. Fallback clearance stays
  unknown by design; no synthetic 10 m value is exposed as measured clearance.
- #199 `4cbfc959`, job `20261012_211901_076023`,
  `/home/cpaxton/runs/emet/review-s0-20261012`: **0/2 find phases**, both scored,
  no infrastructure failure. The retained assessment view shows sky/empty floor,
  not the requested objects. This is not a manipulation trial.
- Model-free Stretch probe `20261012_212729_4d8354`, source `005f0d9a`,
  `/home/cpaxton/runs/emet/stretch-view-probe-20261012`: **5 frames, no probe
  failures**. Base remains upright (~0.999995 up-dot), four turns succeed under
  exploration tolerances. Objects appear at heading ~-1.88 rad (about 250 degrees
  around the initial heading), outside the original 4-view 0/45/90/135 scan.
  View 03 visibly contains the red cylinder, blue cube and table. This is a
  scoped wheel/camera control check, not precision approach or task acceptance.
- Actual Stretch MJCF head capability resolves to pan [-4.04,1.73], tilt
  [-1.53,0.79], both head actuators gear=1 and position bias=-80. This closes
  the model-definition concern, not hardware head-motion acceptance.
- #202 `1c5d85df`: optional existing-history selection, actual Habitat pre-turn
  capture, acquisition timestamps, and identical labeled assessment/final packets.
  431 focused tests passed. Defaults unchanged; no current-only grounding relaxed.
  Initial pilot `20261012_212607_6652d0` was **cancelled, not scored** after finding
  that Habitat's valid X,Z,Y world basis has negative determinant. Correction
  `a410ef7c` accepts consistent orthonormal reflected bases and rejects mixed
  handedness; 45 focused tests pass, including actual Habitat transforms.

Queued serial comparisons (do not modify their source checkouts until done):

| Job | Frozen source | Intervention | Artifact root |
| --- | --- | --- | --- |
| `20261012_213032_b11224` | #202 `a410ef7c` | Pose-selected current/pre-turn/history; Qwen8B int4, same EQA questions/budgets | `/home/cpaxton/runs/emet/selected-eqa4-retry-20261012` |
| `20261012_213404_0f6a06` | #199 `9df2a382` | Same four S0 scan images, now spaced 90 degrees; same find/model budgets | `/home/cpaxton/runs/emet/fullscan-s0-20261012` |

The S0 intervention changes angular coverage (135 → 270 degrees commanded),
not image count; do not call it an unchanged motion-budget comparison. The
eight-view default retains 45-degree spacing. Failed scan turns now stop rather
than count a repeated view as coverage. The map-boundary repair `005f0d9a` also
prevents legacy clamping from inventing visible border cells (15 tests pass).
The scan change has 40 focused passes. Performance is **not yet restored**.

### Live comparison follow-up

- Full-scan S0 `20261012_213404_0f6a06` completed at `9df2a382`: **2/2 find
  phases**, versus 0/2 at `4cbfc959`. Both objects localized from grounded
  objects, one investigation round and one navigation per phase; 164.7 s total.
  The saved assessment image visibly contains the red cylinder and blue cube.
  Four mapping images were retained in both runs, but angular coverage increased;
  this is a single-scene diagnostic, **not manipulation or general acceptance**.
- Selector retry `20261012_213032_b11224` was cancelled after its trace showed
  `view_selection_unavailable/current_acquisition_not_retained`: acquisition
  timing was nested under `Frame.info["info"]` rather than the expected metadata
  key. Its q12 fallback answer is not evidence about the selector. Fix `7c5b43f1`
  passes timing directly through voxel ingestion; 12 tests pass, including an
  actual mapping-to-selection boundary test. Both cancelled selector pilots must
  remain excluded from treatment accuracy.
- Corrected selector job `20261012_213834_3df84b`, frozen #202 `7c5b43f1`,
  `/home/cpaxton/runs/emet/selected-eqa4-live-20261012`, runs q56 first, then
  q12/14/6. Audit selected roles, timestamps and pixels before scoring efficacy.
- Room extension `20261012_214251_680c7a`, frozen #199 `9df2a382`,
  `/home/cpaxton/runs/emet/fullscan-room-20261012`, queues RoboCasa then Molmo
  find-only acceptance behind EQA on the exclusive GPU lock. No parallel heavy
  experiments, no hardware motion, no default selector change.
