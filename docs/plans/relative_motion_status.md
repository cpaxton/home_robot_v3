# Shared relative movement implementation status

## Re-review closeout, 2026-10-25

- Rotation schema, CHAT dispatch and legacy controller all use the same ±180° bound.
- Both navigation execution and legacy receipts use `relative_motion_progress`
  for body-frame displacement and SE(2) arrival. Forward motion intentionally
  checks heading drift; unconfirmed arrival retains the measured displacement.
- Head feedback exceptions return structured failures through `observe_floor`;
  direct blocked-floor aiming also catches unavailable measured gaze. The
  head-sweep loop catches unavailable feedback; forward-distance clipping does
  not read head pose in this revision.
- Synthetic policy IDs are negative while real observation IDs are positive.
  The legacy float result is documented as zero unless arrival is confirmed;
  structured receipts preserve partial progress.
- 134 targeted motion/gaze/room/visual-consumer tests pass. Live recovery and
  manipulation qualification remain separate acceptance gates.

## Re-review: behavior migration and optional gaze startup

The prior change updated verification output but missed three control paths:
close-place exhaustion, proposal blocking and claim retraction still compared
to legacy retrieval ABSENT. These now consume explicit visual evidence;
skipped/missing assessments cannot establish visual absence. The attempt
ledger also understands VISUAL_PRESENT/VISUAL_ABSENT/NOT_ASSESSED. Legacy
retrieval comparisons remain only in score generation, trace diagnostics and
backward-compatible ledger parsing, not these semantic behavior decisions.

Unsupported measured gaze remains an explicit capability error, not a fabricated
zero pose. The shared agent no longer fails startup merely because it constructs
a Stretch compatibility wrapper: that wrapper defers gaze reads until a
gaze-dependent operation and rejects before actuation if unavailable. Actual
registry specs for Sourccey/Nori/Franka are covered by startup/use tests.
194 focused behavior/robot/grounding tests pass (plus lint); new behavior tests
exercise disagreement between cheap retrieval and visual presence.

Job `20261023_132503_95dd0d` completed without infrastructure failure on
pre-fix `f723a151`: q12 wrong, q14 wrong, q6 correct, q56 correct (2/4).
This does not clear accuracy acceptance. Re-run only after review fixes settle.
The re-review message's detailed P1/nav-posture/rotation/map-clipping sections
were truncated and not posted on GitHub at inspection; requested full text.
Do not claim those remaining findings resolved based on their headings alone.

## Review follow-up — current

PR #220 review fixes in `01ae5412`: EQA adapters retain dictionaries;
motion envelopes parse as dictionaries or JSON and malformed/legacy motion
text cannot become success; zero/sub-millidegree requests do not execute or
consume navigation budget; place inspection reads separated visual presence.
The registry-level gaze suite enumerates all adapters/aliases and explicitly
checks the Mobile ALOHA/YOR stubs as unsupported. It caught and fixed Mars'
named `joint_head` observation feedback, outside its arm/base joint vector.
112 focused tests pass, including the previously red room/navigation file.

Matched EQA12 job `20261023_104449_f2f394` finished with CUDA allocation/cuDNN
failures in the candidate arm. This pair is not valid acceptance evidence;
failed artifacts remain preserved. Current GPU inspection is idle and does
not establish which process held memory during the earlier failures. Recheck
GPU availability and owned process cleanup before a new matched run; do not
score infrastructure failures as model misses or claim restored performance.

## Closeout progress — 2026-10-23

Frozen `50575a34` diagnostic `20261023_103515_22ac60` also completed:
bowl found, microwave not localized (123.8 s). It confirms failed-turn batch
stopping and the model's selection of targeted floor recovery. Recovery now
reaches the real head-arrival gate, but returns `head_pose_unconfirmed` with
measured `[0, 0]` for a nonzero requested gaze. Source inspection identifies
the adapter defect: GenericZmqClient only read XLERobot gaze joints and
returned constant zeros for every other robot.

The next repair populates existing `RobotSpec.look_joint_names` from the
already implemented server mappings, reads those measured joints, and fails
explicitly on missing/nonfinite feedback. It adds no new torso/head actuation
mapping and does not certify torso motion as collision-free. Unqualified
adapters do not advertise targeted floor observation. 255 focused gaze,
head/floor and MuJoCo geometry tests pass; live validation is still required.

Frozen `a98f101b` pilot `20261023_102419_6dd703` completed without an
infrastructure failure: bowl found, microwave not localized (109.0 s).
The model did not select floor inspection in this run; therefore it does not
validate the repaired handoff live. Outputs remain at
`/home/cpaxton/runs/emet/relative-floor-handoff-molmo-20261023`.

Next repair separates visual presence, retrieval proposals and geometry
admission in the model-facing verification result; missing RGB is explicitly
not assessed. It also preserves navigation budget for requests rejected before
execution and stops a queued relative-motion batch after a failed turn/move.
339 focused contract/regression tests pass. No safety threshold changed.

Remaining order: isolated simulation observation cases; serial matched
S0/Molmo and 12-question EQA development acceptance; integrate #213 through
main when merged and run its qualified TAMP smoke; review/land the context
and movement stack only after evidence clears; then implement EXPRESS-Bench
and AbstainEQA adapters and small audited pilots. No full sweep or hardware.

Safety/coverage cases and reproducible fast gate:
`docs/testing/agent_observation_cases.md`. Benchmark requirements:
`docs/environments/embodied_qa_extensions.md`. Full agent acceptance is still
held, not implied by the passing contract suite.

2026-10-23; repair branch `fix/shared-relative-motion-results`.

The CHAT and EQA/OVMM adapters now share `move_forward(meters)` and
`rotate_base(degrees)` schemas. EQA registration requires physical map mode
and the ordinary validated navigation methods; unsupported adapters retain
their existing tool packs. Forward requests are bounded to 0–1.5 m and turns
to ±180 degrees. Invalid/nonfinite inputs cannot command motion.

The shared physical implementation transforms body-relative requests into
world goals, resets the planner, validates the swept footprint without an
exploration-goal exemption, then uses the existing validated waypoint executor.
It reports versioned semantic receipts compatible with the TAMP envelope,
measured body-frame displacement, and independent arrival verification.
EQA counts attempts against its existing navigation budget and captures and
assesses the actual post-motion view, including after partial/failed execution.
Capture or assessment failures do not relabel completed motion as failed.

The existing outer-agent legacy movement path now checks server return values
and measured progress rather than reporting commanded motion as completion.
It remains a compatibility path, not authorization to expose it in autonomous
EQA on adapters without the validated-navigation capability.

The preliminary expanded suite passed 150 tests across movement, shared skill
packs, prompts/results, swept footprint, physical contracts and conversation
preservation. These are unit/contract tests, not live navigation acceptance.
No claim yet that the microwave recovery succeeds or that new tool availability
preserves EQA accuracy. Run serial managed simulation pilots before merging.

PR #219 has merged into #213 (`092d34a2`), not main at last fetch. Do not
implicitly import that pending stack. Its planning/execution tools remain
provider-qualified: this change does not expose privileged native planning
inside an ordinary EQA episode or create a second task-provider registry.

Outstanding: live Molmo/S0 and EQA gates; separate visual presence, retrieval
and geometry statuses so weak retrieval does not summarize a visible target as
ABSENT; qualify other robot adapters through the same navigation capability.
