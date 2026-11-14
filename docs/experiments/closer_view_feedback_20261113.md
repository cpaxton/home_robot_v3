# Coordinator-visible closer-view recovery

The previous [paired find pilot](active_sensor_agent_pilot_20261106.md) found
the bowl/cylinder but not their destinations in either arm. The microwave is
visible at about 3.34 m; an appropriate Qwen/SAM2 mask still fails the unchanged
2.5 m depth gate. That is a reason to obtain another observation, not proof of
absence or permission to manipulate a distant object.

## Scope

This branch applies the evidence contract proposed in
[PR #230](https://github.com/cpaxton/home_robot_v3/pull/230): observations,
model claims and measured effects stay distinct. Historical image cues remain
bound to their source capture. No second memory graph, persistent property
inference, automatic retry schedule or new manipulation authority is added.

- CHAT attaches its available camera observation on every tool round, retains
  earlier images in the dialogue and labels them as historical on follow-up.
  It does not claim that a cached camera observation is a fresh acquisition.
- Compact tool results retain capture, visual and geometry feedback. Versioned
  motion envelopes render their measured data rather than Python dictionary
  representations. Existing legacy prose remains supported.
- Rejected mask proposals retain image bounds, edge clipping and calibrated
  image bearings when available. A base yaw offset is supplied only with valid
  camera/world/base calibration. These are **unverified proposal directions**,
  not object XYZ, semantic confirmation or a free-space route. Missing or
  reflected calibration cannot manufacture a base bearing.
- Exploration arrival preserves task gaze instead of unconditionally leveling
  the head. Existing navigation, floor sensing and obstruction checks remain.
- The `model_final` prompt permits model-selected observation/movement actions;
  it no longer requires an INVESTIGATE/EXPLORE sequence. Legacy and grounded-v2
  prompts retain their existing behavior. The central model chooses the next
  action and final answer; executors validate physical motion and admission.

No provider default, depth limit, instance admission, motion tolerance, budget
or answer override changes. Proposal metadata does not resolve a mask provider
that proposes the wrong object. The existing opt-in providers remain separate.

## Contract validation

246 tests pass across CHAT/HTTP history, tool dispatch/rendering, controller
grounding, visual consumers, EQA room/navigation, relative motion and CPU MuJoCo
geometry. The HTTP test uses controlled transport and robot observations: it
checks that a partial move's resulting image, original image and measured
shortfall reach the next model request. It is not a live-model success claim.
The controller test also checks that grounding feedback from a different view
is not presented as current geometry. Lint and whitespace checks pass.

The CPU ray controls demonstrate the existing range boundary (3.4 m rejected,
2.2 m supported) and preserve a blocked-route negative control. They do not
qualify the real robot controller or demonstrate language-model recovery.

The first live context audit exposed remaining legacy rules in per-round state,
tool descriptions and direct submit/finish handlers, despite the model-final
system prompt allowing model-selected actions. The follow-up removes those
contradictions in `model_final`: it preserves measured action/capture history,
allows the model to request completion, and leaves legacy completion gates in
their own policy. Localization still requires independent admitted geometry;
an early semantic answer cannot certify a manipulation target.

## Live development gate

Run one serial paired `molmo_rby1_ithor_s2_idx0` find episode through
`scripts/run_shared_room_pairs.py`. Compare the frozen main-plus-common-clock-
repair revision `35fbabf7` with this candidate; both use `model_final`, Qwen3-VL
8B int4, four scan steps, six rounds, three navigation steps, identical cached
weights and the same 2.5 m range gate. Use the installed Molmo Python override,
safe CPU pool 12,14,20,21, one managed GPU lease and offline weights. Keep at
least 20 GiB free and retain failures.

This is a complete-system diagnostic against main plus the shared runtime
repair, not an isolated prompt/arrival-head ablation. Inspect model inputs and
tool receipts before interpreting the outcome: does the target remain in the
arrival image, does geometry failure reach the model, does it choose a feasible
observation, and does that capture actually improve range/context? Find success
does not qualify pick/place. Record the exact candidate SHA and all failures in
the run manifest. Do not run a broad sweep until this gate is understood.

## GraphEQA review follow-up

Fork commit `f41ac89` normalizes NumPy execution receipts at the execute/history
boundary, rejects unsupported or nonfinite values while preserving the decision
after attempted execution, and removes NumPy wrappers from legacy position
history. Fifty CPU tests pass, including the mapping-config tests. The expected
cross-repository sensor-validator entry point and source hash are documented;
there is no runtime compatibility handshake yet. Rebuilding scene context per
packet remains a profiling follow-up, not a correctness repair. These changes
do not claim to fix the reference planner's empty plans or benchmark accuracy.

## First live pair: mixed, not accepted

Run `~/runs/emet/closer-view-molmo-20261113-r1/` freezes baseline `35fbabf7`
and candidate `d4d52cbf`, before the per-round-state/completion correction.
Both episodes complete normally; the driver exits 1 because the task gates fail.

| Arm | Bowl | Microwave | Complete task | Reported query / episode wall seconds |
| --- | --- | --- | --- | --- |
| Main + common clock repair | pass | fail | fail | 263.01 / 369.90 |
| Coordinator recovery candidate | fail | pass | fail | 136.63 / 189.46 |

Both retain 1/2 find phases. The candidate reaches a clear microwave view and
admits an appropriate mask with 21,574 supported pixels at median depth 0.969 m,
all within the unchanged 2.5 m gate. The accepted isolated mask was manually
checked against the target. Its geometry succeeds after the preceding bowl
phase has moved the robot; this is not proof that the model deliberately used
the new bearing metadata to close a range rejection. The bowl failure prevents
an end-to-end task improvement claim. Initialization/cache effects and different
model-selected trajectories also prevent attributing the runtime difference to
the repair alone. A single stochastic pair does not establish superiority.

Exact requests confirm the conflicting legacy stop instructions in candidate
per-round state. The next frozen pair tests their removal with identical
model, scan, budget, provider and range settings. Preserve this mixed result
separately; do not relabel it as a successful recovery qualification.

## Repeat and failed-investigation feedback repair

Run `closer-view-molmo-20261113-r2` freezes the same baseline and candidate
`b8728dcf`, with the state/completion correction. Both episodes complete;
main again passes bowl/fails microwave (query/episode 93.78/143.18 s), while
the candidate fails both (100.93/150.70 s). Exact candidate requests confirm
the legacy no-submit/limited-choice instructions are absent. This is a red
task-quality gate, not a successful correction or a runtime win.

The initial bowl is plainly visible in the manually checked candidate image.
Both arms first choose an investigation whose planner returns `no_target`.
That tool returns no capture or assessment on failure. Main subsequently
chooses exploration, which captures/assesses even when navigation fails and
thereby admits the bowl. The candidate instead spends rounds on floor/motion
recovery without obtaining that object admission. Its first failure is not
evidence that the model could not see the bowl.

Make failed investigation consistent with failed exploration: acquire a real
new view at the measured current pose and return its explicit assessment
alongside the unchanged navigation failure. Do not aim as if the intended
arrival happened, repeat motion, synthesize object coordinates, or assess a
stale/nonadvancing capture. Capture/model errors remain separate fields and
cannot mask the original navigation receipt. Four new regressions cover fresh,
nonadvancing, disconnected-camera and failed-assessment outcomes.

The follow-up is one **candidate-only** Molmo diagnostic with unchanged model,
provider and budgets, retained separately as `closer-view-molmo-20261113-r3`.
Its purpose is to check this missing feedback path. It is not a new paired
comparison and cannot establish superiority over main. Broader EQA and complete
pick/place acceptance remain outstanding.

### Candidate-only feedback check

The `b9f0c726` diagnostic completes normally: bowl pass, microwave fail, query
69.26 s and reported episode 118.36 s. The bowl uses admitted object geometry.
The saved actual-context audit shows a `no_target` investigation still reporting
`nav_progress=false` / `nav_finished=false` while its new capture and positive
geometry feedback reach the next model context. All six saved coordinator
requests omit the contradictory legacy stop/limited-choice instructions.
This restores the recent main phase score; it does not establish superiority.

The microwave phase first requests targeted floor sensing without a rejected
footprint, then a turn rejected for unknown floor. Subsequent sensing succeeds,
but its 0.5 m forward request is rejected for low clearance. A later turn loses
the target view; the final model still recalls that the microwave is above the
stove, but there is no admitted destination geometry. No useful target-specific
closer view was acquired in this attempt. Do not describe this as another
measured out-of-range mask failure: the remaining motion/context sequence is
different. The safety rejections and absent geometry are retained.

Stopping audit also found that the legacy evidence state could automatically
end model-final episodes immediately after a positive auxiliary assessment.
Disable that automatic submission in model-final mode: the coordinator receives
the feedback and chooses whether to continue or submit. A real-loop regression
checks both continued sensing and explicit submission, while legacy retains its
automatic stop. Normal budget finalization still invokes the final answer model.
Run the frozen head once more as candidate-only `closer-view-molmo-20261113-r4`
to verify that this completion handoff reaches the actual model. These diagnostics
are development evidence; a broader matched EQA/find pilot remains a merge gate.
