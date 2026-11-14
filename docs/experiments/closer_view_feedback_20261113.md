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

233 tests pass across CHAT/HTTP history, tool dispatch/rendering, controller
grounding, visual consumers, EQA room/navigation, relative motion and CPU MuJoCo
geometry. The HTTP test uses controlled transport and robot observations: it
checks that a partial move's resulting image, original image and measured
shortfall reach the next model request. It is not a live-model success claim.
The controller test also checks that grounding feedback from a different view
is not presented as current geometry. Lint and whitespace checks pass.

The CPU ray controls demonstrate the existing range boundary (3.4 m rejected,
2.2 m supported) and preserve a blocked-route negative control. They do not
qualify the real robot controller or demonstrate language-model recovery.

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
