# Living TODOs (small / near-term)

## Next gates for the shared agent harness

Use one production CHAT tool loop for CLI, Discord and robot adapters. Each
adapter declares its sensors and actions. A physical tool uses current
observed evidence, checks the action, reports measured completion or a typed
refusal, and supports a confirmed stop. Save a replayable trace and score
the outcome independently. Follow the
[assessment protocol](docs/testing/agent_harness_assessment.md) and keep
scripted controls, model trials, assisted trials and hardware results labeled.

- [ ] Package the recent Stretch sensing and command changes in reviewable
      PRs and freeze a clean assessment revision. The tracked live runner and
      scorer are under
      [the assessment protocol](docs/testing/agent_harness_assessment.md#reproduce-the-surveyed-stretch-navigation-controls).
      Keep the raw evidence
      linked from the [assessment report](docs/experiments/harness_assessment_20261119.md).
- [x] Verify live observed Stretch navigation on the surveyed simulator start.
      Check one targeted extra floor view with the same clearance sweep, a
      clear drive, an occupied refusal and a stale-odometry stop. The independent
      RGB-D check passed clear arrival, wheel slip and reference-pose loss
      through the normal CHAT waypoint path.
      [Live controls](docs/experiments/harness_assessment_20261119.md#live-independent-motion-guard).
- [x] Run a real-model read-only turn and a bounded CLI drive on the surveyed
      Stretch simulation. Qwen3-VL selected both public tools and received the
      measured drive result.
      [Terminal trials](docs/experiments/harness_assessment_20261119.md#real-model-stretch-terminal-controls).
- [x] Verify Discord delivery in the private test channel. The scene replay
      returned a measured-camera description, and a live operator message
      produced a 240 × 424 PNG that the operator confirmed. The reopened
      session answered a visual relation follow-up and reported an absent mug.
      [Startup and delivery](docs/experiments/harness_assessment_20261119.md#discord-operator-trial-startup).
- [x] Add an automated full Stretch CHAT session covering a current visual
      answer, a spatial follow-up, an absent object and a measured-depth map.
      The full registry runs with an image sink and no Discord connection.
      [Terminal controls](docs/testing/agent_harness_assessment.md#real-model-terminal-controls).
- [ ] Qualify bounded exploration in Discord. The head-view control passed
      with two measured angles and distinct camera images. The cancellation
      control matched the active waypoint and confirmed wheel rest in a
      separate simulator run. Repeat the head views, one guarded forward
      step and explicit `stop` through Discord.
      [Discord protocol](docs/testing/agent_harness_assessment.md#discord-in-simulation).
- [ ] Close the live stationary observed-input gate in
      [#245](https://github.com/cpaxton/home_robot_v3/issues/245). Supply fresh
      object and support geometry, synchronized robot feedback, and the motion
      refiner. Require one public-tool pick and place with independent lift,
      release and settling checks, plus stale-source and post-grasp evidence-loss
      stops. Then test a stopped base handoff with fresh destination evidence.
      The live control measures the destination and synchronized arm pose and
      rejects an expired capture. The cylinder's visible depth and the 17 mm
      head sweep still leave its full shape and rigid pose unqualified.
      A wider scan supplied up to 24 reference landmarks on a textured
      cylinder. Its tabletop contact still prevented a complete bound.
      Qualify a measured or calibrated payload envelope and bind the live
      source and refiner.
      [Transfer protocol](docs/plans/2026-11-18_tamp_agent_closeout.md#remaining-execution-qualification).
- [x] Freeze the RoboCasa365 Stretch pilot and check the 12 adapted fixtures.
      All render with metric depth and pass upstream predicate controls on
      known initial and constructed success states.
- [ ] Qualify measured reachability and action admission for the RoboCasa
      control and held-out episodes. Sink transfer requires the observed
      transfer gate. Drawer opening requires observed articulation. The
      composite also requires the stopped-base handoff.
      Use PandaOmron, RoboCasa365's native embodiment, for the sink pilot.
      Its stopped-base bridge publishes calibrated front and wrist RGB-D with
      synchronized joint feedback, and executes native arm, torso and gripper
      commands through the shared client. The archived tongs transfer passed
      through that client using a simulator-state-assisted reference trajectory.
      Production CHAT starts without issuing navigation or camera motion on
      the stopped profile. The right camera shows both the tongs and sink.
      The optional live provider now discovers the shaker task from
      synchronized depth, an asset-derived shape envelope and measured support
      surfaces. Measured counter depth around the shaker also supports a
      pickup-height check under occlusion. Motion-scoped checking now retains
      fixed chassis contact, and an optional 5 mm local depth map checks arm
      geometry. A sampled departure planner can start in contact with bounded
      overlap slack. Recovery now uses measured surface patches instead of
      voxel contact identities, with bounded sliding and 5 mm patch slack under
      a separate hard penetration cap. Qualify live execution/contact monitoring.
      Audit found missed robot surface returns in the local map; actual visual
      mesh association and vertical-first escape probes now have regressions.
      Scoped original-patch clearance now survives goal connection and runtime
      handoff; regressions cover prefix/suffix commands and newly seen obstacles.
      Panda has executed a full recovery prefix and clearance suffix: five
      commands, 7.04 mm torso progress, zero scored unintended contacts.
      An assisted development run now physically completes the full sink transfer
      with 10.1 cm two-pad lift and zero unintended touches or scored contacts.
      Full observed verification now also passes after fixing historical-reference
      age, uncertainty-inclusive settling and tiny joint-stop overshoot handling.
      The reproducible assisted recipe and stack changes are documented below.
      Collision-aware, held-out and
      model-selected full transfers remain unqualified.
      Shared stack changes and the reproducible development profile are mapped in
      `docs/testing/observed_manipulation_stack.md`: three-camera sensing, native
      feedback, convex contact, bounded IK/Cartesian planning, and runtime path
      revalidation. Require verified lift, sink placement and independent contact
      scoring before closing this item. Native reset now holds the intended absolute
      controller posture; auxiliary RGB-D views share the primary snapshot.
      Checked three-leg corridors admit pregrasp, grasp and lift planning.
      A newer bypass control moved the tool 46 cm with zero scored unintended
      contacts, then stopped on visual unobservability without a verified lift.
      An explicitly assisted bypass run now physically and visually verified a 9.1 cm shaker lift
      with both finger pads and zero scored unintended contacts, then stopped
      during transport on uncertain relative-pose estimates. It does not qualify collision-aware transport
      or placement. A later simulation-only carry model plus surface association
      enabled the physical transfer described above; unfamiliar-object geometry
      and occluded retention still need qualification.
      Native reference motions
      also need self-collision repair; the scorer now measures self contacts.
      Simulator-only collision bypass controls moved the end effector 5.1 cm
      without scored unintended contacts and 12.6 cm with unintended contacts.
      Both stopped on visual association and remain incomplete transfers.
      CHAT now permits at most two autonomous recovery rounds for explicitly
      unstarted planning failures: inspect fresh imagery or query the VLM,
      adapt the view/approach and request a fresh plan. Partial or unknown
      execution does not enter this recovery path. Explicit simulator overrides
      remain recorded; VLM judgment does not certify metric clearance.
      A 5 mm map cell is resolution, not sensor accuracy. Qualify uncertainty
      margins and monitored contact independently on physical robots.
      Fixed-camera tracking can now continue independently of wrist-based
      initialization. Every segment of a proposed home-posture or configured
      demonstration detour is checked before execution; route proposals do not
      waive collisions. The bypass contacts were at a cabinet handle during
      pregrasp travel, before reaching the object. Qualify this corridor first.
      Resolve moving-view association and qualify precise collision checking
      against fresh local geometry. Qualify continuous pose tracking through
      motion and complete pickup, carry and release through production CHAT
      with observed clearance. Qualify the guarded placement fallback on a
      loaded wrist after the bridge's empty-wrist stop controls.
      The shaker is the simpler next target. Its depth patches cover the
      placement footprint. A sensor-seeded native run passed after testing
      collision detours, torso motion and a longer retreat under simulator
      supervision. Bind live evidence before repeating it through CHAT.
      Seed 1 is a development diagnostic. Freeze a fresh held-out scene
      for qualification.
      The earlier Stretch control measured the unloaded approach and
      above-rim reach. Its tongs candidates stopped before closure.
      [Follow-up experiments](docs/experiments/harness_assessment_20261119.md#tabletop-manipulation-and-robocasa-reachability-follow-up).
- [x] Demonstrate a runtime VLM-guided transfer without object CAD or continuous
      surface registration on the tuned PandaOmron development fixture. Qwen3.5-9B
      selected regions/direction; depth and native control completed pick, place,
      release and retreat. Official success passed with 9.35 cm two-pad lift.
      Contact telemetry recorded 355 shallow frames and zero >1 mm or self-contact
      frames; contact itself is not a task failure. Same-image left/right instructions produced separate
      depth-supported basin goals. This is scripted phase routing, not agent
      qualification or a held-out result. See `docs/testing/observed_manipulation_stack.md`.
- [x] Add current-image grasp/retention checks after lift and before lowering,
      with one bounded regrasp when visual evidence and native finger feedback
      agree the hand is empty. Pear loss is detected; successful recovery and
      production CHAT integration remain unqualified.
- [ ] Stabilize VLM-adaptive transfer beyond the tuned grinder fixture. Frozen
      pear/garlic/tongs development screens v1–v8 each scored 0/3 full transfers;
      pear achieved 8.51 cm lift in v2 but slipped during carry. V6 passed the
      approach stop and lifted 7.52 cm, then stopped during lowering; retention
      through carry is unverified. Resolve arrival,
      retention and surface grounding; preserve all rounds. VLM estimates of
      camera, grasp height/direction and approach are allowed harness inputs.
- [ ] Qualify the no-CAD VLM path on frozen 20-case pilots (16/20 full success
      separately for scripted and real-agent routing), including absent targets,
      distractors, changed instructions, recovery, and cross-robot transfer.
      Characterize contact outcomes and verify results with fresh views plus
      native feedback; do not treat VLM-only image judgments as ground truth.
- [ ] Port the observed-transfer contract to each benchmark's chosen
      embodiment. Use PandaOmron for RoboCasa365 and Stretch for the in-house
      tabletop control. Evaluate MolmoSpaces/rby1 next. Select and qualify
      additional native controllers against the tasks they will run.
      [Onboarding recipe](docs/testing/agent_harness_assessment.md#port-a-benchmark-arm).
- [x] Verify that Stretch can revisit and see a block whose position changed
      while out of view. The explicit head revisit recovers the moved block
      without base motion.
- [ ] Qualify free-form changed-object recovery. Keep the base stationary
      when the user asks for a head revisit and bind current-view metric
      grounding in DynaMem.
      [Expanded assessment](docs/experiments/harness_assessment_20261119.md#expanded-simulator-assessment).
- [ ] Verify stationary Stretch and Innate Mars sensing after simulator review.
      Record timestamped RGB-D, intrinsics, robot feedback and TF/pose, and
      characterize dropouts. Review each robot's independent stop path before
      any hardware motion session.
      [Stationary hardware protocol](docs/testing/agent_harness_assessment.md#stationary-hardware).
- [ ] Test portability on a small frozen set before expanding the full
      assessment. Cover a held-out Stretch layout, a second embodiment through
      the same public interface, and a read-only evidence task with a changed
      view or an absent target. Repeat the bounded floor recovery and verify
      independent pose coverage on the held-out Stretch layout. Preserve
      exact action and model packets, source and fixture IDs, independent
      verdicts, frame age, and wall, simulator, model and planning time.
      Export the coverage matrix, a measured route and a success/refusal
      pair for figures.
      [Figure inputs](docs/testing/agent_harness_assessment.md#evidence-and-paper-figures).
- [ ] Qualify Sourccey as a second simulated embodiment. The default-table
      bridge publishes front RGB-D and joint feedback, and the absolute
      table-facing velocity turn passes. Shared CHAT reads the camera and
      returns a map with 76 explored cells without actuation during the
      turns. Qualify measured localization and collision evidence before
      observed waypoint or manipulation tests.
      [Sourccey simulator smoke](docs/robots/sourccey.md#motion-and-mapping-smoke-test).
- [ ] Establish a model baseline before training the tool-using agent. Save
      the observations and receipts available at every decision, including
      what caused a refusal and which follow-up view helped. Compare the
      baseline and a trained model on the same held-out tasks, public tools
      and action checks. Use each embodiment's native controller as the
      control baseline under the shared tool and observation contract. Add an
      embodiment and skill to the controller training track when native runs
      establish a repeatable control or retention gap. Compare learned and
      native controllers on the same task, inputs and held-out fixtures.
      Evaluate the Mars teacher-action policies on this controller track.

## Parallel TAMP controls and performance

- [x] Land [#250](https://github.com/cpaxton/home_robot_v3/pull/250) and
      [#251](https://github.com/cpaxton/home_robot_v3/pull/251). The original
      assisted clutter cohort completed 2/7. The collision-check shortcut saved
      0.37 s in the matched admission replay, and both full searches still expired
      after about 275 s.
- [ ] On the frozen revision, run the reference-certified native stationary
      transfer through `emet run agent`, plus stop-only and separately authorized
      recovery controls. Preserve the assisted input label.
      [First agent test protocol](docs/plans/2026-11-18_tamp_agent_closeout.md#first-agent-test).
- [ ] Profile time to a certified plan and replay cube-back release plus the
      unknown starting volumes against passing controls. Track capture age,
      plan time, recertification and release separately. Keep source age,
      clearance and contact checks fixed.
- [x] Reconcile [#228](https://github.com/cpaxton/home_robot_v3/pull/228) with
      [#231](https://github.com/cpaxton/home_robot_v3/pull/231) and
      [#229](https://github.com/cpaxton/home_robot_v3/pull/229) with
      [#247](https://github.com/cpaxton/home_robot_v3/pull/247). The Stretch clock
      cache and grounding-range feedback are on main. Loaded episode timing and
      the safe closer-view Molmo check remain open below.

## Shared-agent and benchmark work

- [x] Reconcile #247 with main #246 and address the posted review. Preserve
      public TAMP receipts and measured settling; throttle capability retries,
      reject missing snapshot RGB and protect recorded credential fields.
      Frozen integration: 1,776 passed, one optional Habitat skip with a separate
      passing configuration probe; paper builds to 73 pages. Re-review record:
      `docs/experiments/pr247_review_20261117.md`. No new model-quality claim.
- [x] Bind optional gaze arguments to adapter capabilities; native Habitat uses
      acquisition-time confirmation rather than requiring a streaming sequence.
      Preserve bounded movement RGB/pose/timing independently of voxel updates
      and feed it to existing view selection. Keep query images and request
      traces through debug finalization. The focused pack passes 789 tests;
      native/live and frozen-packet qualification are separate gates.
- [x] Document the full registry and evidence/action contracts in
      `docs/agent_tool_contracts.md`; coverage checks include all registry names.
- [x] Qualify the new native head path, frozen Q12 overview/detail packets and
      an end-to-end Q12 replay; preserve failures and abstentions. Native gaze
      passes; Q12 remains wrong despite retained movement views and two tables
      visible in the current image. No accuracy gain established. Report:
      `docs/experiments/observation_contracts_20261117.md`. Prior paired
      micro-pilot: EQA 0/2 both arms; S0 candidate object+destination 2/2 versus
      pre-gaze 1/2 (212.2 vs 656.7 s episode wall). One seed/two questions do not
      establish general superiority or a current-main merge gate.
- [ ] At the next benchmark refresh, run a matched-source small EQA/simple-OVMM pair with complete
      assessment PNG packets saved. The packet audit extension passes 45
      focused tests; do not treat the candidate-only Q12 replay as a pair.
- [ ] Evaluate a small subset of [NVIDIA USDCraft-10K](https://huggingface.co/datasets/nvidia/USDCraft-10K)
      assets for new articulated OVMM/TAMP/EQA scenes (doors, drawers, appliances
      and hidden-object tasks). Assess OpenUSD/PhysX to MuJoCo/MJCF import;
      preserve joint limits, collision shapes, scale and physical properties.
      Validate opening/closing and retain asset IDs, revisions and attribution.

- [x] Have a fresh-context subagent actually operate production CHAT tools in
      live simulation. Independent six-round trial and informed eight-round
      follow-up both fail localization; preserve failures and separate loop
      completion from task success. Report and evidence paths:
      `docs/experiments/independent_operator_trial_20261116.md`.
- [x] Repair the observed CHAT FIND handoff: failed `_find` must set task failure,
      executor-mapped information tools must deliver detailed receipts and permit
      model follow-up. Do not execute tools twice or equate executor liveness
      with localization. Successful/failed FIND now reach the next production
      CHAT/HTTP request with detailed navigation feedback and execute once;
      mapped information tools honor returns_info. The 95-test focused pack,
      including existing scan checks, passes. Live task success remains unproven.
- [x] Qualify rby1/Galaxea-proxy measured gaze in the real simulation bridge:
      the earlier physics worker had died on the physical guard's rejected legacy
      pose hold while publication remained alive. Physical mode now skips that
      hold; fatal physics stops publication. All four live head configurations
      pass unchanged tolerance, with advancing simulation time and an upright
      base; 76 focused CPU tests pass. Mark earlier r2/r3 trials infrastructure-
      invalid. Informed r4 retry sees the microwave and recovers floor coverage,
      but obtains no admitted object position. See the operator report.
- [x] Route CHAT face_toward through the existing world-frame conversion and
      validated rotation handler; preserve measured success/rejection instead of
      reporting legacy executor liveness as turn success. CPU acceptance passes;
      no live retry of this additional repair yet.
- [x] Make current-view object grounding callable by the CHAT coordinator using
      existing RGB-D admission, without requiring a frontier approach first.
      Keep unverified search references distinct from object geometry; preserve
      fresh manipulation authorization. `ground_object` now uses the existing
      capture/admission path and returns geometry/provenance or explicit failure;
      opt-in find profile enables verified instance admission. The 140-test
      focused pack passes, including real CHAT/HTTP continuation checks.
- [x] Qualify the new grounding call on recorded RGB-D. Initial optional
      description was ignored by the VLM's intentional question-context rule;
      a single complete object phrase now reaches both inference prompts.
      Foreground stool mask is manually verified, 20,454 depth points admitted;
      cropped microwave abstains. No robot actions or GT masks/poses. Report:
      `docs/experiments/coordinator_grounding_20261116.md`.
- [x] Retry the bounded operator case with ground_object exposed. All three
      groundings abstain; the target is visually identified but no XYZ admitted.
      A clearer coordinator image is replaced by a cropped grounding capture.
      Active list stays 32 across all twelve rounds (34 registered; task-plan
      pair provider-gated), up one from the prior live session. Report:
      `docs/experiments/agent_tool_surface_20261117.md`.
- [x] Preserve the exact decision RGB-D snapshot, calibration and scope for
      grounding within the existing capability. Return source-image association;
      keep acquisition timing and manipulation reacquisition boundaries explicit.
      Clear decision pixels survive a newer clipped capture in CHAT/HTTP tests;
      live RGB/display/depth association passes 2/2. Foreground-stool positive
      control admits 18,082 points with a manually checked mask; microwave still
      abstains on a clipped view. Fix live-discovered immutable-frame metadata
      error; 187 focused tests pass, including scope-reset rejection. Report:
      `docs/experiments/decision_view_handoff_20261117.md`.
- [x] Qualify general measured task gaze without adding a tool: optional
      pan/tilt on CHAT describe_scene and EQA look_around share floor's measured
      head primitive. Live out-of-range rejection and unclipped microwave view
      pass; Qwen recognizes the target, but depth-range admission still rejects.
      Floor recovery clears checked-pose unknown cells and restores task gaze;
      replanned approach reports 0.230 m actual travel. Longer route rejects on
      occupancy. Fix the diagnostic's local/world clearance-coordinate bug
      separately; 659 focused tests pass. Parent-operated, not independent or
      autonomous Qwen task success. `docs/experiments/task_gaze_qualification_20261117.md`.
- [ ] Run small paired EQA/simple OVMM gates on the new task-gaze revision.
      Audit the occupied cell from the task-gaze pilot and qualify an alternative
      closer-view route without weakening clearance or admitting far proposals.
      Microwave still has no admitted XYZ. No broad sweep.
- [ ] Consolidate CHAT overlaps before further tool expansion: query_memory /
      query_scene_graph first, then head-photo delivery variants. Keep storage,
      wrist authorization and historical-vs-current evidence semantics intact;
      use existing registry/capability machinery. No task-specific tool pruning
      or new routing framework. General task gaze should extend/consolidate the
      inspection interface rather than add separate directional recovery tools.
- [ ] Expose floor-support coverage alongside obstacle clearance; targeted floor
      inspection should consider reachable visible unknown cells rather than
      stopping on the first behind-camera candidate. Preserve drop/step safety.

- [x] Audit the frozen history failure through real preprocessing and controlled
      presentation tests. Both frames reach local preprocessing (575 tokens),
      but both backends abstain on earlier-turn pixels and read them in one
      current message. Recall restores historical color; local int4 then
      confuses past/current presence while Caliban does not. Neutral
      acknowledgement does not fix history. Twenty-two completed diagnostic
      calls; remote precision remains unverified; no runtime policy flip:
      `docs/experiments/history_context_audit_20261116.md`.
- [ ] Compare a bounded current-decision evidence bundle with images attached
      only in earlier turns, preserving original dialogue and truthful capture
      order/roles. Qualify historical positives AND current-view negatives
      before paired EQA/OVMM checks. Use existing view-selection machinery;
      do not force model answers or add another recovery policy.

- [ ] Simplify the active agent before adding recovery policies: separate
      last-observed find geometry from current physical-action authorization,
      remove duplicate advertised tools and task-irrelevant state instructions,
      then compare fixed contexts with local int4/Caliban using matched decoding.
      Caliban HTTP is healthy; live dtype/revision remain unverified and the
      current paired driver deliberately strips endpoint overrides.
      Audit: `docs/experiments/agent_complexity_review_20261116.md`.
      First slice now separates find coordinates from present-view verification,
      preserves invalidation and fresh manipulation checks, removes the advertised
      navigate alias in model-final, and trims find-only memory instructions.
      The serial room driver now supports explicit remote model endpoints with
      unverified precision/decoding labeled honestly. CPU gate: 386 pass, two
      simulator opt-ins skip; live qualification and fixed-packet model calls
      remain separate. Staged test plan and budgets:
      `docs/plans/2026-11-16_harness_simplification_and_model_eval.md`.
      First live simplified check completes without OOM: 1/2 phases; bowl ends
      in four rounds, microwave still lacks geometry after eleven. All 15
      executions match model choices and 30 assistant/tool messages stay exact.
      No stable performance gain is established. Report:
      `docs/experiments/agent_simplification_20261116.md`.

- [x] Connect the sensor-only active GraphEQA baseline. Supplied-model inference,
      learned native graph and observed-space navigation run together; a real
      model-selected q6 frontier reaches a fresh assessment. Bounded retries,
      interrupted-decision traces, answer history and callable frontier projection
      are repaired; 46 fork tests, 25 emet tests and a native 15-packet replay pass.
      Task failures remain baseline outcomes, not blockers for this connection.
      The final two-case check has no successful answers; preserve all truncations,
      wrong answers and failed rows without overrides or further model tuning.
      Report: `docs/experiments/grapheqa_decoding_control_20261101.md`.
- [x] Run the first frozen paired answer-path pilot: images-only, external
      GraphEQA and production model-final answering on an empty graph receive
      identical PNG packets and matched total output allowances, including the
      keyword call. All 36 planned trials complete inference; one GraphEQA reply
      fails formatting. Correct counts are 0/12, 1/12 and 1/12 on four correlated
      development snapshots. Mean answer-path times are 2.44 / 11.97 / 3.61 s;
      these exclude collection/navigation and are not full-agent results. Packet
      audit, question/view/path panels, source pins and limits:
      `docs/experiments/frozen_eqa_comparison_20261103.md`.
- [x] Complete the first bounded sensor-only active EQA pair: q2/q6/q12 at
      `9ff8521f`, all six episodes finish, both agents 0/3. Shared estimated-pose
      body, navigation geometry and motion budgets are wired; sampling is isolated
      from geometry updates. No task-quality improvement is established. Manual
      image audit shows inadequate target context; emet translates 2.46–3.27 m
      while native GraphEQA requests no forward motion. Preserve the 118 versus
      nine model-call imbalance: this is a whole-system diagnostic, not an
      isolated graph ablation. Report/figures:
      `docs/experiments/active_sensor_agent_pilot_20261106.md`.
- [ ] Finish small S0/Molmo find qualification after the common Stretch clock repair.
      Old main timed out at ~0.03x under the loaded one-core protocol. Cache the
      native compiled-model hash per model; 34 targeted tests pass and the idle
      full bridge improves 0.166x -> 0.605x on four safe physical cores. Use
      main+shared repair versus candidate with identical allocation, preserve
      failed rows, and measure loaded RTF before interpreting object search.
      Repaired S0 completes both arms: cylinder found, destination missing
      (1/2 phases each; 573/412 s; no superiority established). Molmo initially
      fails wrapper discovery; retry through the installed Python 3.11 override
      after cached scene merge preflight. Molmo retry completes both arms, bowl
      found and microwave missing (1/2 phases each; 173/143 s). Both arms are
      0/2 complete find tasks across S0/Molmo; task acceptance remains red.
      Neither EQA nor find-only results qualify pick/place.
- [ ] Diagnose S0 candidate waypoint timeout (sequence 27) at healthy ~0.58x
      simulator rate; preserve measured terminal pose in failed receipts before
      attributing the failure to control versus contact. Separately test the
      exploration arrival head policy: hardcoded level pitch produces empty
      sky/floor in the table scene, losing a previously visible target. Retain
      image/command context and avoid task-specific pitch rules.
- [ ] Follow the diagnostic EQA cohort with broader target-covering trajectories,
      matched-history image-only controls and predeclared task budgets. Retain
      model-final answers, raw context and failed actions; do not tune the
      reference baseline until it succeeds. Stronger-model comparisons remain
      optional ablations, not prerequisites for accepting the connection.
- [ ] Recover the visible Molmo microwave with a safe closer view. Cached masks
      cover the coffee maker; Qwen-box/SAM2 finds the microwave but its median
      depth is 3.34 m, beyond the unchanged 2.5 m support limit. Range failures
      now remain distinct from object absence. The alternative provider loses
      the bowl control 0/3 versus 3/3 cached; retain multiple opt-in providers,
      preserve semantic rejections and avoid a global provider/depth default flip.
      Offline diagnostic: 24/24 attempts complete; no task-quality gain claimed.
      Coordinator-visible recovery now preserves task gaze, source-bound image
      proposal bearings, nested geometry feedback and camera observations across
      CHAT tool rounds; 268 contract/CPU geometry checks pass. The central model
      chooses sensing, movement and stopping in model-final mode: legacy
      automatic stops, navigation redirects and inference-failure action
      substitutions are disabled. Explicit approach indices and dialogue survive.
      Failed investigation now returns fresh current-view evidence alongside the
      unchanged navigation failure. Strict live check restores bowl admission
      (1/2 phases); all 10 executions match model selections. The microwave is
      visible at 3.304 m after all three nav steps are spent, so the model's
      correct semantic location still lacks admitted geometry. An earlier 2/2
      pass used legacy post-error navigation and is retained separately.
      Freeze the strict candidate for a broader matched EQA/find pilot; any
      larger horizon must apply to both arms. No system superiority is claimed.
      Plan: `docs/experiments/closer_view_feedback_20261113.md`.
- [x] Run the frozen recovery-budget check: new candidate at 6/3, then baseline
      and candidate at 12/8 rounds/navigation calls per find phase. Driver and
      opt-in acceptance profile now expose/use 12/8; independent navigation
      exhaustion is recorded, and relative motion uses the shared counter.
      Explicit answer commits, bounded HTTP CHAT history and cached-camera
      labeling are repaired/covered. Preserve a new low-budget control because
      answer semantics changed too. All three episodes complete: candidate 1/2
      at 6/3, baseline and candidate both 2/2 at 12/8. Candidate query time
      165 s versus 332 s baseline in this one stochastic episode, not a stable
      speedup claim. Candidate 13/13 actions match model choices; baseline uses
      two legacy redirects. Microwave geometry is admitted at 1.646 m median
      proposal depth versus a 3.476 m rejection in the low control. The bowl
      phase changes the next phase's start pose; deliberate recovery from the
      range receipt and causal budget benefit are not established. Approved
      ten-file deduplication freed 5.24 GiB; the run declared a monitored 16 GiB
      floor. Nov 15 cleanup reclaimed 10.40 GiB of disposable download/lint
      caches, restoring roughly 20 GiB free; recheck before further live tests.
      User-approved raw-array archival subsequently reclaimed 23.76 GiB, leaving
      about 40 GiB free; all 214 batches verified and visual/metric files retained.
      Restore/index: `docs/experiments/debug_array_archival_20261115.md`.
      Review/protocol: `docs/experiments/recovery_budget_review_20261114.md`.
- [ ] Repeat matched 12/8 find checks and small EQA/another-room acceptance;
      preserve failures and compare then-current main before scaling. Add a
      shared-start out-of-range diagnostic to distinguish model-selected closer
      views from incidental gains caused by a preceding task's trajectory.
      Do not add answer overrides or automatic high-level action substitutions.
- [ ] Fix multimodal packet growth before another long-horizon repeat: seed-1
      baseline passes 2/2; candidate bowl terminal localization fails after an
      earlier admission, then destination inference raises CUDA OOM. Last input
      contains 28 historical images (15 unique pixel hashes) plus two current
      images, despite staying within 50 messages. Preserve dialogue/action text;
      deduplicate/budget image evidence with auditable identities. Keep retained
      admitted object evidence distinct from latest-view grounding failure.
      Incomplete pair: `~/runs/emet/recovery-budget-high-20261115-r2/`.
      Episode image packets now preserve dialogue, bound attachments to eight,
      expose stable pixel IDs and support explicit historical recall. Offline replay
      reduces the failing 30 attachments to eight. Live seed-1 retest completes
      without OOM, 1/2 phases; all 15 requests stay within eight images. The
      coordinator uses recall_view but still submits a semantic microwave answer
      without geometry. Find now gets an explicit localization task header;
      semantic EQA keeps its question header. Header-only live check remains
      1/2 without OOM. Removed a contradictory legacy routing-priority sentence
      from the actual tool description; 436 CPU checks pass. Task improvement,
      broader task acceptance and retained-object selection remain pending.
      The third image-only check still raises OOM with eight images: long state
      text also matters. Added a 64k working-character cap that archives only old
      state text by explicit reference; current state and all assistant/tool text
      remain exact. read_history retrieves original text and recall_image its
      pixels. The resource probe completes at 22,773 tokens with a synthetic
      6 GiB competing GPU allocation; live also completes without OOM (1/2).
      Fifteen requests respect both bounds, with assistant/tool text unchanged.
      Fixed a discovered pixel/frame ID naming collision: current tools are
      recall_image(image_id) and read_history(message_id). 442 CPU checks pass;
      current-API live check completes without OOM, 0/2 terminal phases. All 16
      requests respect bounds and preserve 32 assistant/tool messages. Pixel recall
      uses the correct ID. Bowl geometry remains admitted at submit, but the
      terminal adapter rejects it without current visual verification; context
      now states both existing requirements. Investigate last-observed find
      geometry versus current manipulation authorization as separate contracts.
      Broader acceptance remains pending.
      Examples: `docs/experiments/multimodal_context_20261115.md`.
- [ ] Emit actual per-episode simulation duration, model inference cost and
      invocation overhead in paper-run manifests. Clock probes and reported
      episode wall metrics alone do not measure complete task simulation time.
- [ ] Record effective generation config, RNG/seed policy and environment versions
      in paper-run manifests. Qwen's checkpoint defaults enable sampling at 0.7;
      an unseeded one-call contrast does not establish a stable prompt effect.
      The supplied HTTP server correctly rejects unsupported request seed settings;
      diagnostic per-call RNG control is not an API capability or cross-GPU guarantee.

- [x] Fork GraphEQA for supplied model endpoints and paired frozen-context
      evaluation: https://github.com/cpaxton/graph_eqa/pull/1. Planner/room API
      routing, instance IDs, scorer isolation and complete request traces are
      tested; the real Qwen8B int4 current-image/graph pair returns valid replies.
      The active connection is complete; matched evaluations remain separate work.
- [ ] Extract the model-server contract repairs into a focused PR against current
      main: loaded-model identity, complete dialogue, unsupported-setting errors,
      decode receipts and declared native/text tool capabilities. They are tested
      on `feat/harness-evidence-study`; do not treat main's older server as qualified.

- [ ] Qualify a genuine model-only baseline: identical frozen RGB trajectories,
      current/uniform/retrieved/selected context, no graph or interactive tools,
      matched image/token budgets and one frozen backbone. Then run a minimal
      active-tool agent and semantic-voxel/graph/tool ablations on the EQA/OVMM
      pilot. Preserve the common physical executor and collision checks. Plan:
      `docs/experiments/shared_agent_literature_review_20261029.md`.
      A live current-image vs reference-graph API pair now completes on Qwen8B
      int4 (`cede090f`, fork `21b6876`); both answers are uncertain. This is one
      frozen-context wiring check, not matched-history or active acceptance.
- [ ] Audit runnable 3D-Mem and zero-shot ToolEQA baselines; separately qualify
      a GraphEQA controller adaptation and an upstream reproduction. Review
      NavMCP's trajectory-evidence/interface comparison and release dependencies.
      Do not relabel internal `static_graph` or a reduced tool list as a faithful
      upstream system; keep models, segmentation inputs and task populations clear.
      GraphEQA's cached Habitat container passes native learned RGB-D graph
      construction on 2026-10-30 (five fresh views, 16 object-layer nodes).
      Repeated updates and the reference agent-state handoff also pass at
      `476e6c81` (ten fresh views, 20 object nodes and two agent nodes; no model
      requests). The model API fork and paired context runner are available at
      https://github.com/cpaxton/graph_eqa/pull/1; live model/context routing passes.
      The sensor-only pose and measured execution gates now pass in two short
      scene checks; match the same frame basis/body/budgets on the emet side and
      resolve model planning before one complete paired active episode. Report:
      `docs/experiments/grapheqa_model_api_connection_20261030.md`.
      Native warnings remain; the April-2025 image is a dependency base, not
      current-source parity.
- [ ] Qualify GraphEQA with **no privileged scene inputs**: freshly inferred
      segmentation, explicit episode IDs, no annotation enrichment, semantic
      masks/IDs, global scene bounds or navmesh route oracle. First run the
      isolated learned RGB-D -> native Hydra probe (passed at `55ab995e`); then match observed-map
      navigation and pose estimation across the active agents. Historical
      semantics-off rows still have navmesh assistance. Protocol and command:
      `docs/experiments/grapheqa_sensor_only_qualification_20261030.md`.

- [ ] Resolve refresh execution failures before another broad window: rerun the
      five DynaMem missing-viewpoint exceptions on `6ae052fc`; qualify Molmo's
      declared assisted/physical mode and fail fast on dead physics; isolate S0
      tracking and native TAMP release clock/geometry conditions. Preserve the
      original scores and manifests. Diagnosis:
      `docs/experiments/benchmark_failure_audit_20261029.md`.
      Repaired live native two-object TAMP now passes (both goals, zero forbidden
      commands). All five EQA retries now complete without crashes (2/5 correct),
      after offline package-cache pruning restored the disk reserve. This closes
      the identified crash recovery; Molmo and integrated S0 remain open.
- [ ] Qualify all 12 controlled development interventions through the shared
      agent. Full Stretch fixtures and production CHAT bindings now exist; live
      action/evidence audit, learned localization and manipulation are not yet
      accepted. The remote CHAT image keyword and per-user reset bugs have focused
      regression fixes; retain the old and corrected live diagnostic artifacts.
      Qualify official frame packs
      and evaluators for EXPRESS/AbstainEQA. Protocol/results:
      `docs/experiments/harness_capability_benchmark.md`.
- [ ] Resolve cold-map turn feasibility in the new live case: both corrected
      seen/unseen protocols finish and preserve camera/history, but target-directed
      floor inspection returns `floor_target_unobservable` and yaw does not change.
      Compare scripted broad downward inspection with targeted inspection before
      changing planner clearance. Color correctness is not recall/navigation
      acceptance. Audit: `paper/data/harness_live_diagnostic_20261029/`.
- [ ] Complete the frozen-main existing-benchmark refresh across its declared
      237 tasks; preserve errors/pending cases and resume across idle windows.
      Gather source-linked figure inputs, audit selected success/failure traces,
      and assess task quality/cost separately from tooling correctness.
      `docs/experiments/benchmark_refresh_20261029.md`; no RL work is preempted.

- [x] Land the consolidated [merged TAMP review corrections](docs/plans/2026-11-16_tamp_stack_review.md)
      on `fix/tamp-stack-review`, then integrate them into the observed-admission
      follow-up. Original #226/#231/#234/#242/#243/#244 are merged; the new
      review findings landed in [PR #246](https://github.com/cpaxton/home_robot_v3/pull/246)
      as `2ccf390c`, without rewriting historical runs. Integrated and verified
      on the observed-admission branch; its live transfer gate remains open.

- [ ] Profile failed-search cost without answer overrides or target-specific rules:
      final S0 seed 1 misses the receptacle in both arms but candidate takes
      798.6 s versus control 287.2 s. Seed 2 succeeds faster (478.0 vs 790.4 s).
      Evidence: `~/runs/emet/final-stack-review-20261028/`; no consistent speedup claim.
- [ ] Predeclare and validate a larger-object RoboCasa basic acceptance case.
      Seed-0 `robocasa_pp_s1_v2` samples a 2.8 cm sugar cube and is excluded from
      basic gates at user request; retain it in `configs/ovmm/stress/small_object_episodes.yaml`.
      Reduced integration acceptance does not qualify RoboCasa; preserve historical failures.
- [ ] Run the repaired relative-motion → targeted-floor handoff in the Molmo
      pilot, then S0/EQA regression gates. Catalogue: `docs/testing/agent_observation_cases.md`.
- [ ] Author isolated live visible-far, unknown-floor, occupied-route and
      absent-target variants; synthetic contracts are not model/sensor acceptance.
- [ ] Integrate EXPRESS-Bench and AbstainEQA after these gates; pin official
      data/evaluators and validate tiny pilots first. Requirements:
      `docs/environments/embodied_qa_extensions.md`.

Short checklist for agent/hardware polish that is not worth a full plan doc yet.
Strike through or move to a PR when done.

- [ ] Deferred coverage experiment (#198): preserve `feat/measured-coverage-scan`
      at `cb481970`; reopen a small main-based slice only for a demonstrated view
      coverage gap, with same-provider local/coverage, initial-heading and cost
      comparisons. See [disposition](docs/plans/coverage_scan_disposition.md).

## Native TAMP base approach (follow-up to #234)

- [x] Implement checked unloaded base positioning on
      [`feat/tamp-native-base-approach`](https://github.com/cpaxton/home_robot_v3/tree/feat/tamp-native-base-approach),
      [PR #242](https://github.com/cpaxton/home_robot_v3/pull/242),
      stacked on `fix/tamp-native-joint-margin` (#234). Keep it out of #234's
      frozen refinement scope. Plan: [native base approach](docs/plans/2026-11-14_native_tamp_base_approach.md).
      Use private combined base/arm planning, swept robot geometry, ordinary base
      controllers, measured arrival and fresh grasp refinement. No live pose writes.
- [x] Finish the original control/shift/swap gate, the complete eight-task native
      replay and four actual remote-VLM agent cases. November 16: pilot 3/3 tasks,
      full native 8/8 tasks and 18/18 transfers, boundaries 4/4; actual agent 4/4
      tasks and 10/10 transfers. Preserve #234's failed swap and historical native
      4/8 / Caliban 3/4. These are separate development cohorts. Evidence:
      [base approach results](docs/experiments/tamp_native_base_approach_20261116.md).
- [x] Complete independent reviews and integration gates for the opt-in stack
      (#226, #231, #234, #242, #243, #244), landing in dependency order. Observed
      task execution, exploration, household adapters and hardware remain separate gates.
      [Current merge/review plan](docs/plans/2026-11-16_tamp_recovery_dialogue.md#review-and-merge-status-november-16):
      first five foundations are merged; #244 completes the reviewed dialogue
      layer. Updated-main checks pass, including shifted native/agent swaps and
      the actual-agent partial stop/contextual retry pair. Review agents find no
      production blocker; historical populations and failed attempts remain separate.
- [x] Freeze new untouched layouts and native scripted recovery trials on
      `feat/tamp-prospective-evaluation`: 4/4 new native tasks and 4/4 actual-agent
      tasks, 10/10 transfers each; 2/2 scripted recovery tasks, 5/5 boundary checks;
      416 affected tests pass. [Results](docs/experiments/tamp_prospective_20261116.md).
- [x] Exercise between-transfer partial failure through the actual agent: named
      verified progress, no-command step-2 refusal, stop, separate authorization,
      fresh one-transfer recovery. Preserve the initial discovery-filter failure.
      Review: [PR #243](https://github.com/cpaxton/home_robot_v3/pull/243).
      [Feedback/recovery results](docs/experiments/tamp_partial_agent_20261116.md).
- [x] Finish bounded contextual recovery on `feat/tamp-recovery-dialogue`:
      [PR #244](https://github.com/cpaxton/home_robot_v3/pull/244), integrated with landed #243.
      [plan and Agentic GenTAMP baseline implications](docs/plans/2026-11-16_tamp_recovery_dialogue.md).
      Freeze source, test a short retry through the actual CLI and repeat stop-only;
      retain fresh discovery, consumed-plan and physical-effect checks.
      Corrected source `ebbdb1d9`: stop-only passes; short retry completes only the
      remaining transfer and both original goals hold; 485 tests pass. Preserve
      the first invented-handle failure. [Results](docs/experiments/tamp_contextual_recovery_20261116.md).
      Final updated-main replay: 230 integration tests pass; stop-only and short
      contextual retry both pass their distinct gates, with 794 audited commands
      and none forbidden. [Integration record](docs/experiments/pr244_main_integration_20261116.md).
- [ ] Qualify observed task admission and exploration. The completed recovery
      fault is injected planning admission, not grasp loss, unknown dispatch or
      held-object recovery. Short retry now passes one familiar diagnostic;
      broad reliability remains unqualified. The prior restated-goal trial also
      makes extra read-only checks after completion.

## Shared evidence/navigation stack (#199 / #202)

- [ ] Report wall-clock and simulator-clock elapsed time separately for episodes,
      mapping, navigation and manipulation; publish actual sim time with boot/clock
      identity, handle resets/missing clocks explicitly, and retain real-time factor.
      Never estimate task sim duration by multiplying wall time by a sampled ratio.
      Separate successful motion, settling, planning, recovery and VLM compute costs.
- [ ] After correctness/clock reporting, evaluate mjlab/mjenv-style parallel MuJoCo
      execution as a separate infrastructure study (confirm exact projects/APIs).
      Compare physics/contact and robot-model parity, RGB-D rendering, per-env
      state/clock isolation, deterministic seeds, memory/throughput and failure
      reproducibility. Keep the same agent API; no simulator migration or changed
      task dynamics before matched acceptance. Current heavy experiments stay serial.
- [ ] Ablate episode history after the dialogue-contract repair: full multimodal
      dialogue vs tool-results-only vs stateless inference, with identical model,
      seeds, prompts, acquisition/answer budgets and view selection. Compare paired
      EQA and small OVMM outcomes, repeated/no-op tool calls, context tokens,
      latency and memory; inspect changed answers and preserve raw request evidence.
      Keep unsupported-argument rejection enabled in every arm.
- [x] Validate main integration: frozen `0fb37763` EQA diagnostic 1/4, unchanged
      from parent; no infrastructure errors. Not restored EQA acceptance.
- [x] Fix reviewed SAM2 PIL input, lost sweep diagnostics, and invalid-depth
      obstruction; retain unknown measured clearance for planning fallbacks.
- [ ] Clear anchor-less investigate, precision waypoint policy, map-edge coverage,
      and exact-model wheel/head checks before claiming stack acceptance.
- [ ] Integrate timestamped shared view history and actual pre-turn capture;
      compare selectors on fixed histories, then EQA/small OVMM.
- [ ] Audit assessment/final-answer disagreements separately from view selection;
      do not tune answer precedence to q14. See
      [consolidated evidence and review](docs/plans/eqa_context_evidence.md).
## RL controller integration and acceptance (Mars first)

Accepted plan: [Mars controller integration](docs/plans/mars_controller_integration.md).

### Current WBC follow-through (2026-11-15)

Use the [complete setup/status](docs/rl/status_20261114.md) and
[WBC philosophy](docs/rl/whole_body_control.md) alongside the ordered work below.
Historical entries further down retain their original populations and outcomes;
they are not evidence that the remaining stages have passed. Research-code
merges, measured policy improvement and hardware readiness are separate decisions.

- [x] Merge the core WBC implementation/evidence: PRs #224, #227, #232 and #233;
      close superseded #214 without restoring its older code. Preserve the
      stronger source and negative results. Repair serialized paired settings,
      physical-tick reward discounting and duplicate-trial confidence reporting.
- [x] **Finish the corrected learning baseline.** The
      [C1 comparison is complete](docs/experiments/mars_c1_normalization_recovery_20261115.md):
      preserve the original 4,096-decision control and failed 256 unoptimized
      candidate ticks; recover only the remaining 4,096 decisions. Fresh
      regressions pass 1,249/1,249; explicit source/teacher/optimizer equivalence
      and every initial/first-rollout hash pass. All twelve fixed-final outcomes
      and diagnostics are recorded. Shared full mastery is 2/12 versus per-head
      0/12 and initial 2/12; completed arm endpoints meeting 3mm/0.02rad improve
      1/10 to 5/10. This completes the experiment, not whole-body qualification:
      base, quiet-hold and jerk failures remain, with no checkpoint promotion.
- [x] **Separate C1 base interference from arm learning.** All 24 unchanged
      initial/final replays reproduce every continuous metric and outcome.
      Replacing only final base requests with initial zero requests raises
      arm full passes 1/10 to 3/10 and smoothness 1/10 to 9/10, but worsens
      full-amplitude pitch orientation. Preserve the
      [36-rollout diagnostic](docs/experiments/mars_c1_zero_base_diagnostic_20261115.md);
      no training, exported controller or qualification follows from it.
- [x] **Complete the frozen-base/context coordination test.** The new
      [bounded pair is complete](docs/experiments/mars_frozen_base_arm_20261115.md):
      seed90, 16,384 transitions each; baseline arm full 59→57/64 and velocity
      60→61/64, both 64/64 tracking. All 488 expanded outcomes, 244 matched
      initial/final identities and serialized protected-function checks are
      recorded. Each teacher passes 122/122 on the source gated by 1,249 tests.
      Base-only metrics are exactly preserved; coupled remains 2/8. Keep the
      retained source: one positive development seed does not qualify a new
      controller. Preserve the old zero-step queue failure and add no automatic
      retries, extra seeds or checkpoint selection.
      [Bounded follow-through protocol](docs/plans/mars_frozen_base_arm_followthrough.md).
- [x] **Test reference-conditioned base exploration.** Fresh seed90 shared
      GRU pair completes 4,096 decisions each on source gated by 1,260 tests.
      Candidate arm tracking/settling improves 1→7/10 and fixed-base stops
      5→0, but full arm remains 1/10 with nine linear-jerk failures. All twelve
      fixed cases and identical initial anchors are retained; default control
      exactly reproduces earlier C1 metrics, rollout hashes and final weights.
      [Complete report](docs/experiments/mars_reference_base_exploration_20261115.md).
      Keep default scale 1.0 and the retained production source; no promotion.
- [ ] **Resolve remaining coordination and quiet-hold failures.** Use the
      [ordered next-steps plan](docs/plans/mars_wbc_next_steps.md): isolate the
      low-level base feedback/oscillation first, validate current-velocity
      feedback through reset/motion/stop, then train missing directions on the
      corrected plant. Matched current-COM replay already improves frozen
      candidate full success 2/12→8/12 with no PPO or gate change. The completed
      [actual-reset validation](docs/experiments/mars_current_feedback_reset_validation_20261115.md)
      confirms 24 stops and 1,299 regressions; legacy reaches 151/164 general and
      12/12 common cases, but 0/6 TAMP versus teacher 5/6. The
      [four-episode trace study](docs/experiments/mars_tamp_input_diagnostic_20261115.md)
      confirms input delivery and small learned commands, followed by clipped
      open-jaw feedback. Preserve real inputs; train the actual near-contact
      distribution mixed with general tracking after fresh checks/admission.
      The teacher's support-margin rejection now has an actionable public
      failure code; independently admit the measured-state replan without
      weakening the support check. Settling/velocity/GRU frozen probes
      did not close this gap. Keep PR #240's 17 training and 24 audit cases
      distinct from admitted/trained evidence. The conditional curriculum
      pair remains bounded at 16,384 decisions per variant after fresh
      source/teacher gates; preserve reward/representation and report exact
      per-case exposure. Measurement/reward changes are conditional if still
      needed, not simultaneous knobs. Preserve original scores and independent
      stop/force/support/retention checks. Freeze the separate audit cases before
      training; no automatic budget extension or policy promotion. Existing
      offline and native diagnostics do not relabel any candidate failure.
      The opt-in axis-angle feature still needs a separate controlled trained
      comparison; do not confound it with the plant, reward or velocity studies.
- [x] **Repair drift after Cartesian hold entry.** Bounded nonfault reacquisition
      preserves goals, history, grip authority and the original deadline. On
      the explicitly declared 4 mm / 0.04 rad task contract, the retained actor
      now completes grasp pose and executor closure; lift planning rejects
      insufficient support reserve. Keep the original precision failures and
      all [twelve task diagnostics](docs/experiments/mars_task_arrival_20261115.md).
- [ ] **Train actual grasp-pose and fixed-world-hand/base motion.** Extend the
      pregrasp-only primitive boundary, obtain fresh stage-specific teacher
      admission, and mix these reachable references with general tracking.
      Do not increase timeouts or accept the actor's 2.15 mm advance as a
      completed 20 mm support-preparation goal. Keep loaded retention/grasp-fit
      repair separate, with original force/support/retention thresholds.
- [x] Prepare the exact [17-case directional curriculum and 24-case audit](docs/plans/mars_directional_curriculum.md)
      with a strict separate loader, source/control hashes, unchanged contracts
      and disjoint audit resets/motions. This is case preparation only: no
      native admission, training or policy promotion. Keep actual task-stage
      coverage and the latest #239 diagnosis in the next training decision.
- [ ] Admit the prepared directional/audit cases on the corrected plant for
      the selected training stage. Bind the new source's own checks and teacher
      evidence; keep original case rejections visible. Loaded retention failures
      do not qualify loaded training or invalidate admitted unloaded references.
- [ ] **Advance real contact skills in stages.** Follow the
      [safety/task curriculum](docs/plans/2026-11-13_whole_body_safety_curriculum.md)
      from actual pregrasp to closure/lift, loaded transport/braking, and
      placement/release. Freeze each stage's cases, observations, teacher
      admission and budgets before training. Retain independent retention,
      force, wheel-support and measured-stop checks. Supported open-jaw
      tracking and pregrasp do not qualify loaded manipulation. Evaluate
      complete public-tool programs and fresh audit cases before broader claims.
- [ ] **Evaluate a bounded differentiable world-model backend.** Use
      SuperMimic as a method reference, not a claimed Mars result. First capture
      per-tick physical, actuator/servo and executor state, executed actions,
      reference progress and termination semantics. Split complete trajectories
      and case families into training/holdouts; do not leak adjacent windows.
      Measure multi-step errors by horizon and by contact, slip, wheel unloading
      and retention event. Keep privileged model state outside actor inputs.
      Validate trained actors in unchanged native physics, and compare total
      collection, fitting, policy-update and evaluation wall time with corrected
      PPO. Predicted success is not qualification; published flat-ground motion
      tracking speedups are not Mars manipulation speedups.
- [ ] **Establish robustness and transfer.** Identify actuator/sensor timing
      and dynamics, then test declared noise, delay, payload/friction variation,
      disturbances and held-out tasks. Require complete native/alternative-host
      parity before scaling; Caliban's frozen-fixture gate remains open.
      State-of-the-art claims require matched strong baselines and measured
      capability/compute comparisons. Hardware needs verified feedback, limits,
      watchdog and stop behavior plus separately authorized testing.

Olympia rules remain one CPU14 training/preflight worker and at most one CPU15
test worker, one numerical thread each, CUDA hidden and ordinary rendering off.
Use explicit host-visible handoffs and immutable, managed run records. Do not
leave an expired queue described as running or count software-only PPO checks
as an experimental training result.

### Historical WBC evidence and remaining integration work

- [ ] Complete the bounded [hand-accuracy pilot](docs/plans/mars_hand_accuracy.md):
      teacher baseline 66/66, latest base-trained actor 0/8 arm cases with near-zero
      arm requests. Use stage-balanced arm/base resets, smaller reachable arm
      amplitudes and native hand-gated base credit, preserving canonical gates.
      Measure requested versus actual hand/joint motion before adding loaded
      hold/lift and replaying the public learned-controller pick/place programs.
- [x] Clear finite empty-hand arm learning with the [five-output component diagnostic](docs/plans/mars_arm_component.md).
      The completed local 65,536-step mixed fallback remains 0/8 arm cases; retain
      its regressions. Component teacher passes16/16 with wheel/grip requests
      fixed; one32k PPO run passes8/8 at all checkpoints after8k and16/16 on final
      expanded/reloaded evaluation. This is not whole-body or loaded qualification.
- [x] Verify [arm-to-whole-body mean-preserving training transfer](docs/plans/mars_arm_to_whole_body.md):
      exact arm/critic predictions and16/16 physical arm passes before training.
- [ ] Retain full arm smoothness while reintroducing base motion, then loaded hold/lift and public
      pick/place. Keep one production actor; no permanent multi-task architecture
      or teacher/student loss is needed by the current component evidence.
- [x] Correct mixed-training exposure using actual control-step budgets. The
      [first whole-body pilot](docs/experiments/mars_arm_to_whole_body_20261031.md)
      retains16/16 arm tracking but8/16 full passes, with base38/50 tracking and
      33/50 full passes. Forward/reverse receive only389/32,768 steps (1.19%);
      test deterministic budget accounting before another bounded comparison.
- [x] Remove executor-ignored requests from policy sampling/means/likelihood/
      entropy and keep known wheel/grip history in actor units. The completed
      [repair pilot](docs/experiments/mars_command_history_20261031.md) allocates
      6,368/32,768 steps (19.43%) to translation. Arm tracking remains16/16,
      but full arm/base passes are4/16 and32/50; no policy promotion.
- [x] Correct simulator forward/yaw diagnostics from principal-inertia axes to
      regular body axes. Verify rotated inertia frames, identical66-case physical
      metrics/rewards after replay, and constant-request velocity response.
- [x] Run the bounded [frozen-arm base diagnostic](docs/experiments/mars_base_isolation_20261101.md)
      and sustained-request return audit. One 32k gSDE pilot improves expanded
      base tracking to48/50 but only15/50 full passes; arm14/16 tracking and0/16
      full. Keep every failure and retain the unpromoted source bundle. Removing
      direct hand-to-base connections in a final-only diagnostic restores
      canonical arm tracking8/8, but not smoothness; do not deploy that mask.
- [x] Complete the paired [base curriculum and reward comparison](docs/experiments/mars_contract_curriculum_20261103.md):
      1,005 regressions pass and teacher66/66 reproduces exactly. Native-reward
      base-only8k reaches49/50 base tracking,42/50 full base and15/16 full arm;
      aligned reward reaches48/50,41/50 and12/16. Both retain16/16 arm tracking.
      Preserve the stronger native baseline; neither actor is promoted.
- [x] Add tunable force/jerk budget excess costs and a terminal full-contract
      bonus shared with evaluation, with teacher/reset binding and native
      observation/physics preservation checks. Keep the package opt-in.
- [ ] Test a gradual reward transition and quieter refinement exploration:
      the first aligned run removes50 native completion points but pays no
      full-contract bonuses during exploration. Preserve movement credit early,
      repair the remaining base/arm jerk cases and validation translation lag,
      then restore coordinated hand/base cases before loaded/TAMP work. Verify
      exact source-transfer fidelity for partly qualified continuation rather
      than silently bypassing initial admission or relaxing evaluation gates.
- [ ] Port exact frozen TAMP fixture initialization to ARM without relaxing its
      hash/pose checks. Caliban's eight-worker pilot never started: one missing
      camera source is fixed; two exact frozen-fixture checks still reject ARM's
      compiled/reset differences. The current 122-case ARM teacher passes; the
      fresh full gate is 1,077/1,080, with the third Git-context setup failure
      repaired by a focused rerun. An exact binary model/state numeric probe
      passes its listed checks, but full admission/dynamic parity and a portable
      loader remain open. See [evidence](docs/experiments/mars_caliban_portability_20261106.md).
      Keep training local until the complete ARM gate passes.

- [x] Externalize generated control archives/streams into ignored, checksummed
      storage with a retrieval catalog and a regression guard against force-add.
      Preserve original artifacts and branch history before publishing cleanup.
- [x] Wire independent native worker count and per-worker rollout length into
      the mastery trainer, counting telemetry for every worker. Keep this unstable
      workstation serial; use Caliban for bounded parallel probes. The current
      seven-motion trainer now also supports1–8 workers with all-worker telemetry
      and aligned rollout/warmup boundaries. Native two-world parity/PPO tests
      and two/eight spawned-worker ARM plumbing probes pass; all1,095 fresh
      regressions pass. Frozen-fixture portability remains required before scaling
      a trained checkpoint. See [worker evidence](docs/experiments/mars_motion_workers_20261106.md).
- [ ] Admit the GPU backend for PPO: short native/GPU Mars state-parity probes
      pass with no-slip physics intact; still require independent-world resets,
      loaded grasp/lift/placement and per-substep force/wheel parity, then batched
      controller/history/reward logic and end-to-end PPO. See the
      [parallel simulation plan](docs/plans/mars_parallel_simulation.md).

- [x] Address PR 224 runtime review: require all six base velocity DOFs to stop,
      use measured arm feedback for startup/overspeed checks, expose explicit
      simulator timing budgets with a post-inference deadline/freshness check,
      and name/document spatial teacher gains and the task metric. See the
      [review repair report](docs/experiments/pr224_runtime_review_20261024.md).
- [x] Address PR 227's concrete review notes: shared test fixtures compatible
      with importlib collection, scalar critic-unit validation, staged and
      validated component exports, and documented masked likelihood semantics.
      Bind helper sources into regression receipts. See the
      [review follow-up](docs/experiments/pr227_review_20261106.md).

- [x] Generate and execute privileged real-model multi-step development cases
      through public TAMP tools without resets between transfers. Stretch's
      two-object reference and agent pass; Mars delivery/return stops at return
      planning for both selectors. Preserve the failed prefix in the
      [multi-step report](docs/experiments/agent_multistep_control_20261024.md).
- [ ] Fix the frozen Mars post-placement return/regrasp planning case without
      changing its goals or thresholds; budget rejection is not an infeasibility
      label. Then freeze broader multi-step layouts/objects before qualification.
- [x] Add an opt-in fixed-checkpoint TAMP execution backend for Mars simulation,
      with verified robot/actuator/mesh/physics compatibility across primitive
      world fixtures, bound training evidence, actual actor-call counts and
      measured stop on inference/tracking failure. Preserve reference execution
      and the original full-training-model fingerprint check. The real agent
      also exercises discovery/planning/execution with this backend; see the
      [checkpoint handoff report](docs/experiments/mars_tamp_policy_backend_20261025.md).
- [ ] Train and validate on the actual public TAMP command distribution: the
      new explicit Cartesian backend consumes whole FK pose schedules and keeps
      measured configuration/support/retention/base checks. Joint execution
      remains available. Current development programs still fail at pregrasp;
      an executable policy remains open. Keep fixed-base approach
      constraints, force, retention and wheel limits unchanged; a wired backend
      is not a successful learned pick/place controller. Extend the frozen
      development comparisons across layouts, masses and composed tasks.
- [x] Fix checkpoint inference before the first motion command and suppress
      inference during explicit terminal hold. Audit stationary tracking against
      reference/zero control, warm history, categorical normalization and target
      sensitivity. See the [startup audit](docs/experiments/mars_policy_startup_audit_20261025.md).
- [x] Add bounded hold/fixed-current-base curricula, categorical identity
      normalization, all-six-DOF body motion costs and checked training-only
      tracking transfer. Correct near-object hold/closed-empty examples.
- [x] Compare ordinary and learned-feedback motion heads with actual PPO:
      24,576 new steps across five offline W&B runs; archive negative controls,
      teacher comparisons and a fresh 692-test receipt. Ten-second hand holds
      remain within 1.2 mm on five development fixtures. See the
      [hold-repair report](docs/experiments/mars_hold_repair_20261025.md).
- [x] Isolate hold chatter with matched zero/explicit/reference/policy controls
      and fixed-weight wheel-gain ablations. Gain 0.5 gives continuous quiet
      holds on five development pregrasp fixtures; restore the checkpoint's gain
      and retention observation context in compatible UI sessions as well.
- [x] Align optional training settling with all six base velocities; add nearby
      FK orientation/translation coverage, function-preserving angular feature
      scaling and a gripper-close transition during explicit terminal hold.
- [x] Add fixed-base training boundaries matching the 1 cm / 0.02 rad execution
      envelope, tighter heading reward and same-target progress without credit
      for goal changes. Refresh Cartesian executor snapshots every cycle.
- [x] Restore actual Cartesian actor phase/retention/aperture handoff and replace
      the intermediate binary clock gate with continuous lag scaling inside the
      unchanged tracking tube. Archive all three public failures and completed
      65,536-step refinement, with strict angular and quiet diagnostics.
- [x] Complete matched 16k-step goal-gated settling pilots. Both regress from
      the source on moving targets; all public replays still fail. Preserve
      negative results and the recorded-observation normalization/weight audit.
- [ ] Collect independent same-controller simulator challenge outcomes before
      enabling unsupported safety-risk heads. Preserve per-head abstention and
      keep these challenge outcomes separate from promotion qualification.
- [x] Complete four-way 8k-step continuation audit. Fixed normalization and
      smaller updates preserve position better; angular and TAMP gates still
      fail. Archive completed runs and the corrected base-first action audit.
- [x] Compare bounded reachable translation coverage against the same control,
      retaining general base/hand/combined, retarget and disturbance tests plus
      TAMP. Both 16k pilots complete with no task gain; preserve all negative
      results. See the [general tracking plan](docs/plans/general_whole_body_tracking.md).
- [x] Exercise a single-delivery development contract through the real remote
      agent and public tools: reference completes; source learned actor stops.
      Diagnose gain variants, expose per-seed IK versus collision failures and
      add scoped shallow payload-support departure. See the
      [agent/controller report](docs/experiments/mars_agent_controller_20261026.md).
- [ ] Complete the same delivery with the learned controller: grasp-to-lift
      tracking currently trips orientation/base limits. Add bounded loaded
      transitions to training while retaining ordinary tracking; preserve all
      original multi-transfer, force/retention/wheel and general regression gates.
- [x] Add opt-in supported grasp-to-lift resets within carry, preserving 70%
      tracking and the remaining manipulation mix. Require physical quiet closure,
      private path/support admission, pose preview and measured collision stops.
      The first lift subtype uses boxes; other tasks retain mixed objects.
      See the [curriculum plan](docs/plans/mars_grasp_lift_curriculum.md).
- [ ] Fix cylinder load-transfer retention: quiet supported grasps still spin/
      slide during teacher lift; depth changes alone do not fix it. Preserve
      strict retention limits and failed mixed-shape component cases.
- [x] Complete both 16,384-step supported-lift PPO pilots and unchanged public
      delivery/multi-transfer replay. Teacher is 7/8 on identical independent
      lifts; all three actors remain 0/8 and 0/3 public programs. Preserve the
      negative comparison and treatment wheel-unloading failures. See the
      [supported-lift report](docs/experiments/mars_grasp_lift_20261027.md).
- [x] Audit parking incentives and fixed-case learning. Coupling fixed-base credit
      to hand tracking improves the independent component mean to 13.48 mm /
      0.0295 rad with no observed retention/wheel failures, but remains 0/8.
      Preserve every checkpoint, regression and unchanged strict success gate.
- [x] Isolate loaded arm/base failures on matched physics: reference wheels leave
      learned hand error; reference arm leaves learned base stops. Reject naive
      gain increases (1/8 and 6/8 retention losses). Audit world-origin sensitivity
      and prior-tick wrapper reward accounting. See the
      [failure diagnosis](docs/experiments/mars_lift_failure_diagnosis_20261027.md).
- [x] Align retimed preview, reference completion and terminal hold-age clocks;
      preserve paused progress, endpoint packet deduplication, new-path reset
      and existing episode/executor deadlines. Prior fixed-clock metrics remain
      historical; fresh source-bound qualification is required before training.
- [x] Implement/test native same-tick hand-conditioned base credit, coarse-to-fine
      precision and validation-gated later motion regularization. Base-only reward,
      force/retention costs and hard safety limits stay intact.
- [x] Add continuous FK/jerk-bounded TAMP references with exact collinear compression,
      analytic derivative checks and a smooth phase-rate governor; retain legacy
      comparisons and explicitly reject internal v2 records on the v1 wire.
- [x] Add physically verified loaded start/mid/late/hold resets within the existing
      9% supported-lift share, preserving grasp references and logging raw counts.
- [x] Complete the matched 65,536-step-per-variant reward/curriculum retraining
      and unchanged public TAMP replay under both reference profiles. See the
      [jerk-reference report](docs/experiments/mars_jerk_tracking_20261027.md). Keep base-only tracking valid; retain
      70% general tracking and use source-bound regressions before launching.
      Completed v20: both learned variants 0/8 new full-start lifts and 0/3 public
      deliveries under both profiles; no qualification or curriculum promotion.
      Single-case reward wrappers are diagnostic and have no TAMP admission.
- [x] Expose final physical wheel-motor slew in training configuration; reject
      incompatible exact resumes and record dynamics changes on weight transfer.
- [x] Add source-bound, bounded failure-neighborhood reset sampling with whole-
      family train/validation splits and unchanged physical admission checks.
- [x] Generate development failure families and compare the bounded v25 PPO
      refinement against source/teacher and public deliveries. Result regressed:
      0/6 validation children, all base stops; 0/3 deliveries under both profiles.
      Preserve the complete negative result; do not extend v25 unchanged.
- [ ] Move the demonstrated dynamic retiming improvement into the shared TAMP
      reference-generation path and validate on new cases/load ranges. A 1.1x
      duration fixes the teacher's two retention failures: 19/19 admitted on the
      development set (19/20 including the original support-margin rejection).
      Keep the original baseline and every failure; do not cap RL at teacher success.
- [ ] Audit dependence on executor phase labels; compare a tracking actor using
      only measured state, masked pose/gripper references and explicit retain
      intent. Keep task-stage decisions and release authority in TAMP/execution.
- [ ] Qualify matched Cartesian TAMP carry: an explicit privileged pose teacher
      with admitted checkpoint settings gets past pregrasp but violates retention
      in all three cases (lift/preplace), despite about2mm hand tracking.
      Learned finals still fail earlier. See [matched evidence](docs/experiments/mars_matched_reference_20261106.md).
      The resulting unconfirmed stops expose repeated hold-target rebasing:
      the fix latches the first arm/grip target and passes three recorded loaded
      stop resets in0.10/0.25/0.20s, with original thresholds. Fresh full-source
      gates and an initialized-source compatibility audit precede integrated
      retesting. Do not claim retention fixed or hardware ready.
- [ ] Repair full-volume Mars grasp fit: height priorities fail0/30 each over90
      diagnostic episodes. Open-pad overlap is verified against recorded FK;
      axial pinch36/38mm also fails0/30 each after exact baseline replays,
      with one36mm wheel-unloading guard trip (stop confirmed).
      Three solved-contact probes reproduce original physics exactly; inspect
      wrench capacity, contact locations, joint-stop compliance and angular
      creep before calibrating physical parameters. See [fit evidence](docs/experiments/mars_grasp_fit_20261106.md).
      Keep grasp/placement planning fixes separate from learned tracking; no
      candidate is qualified by static clearance alone.
- [x] Complete the orientation-preserving translation262k trial and full post-tests:
      reject the final (base42/50 full, arm14/88 vs source80/88, coupled4/8).
      Fresh full0/12,2/12,7/12; learned public-tool transfers0/3, stops confirmed.
      Preserve original70/94/122 subsets and all failures. See [complete report](docs/experiments/mars_translation_final_20261106.md).
- [x] Audit actual training returns against physical contract scores before
      another PPO run. All292 exact replays complete; all67 lost-full arm cases
      earn lower final shaped return, and the weighted deterministic reward
      rate declines6.52%. See [audit](docs/experiments/mars_objective_audit_20261106.md).
- [ ] Complete the matched gain-bias learning-rate control300→1, preserving
      all data/reward/physical settings and fixed262k final. Require self/parent
      warmup equality, old-subset/fresh/TAMP acceptance and no automatic promotion.
      The completed objective audit must not be treated as a PPO-causality proof. Require exact source/final native rollouts under the
      real terminal contract wrapper, term/discount accounting and all146 cases.
      Direct hand-to-base readout removal improves arm only29→32/64 and
      regresses coupled4→3/8: reject it as a standalone fix. Herman remains gated.
- [ ] Continue the [Mars refinement hill climb](docs/plans/mars_refinement_hillclimb.md):
      retain the stronger 49/50 base-tracking, 15/16 full-arm source; both quieter
      mixed refinements reach 50/50 base tracking but regress arm smoothness.
      Critic-only warmup improves base value calibration but regresses full arm
      to 5/16; return-unit conditioning plus 40/20/40 coordination training also
      remains 0/4 coupled. Gain-bias ×1000 reaches 1/4 coupled but regresses to
      0/16 full arm: reject it. Preserve every negative result. A real public-tool
      replay stops in pregrasp after 56 policy calls; original arm primitives
      excite only two joint directions. Train all seven motion rows on the
      [rank-complete nearby arm expansion](docs/experiments/mars_tamp_arm_coverage_20261106.md),
      then replay actual admitted pick/place checkpoints. The first seven-motion
      final reaches 151 public-tool calls inside the hand-position tube but stops
      on base drift and regresses arm smoothness: reject it. A matched smaller-step
      control completed in an immutable checkout; procedural near-straight postures and
      coupled amplitudes pass 1,080 regressions and teacher122/122 (94 legacy
      physical cases exact). The smaller-step final is also rejected: arm40/40
      tracking but20/40 full and zero public-tool transfers. The262k scheduled
      posture final is rejected (base50/50 tracking40/50 full, arm62/64 tracking
      7/64 full, coupled2/8 full). The matched trainable-context262k final
      improves arm to63/64 tracking29/64 full and coupled4/8, but regresses
      from the retained source. Its two fresh audits give4/12 and3/12 full,
      and actual public-tool transfers remain0/3: reject it. See the
      [complete context report](docs/experiments/mars_context_20261106.md). The rejected posture final has0/3
      public-tool transfers and0/12,2/12 full fresh-audit passes; teacher/source
      have12/12 on each. See [full report](docs/experiments/mars_tamp_postures_20261106.md). Keep both finals
      and actual public-tool/fresh-pose results; no hardware qualification.
      Require coupled
      dynamic-pose, loaded TAMP and real feedback/watchdog/stop acceptance before
      hardware motion. Primitive success alone does not close this gate.
- [x] Add a frozen small Mars mastery suite: 13 training / 13 nearby validation
      cases, paired FK/axle-valid targets, two-second holding and per-case safety/
      smoothness gates. Teacher passes 26/26; old policy passes 1/26 full mastery.
      See [setup and results](docs/experiments/mars_tracking_mastery_setup.md).
- [x] Add signed command/response, per-case reward/action and post-optimizer PPO
      diagnostics. Merge main harness updates; launch three sequential 262,144-step
      base-only seeds after fresh controller/harness/teacher gates. Results pending;
      see [longer-run protocol](docs/experiments/mars_mastery_long_training_20261028.md).
- [x] Complete the single-seed 262,144-step mastery experiment and matched
      settling replays. Translation endpoints reach 0.02–0.33 mm, but yaw/quiet/jerk
      leave full mastery 0/10. Repeats cancelled; do not extend unchanged. See the
      [completed report](docs/experiments/mars_mastery_long_training_20261028.md).
- [x] Add/test opt-in physical wheel jerk smoothing and dense fine-yaw/quiet
      rewards with aligned 2 s bonus. Frozen old actor improves to 6/10 mastery
      and 10/10 smoothness; teacher26/26, regressions873. One 131k-step turning
      refinement running (first16k:8/10mastery,10/10smoothness); see [v2 report](docs/experiments/mars_turning_jerk_20261029.md).
- [x] Complete the filtered refinement: 9/10 full mastery, all10 smoothness,
      remaining train-left yaw. Audit added filter: 0.65 s small-step rise; frozen
      actor fails without it. Preserve baseline; see [filter audit](docs/experiments/mars_filter_audit_20261029.md).
- [x] Complete the unfiltered 64k comparison:5/10mastery,9/10tracking+quiet,
      median measured linear jerk down82%, raw-command jerk down91%. Fixed filtered
      smoothness only3/10; retain every failure and comparison. See [completed audit](docs/experiments/mars_filter_audit_20261029.md).
- [x] Add an explicit 25-pair base primitive/posture training population; preserve
      canonical ten-case evaluation, thresholds and disjoint validation postures.
      Expanded teacher admission is required before training. See [protocol](docs/experiments/mars_crossed_posture_20261029.md).
- [x] Complete the bounded crossed-population refinement and matched expanded
      replay: canonical5/10→6/10, expanded14/30→16/30; tracking stays9/10 and27/30.
      Smoothness is mixed; three left-turn tracking failures persist. See [complete results](docs/experiments/mars_crossed_posture_20261029.md).
- [x] Audit signed yaw error and deployed normalization: correct positive error
      reaches the actor, but current/preview learned yaw gains become negative in
      the failed posture. Add opt-in latched turn-divergence protection and a
      45-case nearby signed-turn replay preset; see [diagnosis](docs/experiments/mars_yaw_sign_20261030.md).
- [x] Add/test opt-in positive restoring yaw gains and opposing learned damping,
      with smooth preview/mask-based selection and preserved general outputs.
      Explicit non-function-preserving transfer; see [experiment](docs/experiments/mars_restoring_yaw_20261030.md).
- [x] Complete the 64k restoring-yaw experiment: expanded tracking40/50→50/50
      before learning and retained50/50 after PPO; full23/50→33/50→34/50. Canonical
      tracking10/10, full6/10; mixed jerk and validation smoothness regressions kept.
      See [complete results](docs/experiments/mars_restoring_yaw_20261030.md).
- [ ] Audit peak-jerk failure timing and measured/commanded channels against the
      integrated smoothness reward before another extension or stage expansion.
      Record mode-radius changes explicitly in restoring-head weight transfers.
- [x] Finish the matched gain-bias LR300-to-1 control and all post-tests: full
      arm14/88→68/88 but below source80/88; public learned transfers remain0/3.
      Preserve the source. See the [completed report](docs/experiments/mars_bias_control_20261107.md).
- [x] Reproduce the pregrasp gap with supported open-jaw tracking probes:
      teacher2/2, both learned actors0/2, while all pass the free-space version.
      Preserve phase/jaw-input interventions, including the deadline failure.
- [x] Admit and train the bounded supported-contact curriculum: preserve all146
      earlier cases, add18 contact cases, reserve12.5% of total control steps
      for them, and freeze12 new-seed audit cases before optimization. Require
      fresh regressions/teacher/source/warmup gates; no hardware qualification.
- [x] Complete the supported-contact final and all post-tests: fresh contact
      full3/12→9/12, but earlier arm full80/88→63/88 and learned public tools0/3.
      Retain source; all1,138 regressions and teacher164/164/fresh12/12 pass.
      See the [complete report](docs/experiments/mars_supported_contact_20261108.md).
- [ ] Prevent base jerk regression during arm learning and close the actual
      pregrasp transfer gap before another broad training extension. Keep all
      earlier tasks, supported-contact audits and full public-tool failures visible.
- [x] Replay all114 arm/coupled cases exactly and isolate wheel readout/shared
      context effects: restoring old wheel feedback helps free-space smoothness
      but loses contact/coupled passes. Reject both in-memory interventions.
- [x] Audit actual jaw-input clipping and hidden activation saturation against
      six exact supported-training replays. Reject static jaw rescaling alone.
      See [feedback/unit diagnosis](docs/experiments/mars_feedback_units_20261113.md).
- [x] Test opt-in jaw-unit curriculum after exact critic warmup, with physical
      std floor0.087275rad, a32768-step geometric ramp and the unchanged262k final.
      Require fresh regression/teacher/source gates and all existing post-tests.
      Fixed 262k and post-tests complete: earlier arm 53/88, supported 4/18,
      learned full public tools 0/3. Reject final and retain source.
- [x] Implement the [v2 general/TAMP training prototype](docs/plans/mars_whole_body_v2.md):
      50/50 physical ticks, one full-history actor, three independent critics,
      training-only privileged state and retimed S-curve velocity references.
      Keep the diagnosed object-motion challenge as a failure, not a pass.
- [x] Complete fresh v2 full regressions/preflight, then the bounded8k pilot
      and paired source/general/TAMP audit. Pregrasp is not full pick/place.
      All 1,170 regressions pass, but final full 2/164 and pregrasp 0/6.
      Preserve repaired harness failures and [complete report](docs/experiments/mars_whole_body_validation_20261113.md).
- [x] Audit per-critic advantage scaling and implement the bounded
      [normalization diagnostic](docs/plans/mars_whole_body_advantage_pair.md).
      Full source-bound regressions 1,187/1,187; original teacher admission
      retained. Historical control completes 4,096 decisions; shared candidate
      stops before its first update on JSON-list/Python-tuple settings comparison.
      No paired result existed at that snapshot; completed recovery is below.
- [x] Repair and test the serialized/in-memory settings comparison (`29a9fcec`),
      including an actual candidate call against the saved control record.
      Preserve that incomplete attempt separately from the completed recovery.
- [x] Complete the declared normalization comparison with an explicit source
      bridge and unchanged initial/first-rollout hashes. Fresh regressions
      1,249/1,249; shared final 2/12 full versus per-head 0/12 and initial 2/12.
      Completed arm endpoints within 3mm/0.02rad improve 1/10 to 5/10, but
      quiet/base/jerk remain unresolved. Preserve all failures and the
      [completed report](docs/experiments/mars_c1_normalization_recovery_20261115.md).
- [ ] Improve actor-active terminal settling and base authority while retaining
      the C1 hand-following gains. Distinguish endpoint accuracy from the full
      quiet/force/jerk contract; do not silently extend the completed C1 budget.
- [x] Correct bundled TAMP reward timing: discount each physical tick, including
      terminal credit, before `gamma ** ticks` continuation; preserve raw rewards
      for evaluation and test bundled-versus-expanded return equivalence.
      All 61 focused tests pass, including the native oracle terminal hold.
      See [timing correction](docs/experiments/pr227_tick_reward_timing_20261114.md).
- [x] Evaluate the opt-in frozen-base/shared-context arm-adaptation pilot from
      [PR #232](https://github.com/cpaxton/home_robot_v3/pull/232). The new 32,768-step
      pair completes after C1: baseline arm full 59→57/64; velocity 60→61/64;
      protected serialized functions and every base-only metric are preserved.
      Record the [complete result](docs/experiments/mars_frozen_base_arm_20261115.md)
      and earlier zero-step queue failure separately. One seed, coupled 2/8;
      keep the retained production source and no automatic extra training.
- [ ] Fix cold-start exploration/completion learning and audit per-critic
      advantages before scaling v2. Keep retained-source comparisons and all
      measured safety/precision gates; add the missing mask/observability strata.
- [ ] Add v2 grasp→lift/loaded/placement primitives and actor-active terminal
      stability audits only after the first pilot's behavior is understood.
- [ ] Master the five base cases first, then add arm, coupled and loaded cases
      while retaining earlier rehearsal. Keep stage IDs out of actor input and
      require both complete splits before restoring the broad distribution.
- [ ] Test base-error feature conditioning and separate arm/base exploration on
      matched recovery/trajectory cases, with function-preserving initialization
      where possible; report safety and generalization alongside tracking error.
- [ ] Extend the TAMP trajectory contract with an XY base region/optional yaw and
      untimed geometric paths, with smooth retiming and load-aware validation in
      the planning/execution layer. Keep task decisions out of the controller.
- [ ] Test relative-frame inputs or consistent world-coordinate augmentation;
      preserve public pose/mask semantics. Add explicit small base-heading recovery
      resets, derivative actor inputs and optional velocity/acceleration tracking
      ablations after reference rollout gates. Require multi-case settling before
      wider randomization, base/transport retiming or hardware admission.
- [ ] Pass moving-goal orientation/base safety, loaded and full public TAMP
      gates. The final 65k-step refinement reaches nearby positions within 0.42 mm,
      but orientation still misses the strict 0.02 rad contract. Two wrist probes
      avoid their earlier base abort; paired no-regression still fails. See the
      [hold/Cartesian report](docs/experiments/mars_hold_cartesian_20261025.md).

- [x] Add opt-in tracking-majority retention training: 70% tracking, bounded
      grasp/carry/place/disturbance shares, retention intent, closed-empty
      counterexamples, mass/COM randomization and separate advisory object-held
      predictor. See [training plan](docs/plans/retention_training.md).
- [x] Run 10,240 additional PPO smoke steps with offline W&B and a new-seed
      paired occupancy diagnostic; preserve the failed arm/action heads. The
      gripper-local head passed six closed-empty and six loaded attempts, but
      full task success has not improved or qualified. See the
      [experiment report](docs/experiments/mars_retention_training_20261024.md).
- [ ] Qualify occupancy prediction on independent objects/grasps with calibration,
      held recall and empty-hand false-positive rates; confirm loaded training
      improves retention without degrading tracking, smoothness or wheel support.
      A short training diagnostic does not close these performance gates.

- [x] Isolate placement rearm bias regression with matched frozen-case ablations;
      add explicit free-space compensation authorization and regression tests.
      Full source-bound requalification remains required below.
- [x] Add simulator task-pose command-protocol prototype, GT TAMP component replay,
      measured-Jacobian teacher, and disabled-by-default guarded driver interface.
      These are not public-tool, learned-policy, or hardware qualification.

- [x] Restore continuous pick/place after explicit stage rearming: v5 at
      `42aa2f30` passed 27/30 versus v4 30/30. Diagnose the 20 g seed 1860001
      retention loss and 50 g seeds 1860007/1860010 tracking timeouts; preserve
      all cases and thresholds. Immutable `b181db61` requalification passed
      298/300 carry and 30/30 continuous pick/place, with 367 fresh regression
      tests and a verified combined readiness receipt. The canonical carry-training
      supervisor remains stopped;
      this evidence does not qualify subsequent public-tool/control changes.

- [x] Add explicit Mars simulator support to public TAMP discovery/planning/
      execution, a frozen 30-case two-support catalog, independently measured
      task scoring, per-stage failure reports and trajectory visualizations.
      Implementation is not task qualification: moving-base retention remains
      under investigation and all failures count toward the acceptance gate.
- [x] Add the real agent prompt/parser/dispatcher harness with three local
      agent seeds per case and an oracle-readiness prerequisite. Missing normal
      observations stop before model calls; GT never substitutes for observations.
- [ ] Pass all 30 public-tool two-support transfers without changing the frozen
      cases or retention/force/wheel limits; diagnose loaded turning, then
      support-confirmed release and post-placement stability.
- [x] Complete and archive the full public-tool v2/v3 reports: 4/30 at
      `a17a5734`, then 16/30 at `9dee06fc` (20 g 10/10; 50 g 6/10; 100 g 0/10).
      All fixtures admitted; the public gate remains false. That snapshot's source-bound
      regression receipt covers 588 passing tests. See the
      [experiment report](docs/experiments/mars_public_tamp_20261023.md).
- [x] Add free-base hand-only pose tracking to the teacher and actual nearby FK
      witness tests for both Mars and Stretch. These free-space checks do not
      qualify loaded tasks, learned checkpoints or real hardware.
- [x] Complete one frozen public-tool 20 g transfer with actual base motion and
      independently measured support/release/stability. This selected positive
      case does not close the full gate. Preserve the earlier complete 0/30
      development sweep and failed fixture-binding report alongside new results.
- [ ] Improve public-task execution time after retention and wheel gates pass;
      the first full successful diagnostic took about 100 simulated seconds.
      Keep geometric retiming and fixed-deadline tracking scores distinct.
- [ ] Install an observed-scene provider and run the normal-observation agent
      acceptance trials after the public oracle gate passes. No real-model
      success is established by parser/dispatcher tests.

- [ ] Integrate the qualified controller with the robot command interface: masked
      base SE(2), hand SE(3), and gripper targets; trajectory updates, measured versus
      commanded state, speed/effort limits, deadlines, stop confirmation and ownership.
      Keep simulator qualification separate from hardware readiness.
- [ ] Test the integrated agent through public TAMP tools, first with ground-truth
      oracle plans and then the agent: physical pick/carry/place, goal changes,
      interruptions and recovery. Record retention, task success, tracking, settling,
      jerk/contact loads and wheel support; include coordinated moving-base carry.
- [ ] Test on Herman (real-world Innate Mars) after simulator and interface gates:
      verify feedback and stop behavior, then staged bounded motion and compliant
      interaction under an explicitly authorized hardware procedure. Compare measured
      trajectories and loads against simulation; do not infer hardware safety from
      simulator success or start motion from this checklist alone.
- [x] Passive Herman connection check (2026-10-23): SSH responds and ROS Humble
      is installed. The default ROS graph showed only metadata topics; actual
      driver domain, modes, watchdog and stop behavior remain unverified. No
      deployment, service start, motion or contact commands were sent.
- [x] Audit upstream velocity units and failed-read behavior; prepare and test
      a patch on temporary copies without changing upstream or Herman. See the
      [hardware feedback audit](docs/experiments/herman_feedback_audit_20261023.md).
- [ ] Review/apply the upstream feedback patch, build the ROS package, propagate
      invalid reads/source ages and actual odometry twist through the bridge,
      and verify installed units/signs before implementing a physical velocity
      driver. Calibrate typed current/load proxies before treating them as forces.

## Proprioceptive contact learning (2026-10-23)

Reference: Bronars et al., [Blind Precise Assembly](https://arxiv.org/abs/2610.23849),
[project](https://younghyopark.me/blind-proprioceptive-assembly/),
[mjfit](https://github.com/younghyopark/mjfit).
See the research roadmap in [the control runbook](docs/mars_reactive_rl.md#proprioceptive-contact-learning-research-roadmap).

- [x] Add the paper citation, related-work scope, and staged research roadmap.
- [ ] Audit available Mars feedback and identify effective actuator response from suitable
      command/measurement logs. Report delay, lag, deadband and uncertainty; fit physical
      friction/armature only when identifiable. Simulation logs are not hardware calibration;
      hardware excitation needs a separately authorized procedure.
- [ ] Test contact observability offline before modifying PPO. Label obstruction, support
      contact and slip from simulation; evaluate history-only inference on held-out episodes
      and geometries, including false negatives, detection delay and uncertainty.
- [ ] Compare temporal convolution and the current transformer with identical history duration,
      cadence, observations and budget. Then ablate auxiliary contact supervision and privileged
      critic inputs separately; ground-truth labels must never reach the deployed actor.
- [ ] Add separate mjfit passive-fit/contact probes and a local insertion fixture with pinned
      generator provenance and explicit timestep/contact settings. Preserve existing carry and
      frozen Stretch fixtures; mjfit is geometry generation, not actuator identification.
- [ ] Compare each candidate with the baseline on tracking, retention, settling, jerk,
      contact loads and wheel unloading. Keep hard effort/safety limits active throughout;
      retain our goal-conditioned SE(2)/SE(3)/gripper interface and TAMP recovery layer.

## Shared-agent acceptance follow-up (2026-10-22)

- [ ] Diagnose S0 optical heading and retained target-view coverage before
      treating black/sky-only RGB as broken perception. Reproduced on frozen
      main/provider; the stationary diagnostic pan faces away from the table
      and CPU camera rays hit the dark floor, not the robot shell.
- [ ] Clear context-stack acceptance without restoring answer overrides:
      EQA4 native scores 4/4 main versus 1/4 candidate; three main scores differ
      from its raw final model answer. Inspect q14 hallucinated TV and q56
      missing curtain context, then test existing opt-in pose history and
      retained-dialogue ablations. Preserve model authority and exact packets.
      See `docs/plans/shared_stack_readiness.md` for evidence and gate IDs.
- [x] Separate fresh pre-turn preservation from optional extra-history ranking.
      q56 gains with selector on because image 2 finally shows bookshelf plus
      blue curtains; added pose-novelty images 3/4 are blank walls. Preserve
      scope/age/dedup guards and model authority. Owner repair `92f31497`.
- [ ] Clear the four-ID EQA gate for independent current/pre-turn handoff before
      promoting the remaining context stack; unit tests do not establish gains.

## TAMP simulator API readiness (2026-10-05)

- [x] Reclaim disposable pip caches; preserve source worktrees, datasets and run evidence.
- [x] Consistent JSON across all four TAMP tools; boot/pose/clearance guards,
      partial execution evidence and stale-state checks. Full regression pack: 206 passed.
- [x] Correct unsupported shoulder-stall diagnosis; retain object-independent failures.
- [x] Complete gate `20261005_183936_f4b352`: failed CHAT smoke, 1/3 admitted,
      2/3 scored replays passed. Simulator readiness remains failed.
- [x] Repair planning snapshots, handle namespaces, and operation-scoped evidence;
      218 regression tests pass. See `docs/experiments/tamp_api_correctness_20261006.md`.
- [x] Inspect CHAT smoke `20261006_160739_730b9a`: pickup passed, placement failed;
      returned handles and operation-scoped partial evidence behaved correctly.
- [ ] Capture the placement command failure reason with private diagnostic logging.
      Resolve general motion failures before repeating the nine-trial readiness matrix.
      See `docs/experiments/tamp_api_readiness_20261005.md` and `docs/api/tamp.md`.

## TAMP agent execution follow-up (2026-09-29)

Evidence: [agent execution and paper figures](docs/experiments/tamp_agent_execution_20260929.md).

- [x] Export baseline coverage/success and task-layout figures as PDF/SVG/PNG.
- [x] Instrument lift outcomes; identify height-only false-positive acceptance and
      metre-scale attachment offsets. Historical 21/25 is not verified grasp success.
- [ ] Validate measured-joint arrival and target-distance guards, repair remaining
      execution failures, then run paired trials across all original admitted tasks.
- [x] Complete 90-candidate oracle expansion: 19 admitted, 19/19 fresh MCTS passes,
      five new admitted case IDs and zero startup errors. Updated oracle-only figures saved.
- [x] Diagnose RBY1 tracking: wall-intersecting approaches and floating-base pose
      reset artifact; endpoint rejection and solver support validated by 102 tests.
      See `docs/experiments/tamp_contact_tracking_20260930.md`.
- [x] Retry repaired execution: kettle relocation passes after reserving IK
      tracking margin; latest case is 1/3 objects, 0/1 full tasks (latch control).
- [x] Execute preselected alternative approaches: three fresh-process repeats
      pass (3/3 full cleanups, 9/9 relocations, 54/54 arm arrivals), assisted latch
      only. See `docs/experiments/tamp_approach_validation_20261001.md`.
- [x] Integrate online alternative-approach selection with read-only collision queries:
      139 regression tests pass; three fresh MCTS repeats score 2/3 full tasks and
      7/9 relocations. See `docs/experiments/tamp_auto_approaches_20261001.md`.
- [ ] Diagnose residual apple tracking and kettle invalid-start failures; verify arm
      collision geometry and payload sweeps before resuming broader latch expansion.
- [ ] Measure the full shared agent on frozen valid tasks with explicit model,
      observations and action budgets; do not label scripted/GT controls as agent results.

## Physical TAMP acceptance (2026-09-26)

Evidence and commands: [physical acceptance](docs/experiments/physical_tamp_acceptance.md).
The user-selected experiment scope is GT/MCTS only, including the full 200-row
registry. Oracle teleport and kinematic latch remain separately labeled controls.

- [ ] Pass three matched physical pick/place trials on each original Molmo and
      RoboCasa fixture, with independent retention, support, contact and actuation audits.
- [ ] After that gate, validate broader layouts and bounded partial-map observation/replanning.
- [x] Complete the 24-case protocol, seven small-registry rows, GT floor/tool controls,
      and all 200 full-registry rows; retain failures, unsupported capabilities and timeouts.
      Full registry: 54/200 successes; corrected 50-row subset: 18 successes, kept separate.
      See the [results report](docs/experiments/physical_tamp_acceptance_20260926_results.md).
- [x] Add scene-backed candidate construction, independent executed reference
      admission, immutable pose certificates, and final-scene scoring for controls.
      See [validated registry workflow](docs/experiments/tamp_validated_registry.md).
- [x] Complete corrected r3 admission and fresh MCTS replay; report construction and
      reference failures separately from the scored denominator. September 28 pilot:
      4/8 admitted and 4/4 fresh MCTS replay passes. Full job `20260928_222600_2cf5c6`
      finished: 25/200 admitted, 21/25 scored successes. No 200-task solvability claim.
- [ ] Expand scene coverage with geometry-based object selection; verify reference
      repeatability and diagnose Mars startup failures before broader claims.
- [ ] Resolve rotational grip slip in RoboCasa and Molmo under the frozen physics
      and retention bounds; both now have independently verified physical pickups.
- [x] Revalidate Molmo grasp with open-gripper geometry; approach and physical pickup now pass.
- [ ] Complete Molmo retained transport and placement. Revalidated route reuse now
      executes carry, but rotational grip drift exceeds 0.1 rad before release.
      Release and open-finger retreat have static certificates; physical acceptance is pending.
- [ ] Add physical adapters and collision coverage for deferred robots before claiming
      multi-robot physical acceptance. Habitat-OVMM, long-horizon tasks, learned-agent
      comparisons and hardware remain deferred. Update paper claims only from scored artifacts.

## Shared query-memory acceptance (2026-09-05, prototype branch)

Canonical plan: [shared-agent acceptance and paper figures](docs/experiments/shared_agent_paper_update.md).
No full sweeps; one shared policy across tasks, capability/configuration adapters
for necessary robot geometry differences. Do not promote based only on graph size.

- [x] Sequential TAMP controls: rby1/Sourccey table, unreachable-grasp rejection,
      and rby1 furnished-scene agent-tools execution passed; 77 focused regression
      tests passed. These are oracle-assisted controls, not learned manipulation.
- [x] Recorded Sourccey chase video and camera stills; publication rendering review
      remains necessary. Source/results are recorded in the canonical plan.
- [x] Started Herman's existing bridge without deployment or motion commands;
      passive head-camera pose check passed (0.5 mm / 0 deg gaze mismatch).
- [ ] Wrist camera: tabled at user request. Driver expects Arducam but device is
      absent from USB enumeration; ROS publisher count is zero. Do not claim wrist
      capture works or issue arm motion to troubleshoot it.
- [ ] Stationary head-stereo/depth/query-memory proof of life; no base yaw/translation,
      arm, or gripper commands. Head tilt is permitted only if needed.
- [ ] Matched OVMM diagnostic submitted: job `20260905_144422_d900d5`, artifacts
      `/tmp/emet-ovmm-paired-20260905`, source `74a7f4e5`. DynaMem, lazy arrival,
      query-driven; episode `hm3d_lamp_bed_00006`; each gets 12 rounds / 8 nav steps.
      Runs serially with CPU-safe affinity and GPU lock. Inspect task outcomes and
      grounding traces, not just process exit. This is one diagnostic, not the pilot.
- [ ] Complete bounded paired EQA/OVMM acceptance and learned manipulation gate.
      Historic lazy 8/16 vs deep-merge 6/16 is development evidence; the new query
      policy has not yet demonstrated improvement. Preserve negative results.
- [x] Post-fix Habitat diagnostic: DynaMem 1/2, lazy arrival 0/2, query-driven 0/2
      at `39640791`. DynaMem now has the shared VLM; inspection escape executes.
- [x] Save actual grounding RGB/depth/masks and verifier outputs (`c5ff2347`).
      Instrumented repeat `20260905_230122_e483a6` scored 0/2. A suspicious VLM
      lamp selection scored 0.0302 and was correctly blocked by admission 0.12.
      Offline replay admits it at 0.03; live thresholds are unchanged.
- [ ] Red/blue MuJoCo comparison `20260905_230647_10f8d9` at `3621d63e`:
      DynaMem agentic 0/2, query-driven 0/2. Shared query policy is now wired
      into the MuJoCo find runner and CLI; no separate robot-specific policy.
- [x] One-shot DynaMem red/blue control `20260905_230946_a9cc38`: 1/2 (object
      found, receptacle error 0.60 m). IMPORTANT: S0 executor ignores the requested
      four-view override and scans eight views, so this is not a same-map control.
      Red-cylinder voxel score 0.106 was rescued by detector confirmation; the
      query-driven 0.14 retrieval floor would exclude that particular candidate.
- [x] Exact existing CPU/Stretch S0 test `20260905_232835_c3cd73` completed:
      FAILED object localization; retained its success assertions.
- [x] Eight-view comparison `20260905_232921_b3f4a7`: both rows 0/2;
      DynaMem 63 s, query-driven 278 s / 22 graph nodes. More coverage alone
      does not fix localization (unchanged thresholds, 12/8 action budgets).
- [x] Unify S0 one-shot/agentic factory, memory profile and mapping protocol;
      record requested scan steps and actual frames added.
- [x] Query-conditioned fresh masks, bounded per-source detector recovery for
      weak voxel hits, independent verification for compound targets. Recovery
      creates search evidence only; admission thresholds unchanged. 95 focused
      tests pass. Post-change live comparison still required.
## ZMQ follow-ups after PR #135

- [ ] Move onboard DINOv3 loading/inference off the publication loop into a
  bounded latest-frame worker; retain source timing and test stalled inference.
  Keep onboard DINOv3 disabled during navigation/benchmark acceptance.
- [ ] Derive simulated lidar hits from sensor site origins/directions in an
  explicit common frame; test reordered and translated sites.
- [ ] Validate Mars image header progression, stereo skew and timestamped TF
  while stationary; propagate provenance to depth and embeddings.
- [ ] Extend image timing and stale-frame handling across Stretch/simulator
  publishers without treating legacy unknown timestamps as fresh.
- [ ] Keep experimental H.264 disabled for acceptance; validate independent
  cadence, encoder lifecycle and bandwidth before enabling it.
- [ ] Complete bounded EQA/Habitat OVMM pilots and separate TAMP controls on
  frozen integrated revisions. Native DynaMem is a system baseline, not a
  planner-controlled memory ablation. Update paper only from measured results.

## Deferred embodiment acceptance (not a blocker for the current pilot)

- [ ] Sourccey: inspect obstructed camera view, then repeat hold/turn/translation
  and scene-health gates. One integrated precision route passed; learned
  search and physical robustness are not established.
- [ ] Galaxea proxy: repair matched support/contact/actuation before claiming
  reliable translation. Existing posture-fault stop evidence is not a route pass.
- [ ] Validate opt-in navigation policies across Stretch/Mars simulation and
  timestamped hardware adapters. Do not enable policies on unsupported peers.
- [ ] Keep passive base support and named settling policies opt-in until their
  embodiment acceptance passes. No new hardware motion is authorized here.

## Config over env flags

We over-use `EMET_*` for robot/controller policy (head sweep, teleport, TTS, rotate
steps). Embodiment defaults belong in `robots.<id>` / `mapping.*` in
[`configs/emet/default.yaml`](configs/emet/default.yaml) (see Stretch
`look_around_head_sweep`). Env should stay as a rare override (`EMET_FORCE_HEAD_SWEEP`
for paper pans) or for host/GPU process state. Do not add a new `EMET_*` for
something that can be a YAML key + `--set`. `run_ovmm_find_recep_slice.sh` no
longer exports `EMET_SKIP_HEAD_SWEEP` — YAML already defaults pans off; only
`PROFILE=stretch-legacy` sets `EMET_FORCE_HEAD_SWEEP`. Same pass later for other
script-exported knobs that duplicate config.

## Eval orchestrator layers (keep three; do not add a fourth matrix script)

Too many bash wrappers all mean “run a bit of Habitat + OVMM + SQA3D.” Canonical
layers — docs in [`docs/evaluation.md`](docs/evaluation.md) and
[`docs/experiments/README.md`](docs/experiments/README.md):

| Layer | Entry | Job |
|-------|--------|-----|
| Path smoke | `scripts/run_simulation_smoke_battery.sh` | Seven-track merge-gate (often GT/mock) |
| Overnight regression | `scripts/run_overnight_cross_track_smoke.sh` | Tier-0 + pytest + those tracks. **Never** chain VLM after this. |
| Fast OVMM | `scripts/run_ovmm_find_recep_slice.sh` | rby1 / teleport (`PROFILE=smoke`) |
| Habitat OVMM VLM | `scripts/smoke_habitat_ovmm_agentic_find.sh` | One HM3D scene, agentic 4/4 |
| Paper numbers | `scripts/run_paper_matrix.sh` + `emet hmeqa overnight` | Real sizes, one GPU lock |

Deleted 2026-09-02: `scripts/run_overnight_eval_smoke.sh` (tiny real-VLM matrix +
figure pack). Replacements above; figures: `scripts/build_eval_figure_pack.py`.

### Later cleanup

- [x] Fold `run_representative_benchmark_sample.sh` into
      `run_dynagraph_tuned_paper_battery.sh` — **keep both**. Representative =
      static_graph comparison + S0 matrix + tables; paper battery = seven-track
      + holdout/bal32 + tuned dynagraph numbers.
- [x] Make `run_overnight_habitat_eval.sh` a thin wrapper around
      `emet hmeqa overnight`. Extra slices stay as dedicated scripts
      (`run_hmeqa_annotated37_h2h.sh`, `run_hmeqa_paper113_h2h.sh`,
      `run_habitat_iter_subset.sh` for paper-20).
- [ ] Leave `run_hmeqa_*_h2h.sh` until `emet hmeqa h2h` covers those ID sets;
      then delete the one-off H2H scripts.
- [x] `run_habitat_ovmm_joint_gate.sh` vs `run_paper_matrix.sh` — **keep both**.
      Joint gate = count/clock 15-qid + rby1 OVMM `PROFILE=slice`. Paper
      matrix = HM-EQA paper-113 (no semantics) + OVMM S0/S1/S2 numbers.

## Grounded graph room-evidence A/B — no-go for scale; focused history pair authorized (2026-08-23)

Canonical record:
[docs/experiments/graph_room_evidence.md](docs/experiments/graph_room_evidence.md).

### Completed evidence and recovery

- [x] Reconstruct the 2026-08-19 reboot. The last record is an NX-protected
      supervisor instruction-fetch kernel fault; there was no OOM, `libcuda`
      crash, NVRM Xid, or active managed GPU job. This is not evidence of GPU
      starvation; details are preserved in `segfault.md`.
- [x] Archive the old pre-manifest treatment/control OUTs as diagnostic-only.
- [x] Complete the A0 resolver gate at `e85cda20`: **4/6**, including **3/4**
      on regression IDs `6,11,12,47`.
- [x] Complete A1 collection/no-leakage at `fae4b89c`: **4/6**; 621 evidence
      views, 55 ledger rows, 11 room events, and all 40 policy-visible event
      lists empty.
- [x] Complete the strongest clean same-commit comparison at `fae4b89c`:
      A1 shadow **4/6**, 29.2 mean planning steps, 2 budget hits; A2 grounded
      **3/6**, 34.2 steps, 3 budget hits, and room events on only 3/6 episodes.
- [x] Record the A2 decision: **do not promote or scale**. It gained q47, lost
      q12 and q76, did not fix q2, and did not demonstrate lower dwell or fewer
      repeats. This compares the whole grounded treatment, not graph evidence
      alone.
- [x] Complete the pre-fix GT-free q11 infrastructure sequence. Three complete
      runs scored **0/3**; nonnumeric IDs and blocked-navigation redirect worked,
      but trash-can discovery/coverage was still unsolved at that point.
- [x] Fix CPU-audited correctness blockers before another run: strict
      off/shadow/agent visibility, semantic handling of a valid `None` answer,
      placeholder-safe fallback, world-frame room refresh, qualified negative
      evidence, pending-answer visibility, enrich-label ID mapping, explicit
      semantic/enrich oracle axes, and staged compact checkpoints that cannot
      retain stale pixel/voxel state.
- [x] Run one explicitly diagnostic, dirty-tree GT-free q11 A2 canary after an
      EGL safe-start. Job `20260822_175646_219c08`, OUT
      `~/runs/emet/gre_q11_a2_canary_d3ee32e8_20260822_175238`, scored **1/1**
      with a direct present+answerable view at obs 76, semantic answer
      `Next to the refrigerator`, no salvage, and no native failure. This is a
      capability signal, not clean comparison or scale evidence.

### Capability gaps

- [ ] q2: parse one target room and bind gray rug evidence to the shower.
- [ ] q11: diagnose the clean-repeat regression. The dirty diagnostic was 1/1,
      but clean HEAD `81a2c689` reached an initial kitchen view without seeing
      the trash can, then exhausted eight rounds and answered by fallback.
- [ ] q12: validate the new direct-image-over-graph precedence on a clean run;
      the prior grounded treatment trusted a graph undercount.
- [ ] q76: enter and cover a bedroom instead of exhausting living-area frontiers.
- [ ] Add `scripts/summarize_graph_room_evidence.py`; fail on manifest parity
      violations and emit paired dwell/path/repeat/budget/evidence metrics.

### Conservative next experiment

- [x] Freeze and push the reviewed GT-free implementation at `f8ceaa49`
      (`--no-hm3d-semantics`, `--no-enrich-labels`) in PR #115. The focused
      suite, `agent-regression`, and pre-commit pass.
- [x] Finish the consistency audit: apply quantization/tool/nav budgets at
      runtime, reject unfrozen policy environment, hash the small dataset
      inputs, fail closed when requested semantic assets are unavailable, and
      make generic batch resume parity-safe. Focused audit tests pass; keep PR
      #115 draft until the separate CPU baseline gate is resolved.
- [x] Close the simultaneous-launch race: `emet jobs run` now auto-detects
      GPU-like experiment commands and holds a host-wide flock for the full
      job lifetime; zombie supervisor PIDs no longer stall queued jobs.
- [x] Keep benchmark identities explicit: q0–112 is the historical emet
      113-row slice; the upstream GraphEQA HM-EQA semantic-filtered enrich
      sequence has 114 rows. Do not call either mapping authoritative without
      naming which one is used.
- [ ] Close the CPU baseline gate before another GPU run. The full no-sim run
      has three failures after isolating BLAS thread settings: the trajectory
      color and two crop-size assertions in `test_map_grid.py` /
      `test_map_snapshot.py`. Those source files are unchanged from
      `origin/main`; fix or explicitly record that baseline separately, and do
      not describe the full suite as green.
- [x] Repeat the q11 A2 canary from clean PR HEAD `81a2c689` after detached
      EGL probe `20260823_003909_ab5972` passed. Job
      `20260823_004358_462498`, OUT
      `~/runs/emet/gre_q11_a2_clean_81a2c689_20260823_003735`, completed
      without a native failure but scored **0/1**: the first kitchen check and
      every later view marked the trash can absent, then the eight-round budget
      forced `mcq_debias` answer A instead of D at 55 planning steps. This
      fails the direct-evidence gate; stop the scale ladder.
- [x] Fix the first-launch CLI-to-script manifest handoff exposed by the
      canary: the inner script now validates a launcher-prepared manifest
      without switching episode semantics to resume. Managed H2H paths also
      skip in-job stale-process cleanup.
- [x] Finish the PR #115 state-integrity gate in two reviewable commits:
      managed lock/environment/completion integrity first, then grounded
      graph-state contracts and image-free clean/dirty replay regressions.
      **PR #115 CPU integrity gates: complete** — 141 lifecycle/completion,
      182 agent-regression, 108 graph-memory/evidence, and 11 focused
      state/metadata tests pass. The PR stayed draft through the clean q11 and
      action-history GPU diagnostics. Their capability result is negative, but
      the treatment remains opt-in, so that result does not block merging the
      audited state/lifecycle infrastructure.
- [x] After the CPU gate, detached EGL probe `20260823_081023_d24734` passed,
      then clean q11 A2 job `20260823_081229_66d6e3` ran two native-clean
      attempts at `b91059e5`. Both exhausted eight rounds with no direct
      trash-can evidence and answered A instead of D; the capability gate
      remains failed.
- [x] Fix the completion-policy bug exposed by that run: object-crop mosaics
      are best-effort when an episode has no usable instance crops, so absence
      is valid while any diagnostics-declared mosaic remains strictly checked.
      The focused completion suite passes 12/12.
- [x] Add a strict config-file choice for dedicated grounded action history.
      `configs/benchmarks/hmeqa_action_history_{shadow,agent}.yaml` hold
      decision/graph/room/oracle axes fixed; only ledger visibility and variant
      ID differ. Shadow now filters recent outcomes, loop flags, global/place
      attempts, and mirrored attempt events from policy state while retaining
      auditable rows. Focused tests and the paper/docs protocol pass.
- [x] Run the user-authorized, history-only pair on `6,11,12,47`, one managed
      job at a time after successful detached probes. Shadow job
      `20260823_085244_c42458` and agent job `20260823_091138_036fc6` both
      completed native-clean with validated 4-unit `DONE`. Both scored **3/4**
      with identical letters; agent-visible history raised mean planning steps
      **35.25→39.25** and attempts **31→37**, while repeat-failure rows fell
      **6→4**. q11 still failed A vs D; q12 took 16 extra steps. Mechanism and
      no-leakage pass, capability/efficiency do not. Keep it opt-in and do not
      scale; frozen summary is
      `paper/data/hmeqa_agentic_h2h/action_history_pair_20260823.json`.
- [x] Implement typed cross-round action memory and a benchmark-scoped
      pre-router progress gate. Grounded state v3 now leads with semantic intent
      + stable place/frontier/view identity rather than `r0`/`obs=15`, and
      `action_history.py` defines deterministic work/equivalence keys, progress
      tokens, typed outcomes, and one transient retry. The independent
      manifest-frozen axis is `eqa.action_progress_mode=off|shadow|enforce`
      (default off). Shadow leaves cards/execution unchanged; enforce omits
      unchanged terminal/no-progress variants from the exact allowlist and
      covers investigate aliases, explicit verifies, and frontiers; automatic
      post-motion verifies are summarized by their parent action. Alternate
      approaches, new views/evidence/geometry, and partial motion remain
      eligible. Manifest schema v4 freezes the new axis, rejects legacy-policy
      combinations, and validates matching summary/trace diagnostics before
      completion. Exact q11/q12 trace-derived replay fixtures plus the 182-test
      agent, 151-test lifecycle, and 134-test graph-memory CPU gates pass; see
      `docs/attempt_ledger.md#static-world-action-progress-policy`.
- [x] Finish reproducible paper tooling: expose the optional `latexmk` + TeX
      Live bundle through `emet install paper`, `install.sh --paper`, and the
      interactive install menu. The package simulation and 59 focused CLI tests
      pass; documented Docker fallback built the current 38-page appendix/PDF.
- [x] Run the managed static-policy comparison on `6,11,12,47` after CPU gates:
      shadow job `20260823_155106_470fe6` scored **3/4**, mean steps **35.25**,
      attempts **31**; enforce job `20260823_162608_ca49e0` scored **2/4**, mean
      steps **43.25**, attempts **39**. Shadow matched the prior action-history
      shadow baseline; enforce regressed on q47 and added dwell on q12 without
      explicit suppress dispositions. Mechanism passes; do **not** scale enforce.
      Frozen summary:
      `paper/data/hmeqa_agentic_h2h/action_progress_pair_20260823.json`.
      **Merge tooling** with defaults off (`action_progress_mode=off`).
- [ ] Generalize suppression for dynamic worlds before enabling it as lifelong
      memory. Target/environment change events, map revision, elapsed-time/TTL,
      and evidence staleness must invalidate or decay suppression. Consider a
      scheduler that cools down/re-ranks actions instead of removing them.
- [ ] Keep the bundled A0→A1→A2 ladder, `2,76`, A3, rooms-11, holdout, and
      bal-32 blocked until both process and letter gates pass.
- [ ] Keep q104 deferred until scale; it is a known native-crash hot scene.

## HM-EQA count/clock — FIND views, not node-list answers (PR #130, 7/15 = ceiling)

The instance graph **finds** candidate views; it is not the answer (no integer counts
from YoloE node lists). PR #130 (`feat/instance-graph-repair`) hardens graph admission
(`instance_min_confidence: 0.12`, `instance_min_mask_points: 25`), requires label-match
for instance fusion, marks `CONFIRMED_MEMORY` as **LOOK** image ids, pins FIND/Action
RGB as Image 1, attaches detector crops, exposes **VIEW_STATUS** per-Image counters,
and releases the controller stay when a FIND view is spent / Unknown without `read N`.

Frozen 15-qid slice (`12,21,28,32,33,43,47,48,51,60,78,84,86,88,93`,
`scripts/run_hmeqa_countclock_slice.sh`) — **HEAD `fb9fa75d`+VIEW_STATUS: 7/15**, equal
to the no-YoloE ceiling `2f4b4d4f`. q93 (bar stools) recovered via attached FIND RGB;
q47 stays OK; q28/q78 OK via crop-as-Image1. Merged 2026-08-25 (PR #130).

### Remaining gaps (documented in PR, not blockers)

- [ ] **Exploration/coverage**: q86 (table lamps) still finds only 1 of 2 lamps. The
      agent pins the FIND view and loops `read1`/`1` on it (up to 20 planning steps)
      instead of going after the second instance. q32/q48/q51/q60 similarly stop early
      or answer from a partial view. Need coverage-aware targets / second-instance search.
- [ ] **Close-look / legibility**: q33/q43/q84 ("What time is it now?") fail because the
      clock is *visible* but not *legible*. `read N` currently re-attaches the same full
      frame (no detector bbox for clocks ⇒ no crop). A center-zoom / upscale of the
      `read N` target view is the low-hanging next step.
- [ ] Consider a `read N` re-crop path: on `read N`, attach a tighter/zoomed version of
      that image (not just re-send the full frame), so written dials/answers become
      legible without navigating again.

## OVMM agentic find — PR #110 / #111 follow-ups (validated 2026-08-11)

Context: teleport-mode OVMM find on the shared AgenticEQA loop. PR #110 fixes nav
(sample_target_point projection, chunked-nav-as-progress); PR #111 (stacked) adds the
`NavOutcome` enum + question-type-aware verification + camera diagnostic. What's left:

**Branch `feat/tamp-ovmm-perf` (2026-08-23 → 2026-08-28):** OVMM-scoped routing — unified
`_recall_nav_hypotheses`, GT placement seeds, find_recep nearby-investigate bias,
`nav_outcome` in router Recent actions, richer `trace_meta` from find-phase harness.
**2026-08-28 unify mapping (this PR):** `run_mapping_protocol` with `explore_steps>0`
now calls `run_agentic_eqa_result(agent, None, goal="explore and map…", max_nav_steps=n, max_rounds=n+1)`
— same `AgenticEQAExecutor` `mode=explore` as HM-EQA (router only `explore_frontier`/`finish`),
coverage-first frontier picks (no object `toward`), arrival `look_ahead` (tilt 0) facing frontier
then `update` (not `look_front -30°`/pre-sweep), hop-until-arrival for 27-wp kitchens;
`S0` `explore_steps==0` stays rotate-only. Voxel `localize_text` on finished map now beats
camera-pose-at-feet cards (`CAMERA_POSE_PLACE` redirect), SigLIP on arrival RGB is the query.
**Efficiency pivot:** Stretch 3-ep slice took ~3.5 h with 0/3 success — default gate
is now **rby1** (`PROFILE=smoke` / `slice` in `run_ovmm_find_recep_slice.sh`);
`look_around` skips head pans on non-Stretch. Docs:
`docs/experiments/ovmm_agentic_find_teleport.md` + `docs/ovmm_find_phase_benchmark.md`.
**Integration (2026-08-25):** merged `feat/instance-graph-repair` (PR #130) for
clock/count FIND view pinning + `close_look_label`; extend investigate bias to
`find_object` and close-look questions on top.

### 2026-08-28 assessment — does unified mapping fix OVMM failures?

**Historical baseline (pre-unify):** teleport 9-ep sweep ~1/9 FindObj, 0/9 FindRec
(`recep_slice_20260823_172713` 0/3 in 3.5 h). Real-physics 9-ep ~1/3 obj, 1/6 recep.
Mapping was `run_mapping_protocol: spin + N×execute_action("")` (DynaMem multi-goal A*,
`look_around` before drive, `look_front -30°`) → voxel saw floor/sky, not kitchen; find
started a *new* `AgenticEQAExecutor` and scored wall-node cards; `localize_text("jar")`
often empty or floor. Stretch 27-wp kitchen paths truncated at 8 wps (never reached).

**Expected after unify (1–2 turn prompt sanity):**
*Turn 0* `inspect_graph` → hypotheses from `localize_text` on finished voxel map (or none) +
graph nodes/frontiers; prompt = `Rooms: …` + merged `SCENE_GRAPH` (`CONFIRMED_MEMORY` folded,
no dupe), `Recent actions: … nav_outcome=…`, stable allowlists. No object `toward` during
mapping, so frontier picks stay uncovered-first.
*Turn 1* `explore_frontier` → `target_theta` toward frontier, `look_ahead` 0° at arrival,
`capture_and_update` → new observation → SigLIP points on that frontier surface enter voxel.
If a jar/cab was in that frustum, `localize_text` next turn produces `proposal` card
`obs_id<-3M` that **beats** any `CAMERA_POSE_PLACE` view.
*Turn 2* find: `inspect_graph` on finished map now has SigLIP-backed proposal; router
should `investigate(proposal)` before `explore`, `verify_siglip` on that view; one
`explore` only if `ABSENT` (not 150 wall cards). Prompt after 1–2 turns should list
`detections: 1+` with `source=voxel` and `views` separate, `visible_frontier_ids` shrinking.

**Verdict:** unify addresses the load-bearing mapping bug (coverage vs object-biased
`toward`, arrival tilt, hop-until-arrival). Voxel-first find + `CAMERA_POSE_PLACE`
redirect addresses the scoring bug. It **does not** by itself fix the “recep loop
explores away” targeting problem — that is still a search/ranking issue (see below).
Needs a GPU `rby1` smoke + `stretch-kitchen` to confirm `localize_text non-null`
and that `n_explore` now increments `mapping_n_explore` in JSON.

- [x] **Unify OVMM mapping with agentic explore (2026-08-28):** `run_mapping_protocol` for
      `explore_steps>0` now uses `mode=explore` coverage loop; arrival `look_ahead` facing
      frontier; tests for mapping entrypoint/arrival look/voxel-first; docs updated.
- [x] **Object targeting fixes (2026-08-28, live rby1 S0 + kitchen):** three systematic
      find-loop bugs found and fixed on the **shared** AgenticEQAExecutor:
  1. **SigLIP released across find phases** (`agentic/answer.py:_do_submit_answer` →
     `release_siglip_for_vlm` drops `voxel_map.encoder`; nothing re-attached it). FindRec
     (second phase) could never `localize_text`. Fixed: `re_attach_siglip_encoder` wired into
     `warm_siglip_confirmed_memory` (`eval/dynagraph_vram.py`), so each phase localizes on
     the finished map. Rby1 S0 FindRec went from `recep_localize_source: None` → voxel err 0.0.
  2. **Voxel proposals re-chased after close ABSENT** — the VLM router re-emits
     `investigate(obs_id<0)` on the same wall point (`-3000000` visits 2..N). Fixed: proposals
     are one-shot in `_hypothesis_nav_blocked` (`agentic/capture.py`), not just
     `_unused_detection_hypothesis` — S0 obj loop went from 3× chases to 1×.
  3. **Disproven voxel pins were still scored** — a close ABSENT leaves the pin; the harness
     `pinned_xyz_from_phrases` fallback scored the wall point. Fixed: `unpin_localize_xyz`
     (`mapping/voxel_localize.py`) called from `_maybe_retract_claim_after_station` on ABSENT,
     plus clearing the loop-scored voxel record.
  **Measured (rby1 S0 `default_table_rby1_s0_distinct_recep`, 5 runs): FindObj 5/5 (voxel,
  err 0.0); FindRec 3/5 (voxel err 0.0 when found).** FindObj is stable; FindRec is SigLIP
  marginality on the small blue cube (YOLO labels in this scene are garbage:
  box/sign/tv/monitor/divider — graph recall is useless; voxel is the only reliable path).
  Mixed gate still validates EQA: countclock **7/15 = gateAB** (no regression from the
  shared-loop changes).
- [x] **Kitchen explore stall (2026-08-28):** live agentic mapping still creeps (8–20 hops
      cover only ~3 m²; the nearest-uncovered frontier clamps to ~0 and the loop re-picked
      the same frontier). Added a no-progress goal block in `_tool_explore_frontier`
      (`agentic/explore.py`): after a nav that moved < 0.10 m, block the FRONTIER XY in
      `_habitat_recent_goals`/`_habitat_blocked_goals` so the next pick chooses a different
      frontier or falls through to multi-goal explore. Frontier selection now rotates
      (verified live). Kitchen find still 0/2 — coverage volume, not the loop.
- [x] **SigLIP re-attach scoped to OVMM (2026-08-28):** re-attach was briefly in the shared
      `warm_siglip_confirmed_memory`; HM-EQA countclock q47 flipped False, so it is now
      called only from the OVMM harness `run_ovmm_agentic_localize` (before each phase) —
      HM-EQA keeps its released-SigLIP behavior.
- [x] **One-shot proposals refined to close-ABSENT (2026-08-28):** the first one-shot gate
      blocked a proposal after *any* one nav; countclock dropped to 5/15 (q21/q47/q93
      flips). Refined: a proposal is blocked only after a **close ABSENT** recorded on that
      card (`_hypothesis_nav_blocked` + `_unused_detection_hypothesis` now both check
      `_place_inspect.last_verify == ABSENT`), so HM-EQA count/locate targets stay
      re-approachable from a new bearing while OVMM wall-chases stay one-shot. **Countclock
      back to 7/15 = gateAB** (q21/43 recovered; q93 True solo — variance, not systematic).
- [x] **Cached-map robocasa find (2026-08-30, stretch-kitchen):** with the re-attach +
      one-shot + unpin fixes, the cached stretch map localizes the **jar (FindObj 1/1,
      voxel err 0.176 m)**; **FindRec (cab) 0/1** — the VLM never saw the cab (navigated to
      microwave/fridge). Live-mapping robocasa scored 0/2 (small-object jar SigLIP miss) and
      took 3.1 h/ep, so the paper S1 column uses the cached-map path. **Paper OVMM S0:
      obj 5/5, recep 3/5; S1 (cached): obj 1/1, recep 0/1; S2 (molmo): pending.**
- [x] **fp16-vs-int4 VLM ablation (2026-08-29→30):** countclock fp16 **10/15 vs 6/15**;
      30-qid clean same-harness **fp16 16/30 vs int4 9/30 (+7, +78%)** — quantization is a
      real, task-dependent lever (fine-detail count/clock). Records in
      `paper/data/vlm_quantization/` + `docs/experiments/vlm_quantization.md`; appendix
      `06_model_choice.tex` `tab:vlm_precision` + discussion (PR #148).
- [x] **TAMP signal run (2026-08-30, 25-ep subset):** battery 8/8; cleanup **9/10** across
      4 robots; nav_goal cleared all clutter but the terminal straight-line teleport chord
      hit furniture (refrigerator) → 0 reached. Fixed in PR #154: post-clear route now uses
      an **8-connected planner probe** (`nav_path_open_around_disks`) matching the GT
      validity definition.
- [x] **TAMP nav_goal landmarks were unwinnable (2026-08-31):** even with the 8-connected
      post-clear probe, all 15 nav_goal rows scored 0 because `_goal_for_landmark` picked
      by *straight-line* distance (< 10 m), not navigability — landmarks sat behind
      furniture barriers. Fixed (`02be11d0`, PR #154): landmark selection now requires an
      **8-connected route from the robot start** to the approach point (farthest navigable
      preferred). **Full 15-row rerun: 5/15 success** (rby1 scene-0 3/3, nori scene-0 2/3);
      remaining failures: stretch `path_open=True` but teleport doesn't land (nav-execution),
      innate_mars + rby1 scene-1 `path_open=False` (post-clear route from the bin blocked by
      dense furniture). Two documented next-step fixes.
- [ ] **Recep/object targeting residual — small-object SigLIP marginality.** FindRec (blue cube)
      is 2/4 because the cube's SigLIP cosine hovers at the localize bar (top_sim ≈ 0.10–0.14;
      `localize_text` threshold 0.14). When the map lacks good blue-cube points, no pin forms.
      Levers: (a) lower the voxel localize bar for find (risky false positives; one-shot
      proposals now make ABSENT cheap), (b) a closer start-recep look so YOLO/SigLIP see the
      object, (c) accept variance and report voxel err distribution.
- [ ] **Recep loop explores away from the target, never converges.** "Where is the table?"
      runs all 8 rounds with `nav=0..2 explore=N`; the router keeps picking
      explore_frontier and the assess returns not-present (table not in those views).
      Obj loop now verifies in 2 rounds, so the verify gate + camera are fine — this is a
      **targeting/search problem**. Next steps:
  - [ ] Debug why investigate cards for "table" are absent/ranked low: `hypothesize_nav_targets`
        (graph_memory.py:2689) recall for receptacle labels; dump the card list the router saw
        (`_investigate_hypotheses` at agentic_eqa.py:1015) per recep round.
  - [ ] Check `_prefer_explore` / `_not_present_streak` / `_update_escape_streak` loop: close-look
        ABSENT nudges explore (agentic_eqa.py:2593), which may pull the robot away from the start
        table instead of toward it. Add an investigate bias when the question target is near the
        robot's start receptable.
  - [ ] Consider a distance-gated investigate fallback: if a place card is within `X` m and
        unvisited, prefer `investigate` over `explore_frontier` even when the router leans explore.
- [ ] **Open-question verification semantics are now present-only** (agentic_eqa.py:2441 `open_view_present`).
      Confirm this doesn't over-trust a single glance for location questions where the target is
      visible but mis-localized — verify the verified-obs XYZ error distribution in the next
      full OVMM run (teleport) against GT.
- [ ] **Full teleport OVMM find validation run** (all 9 episodes, `EMET_SIM_NAV_TELEPORT=1`) on
      `feat/nav-outcome-enum` after the recep-targeting fix: want find_object + find_recep rates
      vs the real-physics baseline (1/3 obj, 1/6 recep) and the pre-fix teleport run (1/9 obj, 0/9 recep).
- [ ] **Camera diagnostic** `scripts/debug_ovmm_camera_quality.py`: confirmed no all-black frames
      (mean≈122, std≈93, black_frac=0). Depth stream reads 0 because the diagnostic client is built
      with `allow_missing_depth`; for a true blocked-camera depth check, request depth on the client
      (drop `allow_missing_depth` or set `EMET_ZMQ_FULL_HZ` depth track). Ray trace (`mj_ray`) needs
      model/data on the client — either expose it or run the ray check inside the sim server process.
- [ ] **Slowness**: teleport base nav is ~4s/step (vs ~50s real physics) but each episode is still
      ~30-50 min because (a) every investigate does a full 4-pan head sweep (~40-80s) and (b) two
      agentic loops per episode (obj + recep). Options: `_fast_explore_lookaround` (2-pan) in the
      OVMM find path like run_dynagraph does, and cap recep rounds when the recep is the start
      receptable (already known).
- [x] **NavOutcome in router history**: typed recent actions now render the
      motion enum alongside semantic target/outcome/progress, so the VLM can
      distinguish reached vs progress vs blocked.
- [ ] **NavOutcome in durable nav ledger**: propagate the enum through
      `NavAttemptResult` / `graph_memory.record_nav_attempt` instead of relying
      on status/note reconstruction in offline artifacts.

## Habitat find proxy — object_relation versioning (PR #170)

`configs/ovmm/habitat_find_phase_nearest_v2.yaml` (`object_relation: nearest`) is
versioned next to the legacy `on` manifest; unknown relations fail closed. The
relation field is threaded through FindObj question phrasing and result/trace
metadata only — it does not branch the scorer.

- [ ] **Implement support-relation (`on`) GT selection distinct from `nearest`.**
      `pick_find_object_gt_body` disambiguates multiple object-category bodies by
      nearest `start_recep` for **both** relations, so legacy `on` episodes are
      scored with nearest-recep proximity rather than an actual support ("on")
      relation. Until that lands, `on` and `nearest` scores are distinguishable
      only by the `object_relation` metadata tag — do not pool them.

## Next experiments + tuning (2026-08-31)

Priorities after PR #148 merged (OVMM find, fp16 analysis, TAMP signal) and PR #154
(TAMP nav_goal 8-connected fix). **EQA is the strong result; OVMM/TAMP are exploratory
pilots — treat as preliminary until a strong pilot lands (like EQA's).** Four axes:

### 0. OVMM pilot status (preliminary — not paper-commit yet)
- [ ] **Pattern: FindObj works, FindRec (fixtures) fails.** S0 obj 5/5, recep 3/5; S1
      (cached robocasa) obj 1/1 jar, recep 0/1 cab; S2 (cached molmo) obj 1/1 bowl, recep
      0/1 microwave. Small objects localize via voxel; fixture receptacles (cab/cabinet,
      microwave) are not surfaced by recall (multi-label dilution) — a recall/ranking gap.
- [ ] **FindRec recall WORKS — the failure is investigate geometry (2026-09-01 debug).**
      `EMET_DYNAMEM_MAP_DEBUG` on the cached robocasa recep showed the loop recalled 5
      cabinet cards (`42:cab,44:cab,69:cab,71:cab,15:cab`) and investigated all 5, but every
      arrival view was "kitchen/toaster" — the **cached graph node XYZs are camera-observation
      poses, not object positions** (`_obs_nav_anchor` returns `node.xyz`), so the robot
      approaches where a camera once stood and sees a toaster/counter, not the cabinet. Same
      artifact class as the jar-wall proposal.
- [ ] **FindRec phrase-expansion was negative (2026-09-01):** expanding "cab" → "cabinet"/
      "kitchen cabinet" (from matched graph-node labels) did not change the result —
      `recep_localize_source` stayed None. Probe of the cached voxel map: 8002 points across
      the kitchen but **no cabinet-feature points** (the recep's new captures also showed
      kitchen/toaster), so there is nothing for localize_text to ground. **FindRec on the
      cached map needs the cache rebuilt with correct object anchors** (live mapping must
      first produce good coverage + geometry) — defer while OVMM is exploratory. The
      phrase-expansion code is kept (principled; helps other terse queries).
- [ ] Explore OVMM as pilots only: backend matrix, OVMM full (pick/place), Habitat-OVMM
      are deferred until FindRec-recall improves OR the paper explicitly reports FindObj-only
      with the fixture gap documented. Do not sink GPU into the full matrix yet.

### 1. More experiments (fill the paper tables)
- [ ] **HM-EQA paper-113 fp16 (full)** on caliban — we have the 30-qid subset (16/30 vs
      9/30 int4); the full 113 with fp16 replaces/extends `tab:hmeqa_vs_prior` (49.6% int4).
      ~20–28 h serialized; run after the quick wins below.
- [ ] **OVMM S2 (molmo) column** — `tab:ovmm_find_backend_tier` S2 is pending. Use the
      cached `molmo_ithor_train_idx0_stretch_gt` map + agentic find (like the robocasa S1
      cached path that found the jar). rby1 or stretch.
- [ ] **TAMP full 200-registry** (E4) after the nav_goal fix lands — fills cleanup S2 +
      nav_goal rows for `tab:tamp_clutter` (S1 signal already in).

### 2. Improve failing cases
- [ ] **FindRec (fixture receptacles) on OVMM** — cab/cabinet, microwave are not surfaced by
      `hypothesize_nav_targets` recall (nodes carry 20+ multi-labels diluting the fixture
      signal). This is the FindObj-vs-FindRec gap across S0/S1/S2; fix fixture-node ranking
      for recep questions. Highest-value OVMM lever.
- [ ] **Clock/count close-look** — q33/q43/q84 fail because the clock is visible but not
      legible (`read N` re-attaches the full frame, no detector bbox for clocks). Implement
      the `read N` re-crop / center-zoom so dials become legible without re-navigating
      (TODO § count/clock "Close-look / legibility").
- [ ] **Small-object SigLIP marginality** (blue cube, jar) — levers: lower the localize bar
      for find (cheap now that ABSENT is one-shot), or a closer start-recep look so
      YOLO/SigLIP see the object.

### 3. Does more budget help?
- [ ] **Explore-steps sweep on OVMM kitchen** — 8 → 20/40 explore steps: does mapping
      coverage / FindObj improve (earlier: 8 steps ≈ 3 m², 0/2)? Isolate coverage-volume vs
      detection.
- [ ] **Nav/round budget sweep on countclock + paper-113** — more `max_nav_steps` /
      `max_rounds` for the agentic loop; measure accuracy-vs-budget curve (mean planning
      steps already logged).
- [ ] **fp16 full-113 vs budget** — does the fp16 gain grow with more rounds?

### 4. Make tamp / molmospaces / habitat-ovmm work well
- [ ] **TAMP E1 battery re-run** (after chord-sample + 8-connected nav + navigable-landmark
      fix) + **E3 small** + **nav_goal 15-row rerun** (scene-0 winnable now) + **fill
      `tab:tamp_clutter`** from `aggregate_tamp_clutter.csv` (exclude `skipped_invalid`).
- [ ] **TAMP → agent**: expose `plan_clear_clutter` as a `clear_clutter` CHAT skill so the
      LLM agent can parse "clean up the room" / "get to the sofa" end-to-end (the real
      product blocker; MCTS pick/place already works).
- [ ] **MolmoSpaces smokes**: `molmo_ithor_rby1_s2_bowl_pp` (manip=sim) reconfirmed; run
      `scripted_tamp_pick_place` + rby1 iTHOR kinematic smokes when `.venv-molmospaces` warm.
- [ ] **habitat-ovmm**: validate the robocasa cached-map find + rby1 kitchen (explore_steps
      20 live) end-to-end via the mixed gate; report S0/S1/S2.

**Status of running jobs:** TAMP nav_goal 8-connected + navigable-landmark rerun (15 rows)
pending — scene-0 confirmed winnable; TAMP battery 8/8 (pre-fix, re-run needed); OVMM
molmo-robocasa live sweep cancelled (0/2 robocasa, 3.1 h/ep — use cached-map path for paper).

## Embodied agent planning (world model + tool calling + motion)

Branch `feature/agent-world-model`. Phases 1–3 + Phase 4 helpers are **landed**
(ledger default **off**). Do not touch pinned HM-EQA/OVMM configs.

| Doc | Role |
|-----|------|
| [docs/attempt_ledger.md](docs/attempt_ledger.md) | **Shipped** operator/dev reference (enable, schema, writers, tests) |
| [docs/plans/2026-08-08_embodied_agent_planning.md](docs/plans/2026-08-08_embodied_agent_planning.md) | Design history + phase checklist |

### Phase 1a — nav ledger store
- [x] `AttemptRecord` store on `GraphEQAMemory` (`attempt_ledger.py`); `record_nav_attempt` dual-writes when `eqa.attempt_ledger` / `EMET_EQA_ATTEMPT_LEDGER` on (default **off**); export/import + derive helpers; tests in `src/test/memory/test_attempt_ledger.py`.

### Phase 1b — ABSENT persistence
- [x] Config-gated persistence of verify-ABSENT evidence past the per-question `_retracted_nav_claims` reset (`persist_absent_claims` / `EMET_ATTEMPT_LEDGER_PERSIST_ABSENT`; retract also writes a `verify:absent` ledger row).

### Phase 1c — manip outcomes
- [x] CHAT executor pickup/place failures write to the ledger via `ToolOutcome`.
- [x] Stretch / AnyGrasp `_pickup` / `_place` propagate real success (`agent.manipulate` / `agent.place` / `GraspObjectOperation.was_successful`).

### Phase 1d — surface to planners
- [x] Enriched `[attempts: …]` place cards + CHAT `navigation_diagnostics` recent attempts + CONFIRMED_MEMORY `attempts:` tags.

### Phase 2 — shared tool-outcome schema
- [x] `ToolOutcome` shape shared by CHAT `_dispatch_tool_calls` and EQA `handle_tool` (`emet.agent.tool_outcome`); both write to the ledger.
- [x] Router prompt hygiene: shared `_EQA_RULE_*` atoms compose `_EQA_FORMAT_BLOCK_*` + byte-stability test in `test_room_policy.py`.

### Phase 3 — motion planning behind a narrow interface
- [x] Structured `NavAttemptResult.status_code` + `emet.controller.nav_attempt.sync_nav_attempt_to_ledger` from `_log_nav_attempt` (feeds ledger + `_last_nav_plan`).
- [x] `aim_arm_at` → `closer_look.aim_wrist_at_phrase` (localize + kinematic EE aim when available; structured failure codes otherwise).
- [x] OVMM-full pick/place path records attempts to the ledger (`record_manip_attempt` in `ovmm_full.py`).

### Phase 4 — eval (no regressions)
- [x] Repeat-failure metrics helpers (`attempt_metrics.summarize_repeat_failures`) + unit tests.
- [ ] GPU deltas vs pinned baselines on HM-EQA agentic arm + OVMM find (via `emet jobs`; ledger opt-in). Report repeat-nav-failure / wasted-rounds deltas; never by flipping pinned configs.

### Persistent interaction memory (proposed 2026-11-11)

See [the interaction-memory plan](docs/plans/2026-11-11_interaction_memory.md) for contracts and acceptance gates.

- [ ] M1: link attempts to bounded action-window views; versioned persistence, restart/reset isolation, eviction and legacy-import fixtures.
- [ ] M2: shadow interaction analyzer with uncertain provenance-backed claims; held-out failed-grasp/locked/occluded/change diagnostics.
- [ ] M3: entity/interaction/temporal retrieval with per-source budgets, packet traces, old-evidence recall and same-label instance tests.
- [ ] M4: independently gated dynamic invalidation/retry scheduling; never renew physical evidence or let failed attempts unlock themselves.
- [ ] Freeze model × memory comparisons (Qwen/GPT), complete-sequence task splits, assistance tiers, budgets and promotion margins before live runs.

### Docs (this branch)
- [x] Canonical [docs/attempt_ledger.md](docs/attempt_ledger.md) + links from README doc map, `AGENT_RUN.md`, `graph_eqa.md`, `dynagraph.md`, `evaluation.md`, `emet_config.md`, `environment_variables.md`, `motion_planning.md`, `ovmm_full_benchmark.md`, plan index.

### Follow-ups (not blocking merge of ledger v1)
- [ ] Dual-write exit: once planners prefer the ledger, retire loop-local `_tried` / `_place_inspect` / `_nav_loop_flags` (or make them views). See plan Phase 1a.
- [x] Closer-look polish: `take_ee_picture` requires a successful `aim_arm_at` grant (consumed on capture); prompt + skill specs updated.
- [ ] Mars/Stretch wrist-frame + collision defaults dogfood on hardware (real EE stream after aim).
- [ ] Mobile-manip eval sketch: OVMM full + ledger metrics (pick attempts per success, re-grasp count) once orientation IK lands.

## Prompt / information flow

- [x] **One fact, one line — de-duplicate EQA memory blocks**: folded CONFIRMED_MEMORY → SCENE_GRAPH is now the **default** (`eqa.merged_memory` on; env can flip). HM-EQA paper row pins `merged_memory: false` in `configs/benchmarks/dynagraph.yaml` to keep published numbers on the standalone block. IMAGE_DESCRIPTIONS (when enabled) omit labels already tagged on SCENE_GRAPH Image-N lines.
- [x] **Room context never reaches the answer VLM**: done — `to_string` emits `Rooms:` + per-node `(room)` tags and `query_answer` passes them to the answer model.
- [x] **EQA answer format → JSON**: HM-EQA/MCQ use `{"reasoning","answer","confidence","action","confidence_reasoning"}` (`eqa.answer_format` / `EMET_EQA_ANSWER_FORMAT`; default json for hmeqa/mcq). `parse_answer` prefers JSON with labeled-field fallback; prefill is `{"reasoning":`.
- [x] **Unified token budget for the EQA prompt**: `eqa_vl.eqa_prompt_max_tokens` default 2500 (`EMET_EQA_PROMPT_MAX_TOKENS`); truncation order HISTORY → CONFIRMED_MEMORY → edges → labels via `build_eqa_prompt_text`.
- [x] **Router prompt hygiene**: shared `_EQA_RULE_*` atoms in `agentic_tools.py`; byte-stability test pins format-block SHA256 + identity across calls.
- [x] **HISTORY loop risk**: HISTORY stores one-line outcomes (`Iter: answer=… conf=… action=… salvage=… | reason`) plus `Nav_result`, not raw model replays.
- [x] **Prompt after 1–2 turns (agentic):** after `inspect_graph` + 1×`explore_frontier(look_ahead→capture)` the state
      message is `Rooms: …` + merged `SCENE_GRAPH` (no dupe `CONFIRMED_MEMORY`), `Recent actions` with
      `nav_outcome, target_theta, room_aligned`, stable allowlists
      `place_ids/place_obs_ids/frontier_ids` via `action_gate`, typed `HISTORY` lines and
      `visible_event_ids`. Verified 2026-08-28 on `AgenticEQAExecutor` `mode=explore` mock: `target_theta` toward frontier,
      `look_ahead` before `capture`, `detections` vs `views` split in `inspect_graph`.
- [ ] **CHAT `_FORMAT_BLOCK` is ~90 lines of routing edge cases** (prompt.py): consider tiered prompt (short default; detailed hints appended only for 4B-class routers) and measure system-prompt chars/tokens with and without hints.
- [ ] **describe_scene grounding**: currently caption + optional graph labels appended ad hoc (`describe_head_camera_scene_text`, controller_dynamem.py:1049). Define one consistent grounding format shared with `query_scene_graph` so the chat VLM sees the same memory vocabulary as EQA.

## LazyGraph memory backend (PR #131 — merged 2026-08-25)

- [x] Land **`lazy_graph`** backend: Dynamem voxel find + viewpoints during `update()`, Qwen commit on arrival only (`emet run lazy-graph`, `--memory-backend lazy_graph`). Does not change `dynagraph` defaults.
- [ ] Robocasa smoke: `emet run lazy-graph` with sim server (or `scripts/queue_lazy_graph_robocasa_smoke.sh` via `emet jobs`). Job `20260825_163104_886a12` queued 2026-08-25.
- [x] Herman hardware smoke: `emet run agent --memory-backend lazy_graph --connection herman --host caliban` loads lazy_graph + remote VL (2026-08-25; wrist still down).

## Herman / Orin offload (branch `feature/herman-orin`)

Bridge on Herman (192.168.1.43) is live via `emet mars start --connection herman --deploy`; head stereo OK, wrist Arducam still unplugged.

**What can run on the Orin today (documented paths):**

| Component | Where | Notes |
|-----------|--------|--------|
| ROS bridge + ZMQ | Herman Jetson (innate-os) | `emet deploy` / `emet mars start` |
| DA3 depth (SMALL) | Herman Jetson | `--onboard-da3` on bridge; `depth_source: auto` skips workstation DA3 |
| Qwen2-VL-7B text+VL | LAN Orin (e.g. caliban) | `emet deploy llm --profile unified-7b`; agent `--host caliban` |
| Nav2 / drivers | Herman Jetson | innate-os (not emet) |

**Still workstation-only (no Jetson deploy path yet):**

- SigLIP, YoloE, DINOv3 encoders (`emet/perception/encoders/`)
- Voxel map + Dynagraph/LazyGraph memory loop
- Full `emet run agent` stack when not using remote `--host` VL

- [ ] **Onboard DA3 dogfood**: `emet mars start --connection herman --deploy --onboard-da3`; confirm ZMQ depth + `emet stream --backend voxel_only`. **Blocked 2026-08-25:** Herman pip install fails — `depth-anything-3` needs `pycolmap` (no Linux aarch64 wheel). Workstation DA3 still used via `depth_source: auto`. Bridge code synced; retry when pycolmap/Tegra wheel exists.
- [x] **Remote VL smoke**: `emet llm health --host caliban` + `emet run chat --host caliban --once` → `Pong` via caliban:8000 (2026-08-25).
- [ ] **DINOv3 on Orin**: caliban deploy path added (`docker/jetson_dinov3_server.py`, `scripts/deploy_caliban_dinov3.sh`, `EMET_DINOV3_ENDPOINT` client + tests). Run `./scripts/deploy_caliban_dinov3.sh --host caliban` after caching `facebook/dinov3-vits16-pretrain-lvd1689m`.
- [ ] **Wrist camera**: reseat Arducam USB on Herman; SSH verified no `/dev/v4l/by-id/*Arducam*` (2026-08-25).

## ZMQ protocol / comms (Herman + sim)

Mars bridge publishes **`/scan`** as `lidar_points` + `lidar_timestamp` (float32 N×2). Slim image wire format (`zmq_obs_slim`) drops duplicate JPEG aliases. Sustained benchmark: `scripts/benchmark_zmq_obs_payload.py` (one-shot sizes) and `scripts/benchmark_zmq_obs_stream.py` (fps/jitter/unpickle).

- [x] **LiDAR in ZMQ contract**: `innate_mars_bridge` subscribes `/scan`, publishes `lidar_points` + `lidar_timestamp`; `zmq_lidar_f32` capability when float32 slim is active ([`docs/zmq_session_metadata.md`](docs/zmq_session_metadata.md), [`BRIDGE_CONTRACT.md`](src/emet_core/BRIDGE_CONTRACT.md)). Audit Herman topic name/frame with `ros2 topic info /scan` when stack is up.
- [ ] **LiDAR → mapping**: fuse scan into 2D explored/obstacle layer (voxel `TODO` at `voxel.py` ~1001); avoid naive double-count with DA3 3D cloud — start with base-height obstacle mask only.
- [x] **Sim LiDAR (innate_mars)**: MuJoCo `base_lidar*` → `lidar_points` via `mujoco_lidar.py` (360 rays, float32).
- [x] **Profile ZMQ payload**: `scripts/profile_zmq_obs_payload.py` + `scripts/benchmark_zmq_obs_stream.py` — bytes/step, fps, jitter, unpickle/decode cost; flags duplicate JPEG aliases and float64 lidar.
- [x] **Slim messaging v1**: one JPEG per logical camera (`zmq_obs_slim`); `lidar_points` as float32 (`zmq_lidar_f32`). v2 (subscribed streams only) still optional — document schema bump in `zmq_session_metadata.md` when needed.


- [x] **One open-vocab scene graph (not two builders)**: CHAT uses mutually exclusive `agent.memory_backend` (`dynagraph` | `graph_eqa` | `open_vocab` | `dynamem` | `lazy_graph`). Discord presets → Dynagraph memory plug-in only; GraphEQA baseline left frozen for paper. Lifelong save/load writes the active plug-in only.
- [x] **Arm IK “closer look” (v1)**: CHAT `aim_arm_at` → `closer_look.aim_wrist_at_phrase` (localize + kinematic EE aim when available). Follow-ups under Embodied agent planning (EE picture after aim; hardware dogfood).
  - Fallback when aim unavailable: “closer look / inspect X” → **change viewpoint** (rotate / small drive) then `describe_scene` / `send_image` (head), never raw `take_ee_picture`.
- [x] **Chat verify ≈ EQA look**: prompt + tools route “look at / closer look / are you sure” → `face_toward` then `describe_scene` (not blind ±45°); tests in `test_agent_prompt_and_tools` / `test_face_toward`. Full EQA-style `navigate_to_obs` + `verify_siglip` in CHAT still optional later.
- [x] **`emet run` + `--connection`**: wrapper no longer injects default `--robot-ip 127.0.0.1`, so `--connection herman` resolves the profile host.
- [ ] **Interruptible explore**: Discord messages queue until `explore` finishes; drain `unified_input_queue` mid-nav.
- [ ] **VRAM for Discord house chat**: default Mars preset loads tool LLM + EQA 8B VL + SigLIP (+ DA3). Easy OOM on 24 GiB when map update / DA3 overlaps generate. Mitigations: `--onboard-da3`, share one VL (`--llm qwen3-vl-eqa --share-memory-vllm`), or lazy-unload captioner between turns. (Dual OV+GE builders removed; still watch two VLMs.)

## Mapping / safety

- [x] Map-clip `move_forward` (including 0.1 m); refuse when map empty/blank.
- [x] Empty cloud guard in `list_objects_in_an_image` (navigate crash).
- [x] Clearance-aware A* + abort on waypoint timeout (prefer open space; do not raise `dilate_obstacle_size` as the primary fix).

## Memory stack (voxel + graph) — review backlog

- [ ] **Drift correction / relocalization** (long real-world runs): voxel map is robot-start-relative with a fixed `grid_origin`; only `--refine-start` applies an SE(2) fudge at checkpoint load. Plan ICP / pose-graph correction against the voxel map or re-anchor graph nodes during operation.
- [ ] **Pin depth units per source + runtime assert**: `sensor_graph_builder.py:80` assumes meters (`scaling=1.0`) while `core/interfaces.py:196` defaults to `1e-3` (mm). Verify per source (Stretch d405, DA3, Habitat) and assert once at attach.
- [ ] **Robot self-filter lost in DynaMem `add()`**: `voxel_dynamem.py:1109` copies base `voxel.py:403` minus the URDF mesh self-filter — the robot may see itself on real hardware. Reintroduce or document why it is off.
- [ ] **Split `query_answer` + de-dup renumber/rebuild blocks**: `build_eqa_prompt_text` is extracted (#104); `query_answer` is still ~477 lines — still need `run_eqa_prompt(commands)` / `finalize_eqa_answer(parsed)`. The renumber+rebuild block in `maintain`/`_drop_nodes_near`/`absorb_object_node` is copy-pasted 5×.
- [ ] **Rooms as first-class nodes**: runtime `RoomCluster` + prompt `Rooms:` / `(room)` tags exist; still recompute every refresh. Persist room id / name / bounds on nodes and in `graph.json` exports.
- [x] **`room_clustering/` + `partition()` + `proximity`**: naive `near` + XY radius; `room_clusters.py` is the naming/stamp facade.
- [ ] **`occupancy_cc` room backend**: flood-fill free/explored cells on the voxel 2D map; assign instance nodes to occupancy CCs (respects mapped walls).
- [ ] **`portal` room backend**: occupancy CCs cut at narrow passages / doors.
- [ ] **Room clustering backend sweep** once a second geometry backend exists (`eqa.room_clustering.backend` / `EMET_EQA_ROOM_CLUSTERING_BACKEND`). Do not mix with OVMM S0 instance labeling.
- [ ] **Bound voxel memory growth**: `observations` never pruned, `semantic_memory` keeps every subsampled point, pickles dump everything. Trim frames + downsample old semantic points on a schedule.
- [ ] **Prefix-KV timeout leaves a live CUDA worker** (`qwen3_vl_client.py:549-569` raises while the generate thread keeps running): next generate can race. Serialize on a lock or hard-kill the worker.
- [ ] **Silent exception hygiene**: ~23 `except Exception` in graph_memory.py — at least `_logger.debug` with node/obs context. (Bare `except:` in `voxel_dynamem.list_objects_in_an_image` now logs and retries.)
- [x] **Dead code sweep**: removed unused `Qwen3VLClient = Qwen35VLClient` alias from `qwen_client.py` and discarded Candidate 1/2 comment blocks in `voxel_map_dynamem.py`. (`get_change_events` / `alternate_nav_target_for_failed_action` remain in use.)

## Docs / ops

- [x] Document Herman Discord happy path: `innate_mars_hardware.md` Discord section covers `EMET_BASE_ROTATE_ONLY` + `EMET_ALLOW_SDPA_ATTN` / flash-attn with a tethered copy-paste env recipe.
- [x] Action-outcome ledger docs for `feature/agent-world-model`: [docs/attempt_ledger.md](docs/attempt_ledger.md) (see Embodied agent planning § Docs).

## TAMP clutter benchmark + Nori A3 (follow-ups)

Benchmark: [docs/experiments/tamp_clutter.md](docs/experiments/tamp_clutter.md) ·
`scripts/eval_tamp_clutter.py` · `scripts/generate_tamp_clutter_registry.py` (200 episodes,
4 robots: rby1 / stretch / innate_mars / nori). Nori backend: [docs/robots/nori.md](docs/robots/nori.md).
GT+MCTS battery: [docs/experiments/tamp_clutter_testing.md](docs/experiments/tamp_clutter_testing.md).

**Eval code is merge-ready** (blocked-nav scoring, chord-sampled no-snap, reachable
landmarks, sim default manip for mars/nori). **#150** / **#151** already on `main`.
Remaining items are **GPU experiments / product**, not code blockers. Operator table:
[docs/experiments/tamp_clutter.md](docs/experiments/tamp_clutter.md) Remaining experiments
(E1 battery → E5 latch → E3 small YAML → E4 large registry → fill `tab:tamp_clutter`).

- [x] **GT+MCTS battery 24/24** (2026-08-28): pickplace / declutter / navblocked / navclear
      pass for nori, innate_mars, rby1 × iTHOR scenes 0–1, sim-oracle manip, zero AI models.
      That run predates chord-collision + reachable-landmark; re-run below before citing it.
- [x] **GPU GT+MCTS battery re-run** (2026-09-05, three robots × scenes 0–1; 21/24 pass).
      Furniture on
      the spawn→landmark chord can now fail `navclear`. Queue, do not run inline:

      ```
      NEED_MIB=8000 uv run emet jobs run --name tamp-gt-battery --need-mib 8000 -- \
        uv run python scripts/eval_tamp_clutter.py --test-battery \
        --battery-robots nori --battery-scenes 0,1
      ```

      The three-robot run is complete at `logs/tamp_validation_20260905/gt_mcts_battery/`;
      three Innate Mars navigation rows remain tracked below.
- [ ] **Investigate Innate Mars nav-goal failures (validation 2026-09-05).** The fresh
      24-case sim-oracle battery scored **21/24**: all cleanup rows passed, and all Nori/rby1
      rows passed; Innate Mars failed `navblocked_s0` (cleared 8/8, GT probe blocked, final
      goal not reached), `navclear_s0` (same sampled cabinet goal not reached with no clutter),
      and `navblocked_s1` (cleared 8/8 but the GT probe classified the ring as unblocked, so
      the row is invalid). This isolates a Mars navigation/landmark or scene-validity issue,
      not manipulation: `n_relocated` is correct in every failed blocked row and there are no
      episode errors. Reproduce from the saved artifacts at
      `logs/tamp_validation_20260905/gt_mcts_battery/`; compare Mars `move_base_to` / achieved
      pose with rby1 on the same scene and fix landmark sampling or Mars teleport/nav frame
      handling before citing the three-robot battery.
- [ ] **Investigate Sourccey RoboCasa cabinet pre-place IK failure (validation 2026-09-05).**
      `scripts/scripted_tamp_pick_place.py` reaches and grasps `obj_main` successfully, then
      fails at `preplace_ik_failed` for `cab_1` at `[0.5, -0.2, 1.85]`; the table scene and
      infeasible-grasp ranking both pass. Check cabinet opening/target height and the side
      approach/standoff for the tall cabinet, then rerun the saved command in
      `logs/tamp_validation_20260905/sourccey_kitchen.log` before claiming RoboCasa support.
- [ ] **Fill `tab:tamp_clutter`** from `aggregate_tamp_clutter.csv` (scored denominator;
      exclude `skipped_invalid`). Results section is still placeholders.
- [ ] **Large registry (200 templates)** via `emet jobs`:

      ```
      NEED_MIB=8000 uv run emet jobs run --name tamp-clutter --need-mib 8000 -- \
        uv run python scripts/eval_tamp_clutter.py \
        --episodes configs/ovmm/clutter_episodes_large.yaml
      ```

- [x] **rby1 latch paper row smoke** (2026-09-05): `ithor_cleanup_s1_bin_n3` relocated
      3/3 objects with 100% manipulation success. The large-registry paper row remains open.
      Default small-YAML smoke `ithor_cleanup_s1_bin_n3` and
      large-registry rby1 `latch` episodes. Stretch / mars / nori stay `sim` on floor clutter.
- [ ] **Live latch smokes (GPU, via `emet jobs`)**: rby1 default latch
      `--episode-id ithor_cleanup_s1_bin_n3`. Mars/nori floor objects need `--manip-mode latch`
      (expected weak: Nori IK bottoms out ~0.29 m vs z≈0.02 floor). Unit/offline path already
      passes.
- [x] **Integrate TAMP single-object path smoke** (2026-09-05): the kinematic agent-tools
      gate passed. **Integrate the multi-object clear chain** remains the real product blocker
      now that MCTS pick/place
      works. The single-object semantic tools (`scene_tasks` / `plan_pick_place` /
      `execute_pick_place_plan` in `emet.controller.task.tamp.agent_bridge` + `emet/agent/tools.py`)
      exist, but the **multi-object clear chain (`plan_clear_clutter`) is not exposed**. Add a
      `clear_clutter` CHAT skill (resolve scattered objects from scene graph/memory → run the
      MCTS chain to the bin → optional landmark nav) so the LLM agent can parse "clean up the
      room" / "get to the sofa" and drive TAMP end-to-end; reuse `AgentTaskRef`/`AgentPlanBuild`
      handles and the same no-AI test battery for the agent path.
- [x] **innate_mars actuator naming**: the innate_mars MJCF actuators are *unnamed*, and the
      robosuite server applies `{"joint": vec}` via `mj_name2id(mjOBJ_ACTUATOR, aname)` —
      unnamed actuators never receive ctrl from `set_actuator_positions`. Name them (like
      nori_a3.xml) so the kinematic streaming path actually drives the arm.
- [ ] **Nori real-hardware client**: implement an `AbstractRobotClient` adapter over
      `nori-sdk` (WebRTC jog streams → emet motion contract). The SDK is teleop-oriented;
      absolute-joint moves need the action-completion path.
- [ ] **Nori MolmoSpaces spawn metadata**: `emet molmospaces write-spawn-metadata --robot nori`
      → commit `molmospaces_spawn.json`; then `emet serve mujoco --scene ithor --robot nori
      --headless` smoke (see supported_robots.md extension checklist).
- [ ] **Nori RoboCasa** row: strip-replace handling / spawn guards if kitchen scenes are wanted.
- [ ] **Stretch `latch`** via the combined robosuite server — see next section.

## TAMP clutter: kinematic `latch` on Stretch via the combined robosuite server (follow-up)

The capability gate + per-robot arm parser (Phases 1–2) landed **innate_mars** `latch`
(capability gate in `robosuite_server.py`, curated `ArmChain` on the innate_mars spec);
Stretch is deferred. Stretch `latch` needs:

- [ ] Point `get_robot_spec("stretch").mjcf_path` at `src/emet/assets/robot/stretch.xml`
      (or `stretch_mj_3.3.0.xml`) so `ArmManipProfile` can build an offline IK model and
      `KinematicPickPlaceExecutor._ensure_model()` succeeds.
- [ ] Fill the Stretch `ArmChain.actuator_names` (MJCF actuators are unnamed today) once
      the mjcf_path is set.
- [ ] Route Stretch iTHOR / Robocasa scenes through `RobosuiteZmqServer` (the merged-MJCF
      path rby1 / innate_mars use) instead of `MujocoZmqServer` — "combine the Stretch sim
      server". Harness uses `GenericZmqClient` for stretch sim (`EMET_STRETCH_GENERIC_ZMQ=1`
      already exists). Keep **default-table Stretch on `MujocoZmqServer`** so existing tests /
      interactive behavior are untouched; gate the switch on scene kind or env.
- [ ] Enable `ROBOT_DEFAULT_MANIP_MODE["stretch"] = "latch"` after a stretch iTHOR `latch`
      smoke (`plan_clear_clutter`) passes.
- [ ] Risks: stretch telescoping prismatic arm IK + RRT; the curated arm chain already
      lands in Phase 2; verify no regressions on MujocoZmqServer stretch tests
      (`emet test --no-sim` gate).

## Manipulation / MolmoSpaces + rby1 (PR #83 follow-ups)

Offline units + scripted table smokes exist; these are the remaining **real / integration** and product gaps.

### Config / agent path
- [x] **Wire `agent.manip_mode` / `manip_collision` / `manip_planner` into `DynamemTaskExecutor`**: `emet run agent` merges the finalized chat `agent:` section (YAML / `--set agent.manip_*`) into `parameters["agent"]` before executor init; `EMET_MANIP_*` env vars still win. (Manip-only top-level `agent:` blocks are now recognized as chat-agent sections by the loader.)
- [x] **Stretch MuJoCo pick/place default (docs)**: with visual-servo off, any sim advertising `sim_set_body_pose` takes **GT teleport** (`prefer_sim_teleport_manip`; table in `docs/molmospaces.md`; loud callout in `AGENT_RUN.md`). When comparing paper/OVMM Stretch numbers to the old grasp path, pass `-V` / `--visual-servo`.
- [x] **Stretch / AnyGrasp `_pickup` / `_place` always return True**: fixed — Stretch path propagates `agent.manipulate` / `agent.place` / `GraspObjectOperation.was_successful` (and declines confirmation → False).

### Real tests (not yet green on every machine)
- [ ] **MolmoSpaces ithor + rby1** kinematic and teleport smokes when `.venv-molmospaces` + assets are warm (`scripted_sim_pick_place` / `scripted_molmo_grasp_mp` / `scripted_tamp_pick_place` with `--sim configs/sim/molmospaces_ithor_train_0.yaml`).
- [x] **Molmo kinematic approach frame**: fixed `_world_base_xyt` + place detach/`sim_set_body_pose`/verify-before-retract; bowl→microwave kinematic PASS (grasp_err≈0.027, place_err=0; 2026-08-03).
- [x] **OVMM full** episode `molmo_ithor_rby1_s2_bowl_pp` with `manip_mode=sim` (find + teleport pick/place) — reconfirmed 2026-08-03 post base-frame/place fixes.
- [ ] **Robocasa / Stretch** pick-place smoke with and without `--visual-servo` to lock the teleport-vs-servo behavior.
- [ ] **CI / overnight**: mark or gate the above under `RUN_MOLMOSPACES_TESTS` / sim markers so agents use `emet test --no-sim` for the offline pack only.

### Motion / grasp quality
- [ ] **IK orientation**: MuJoCo IK is **position-only**; Molmo grasp quats set approach standoff but are not enforced at the EE. Need orientation (or constrained) IK before claiming grasp fidelity.
- [ ] **Kinematic gripper cmd failures**: `_set_gripper` open/close now logs a warning on ZMQ/actuator errors but still continues (attach/detach is the real hold). Decide whether gripper failure should fail the grasp/place result before attach, or verify finger joints after cmd.
- [ ] **GT body / receptacle disambiguation**: teleport + kinematic still take first category substring match (`receps[0]` / object query). Prefer nearest-to-nav-`point` (or explicit `object_gt_body`) when multiple bowls/microwaves exist.
- [ ] **Voxel / AABB arm collision in live agent**: kinematic path supports `EMET_MANIP_COLLISION=voxel|aabb`; default is `none`. Dogfood on a cluttered Molmo scene and turn voxel on by default only if latency is OK.
- [ ] **Instance-memory `plan_to_frontier`**: still one-sample → one plan (often RRT). Dynamem explore already uses multi-goal A*; port the same top-K → `goals=` pattern if SVM/instance agents stay in use.

### Hardware / product
- [ ] **Real-robot manip** (Stretch / Mars): no GT `sim_set_body_pose` — keep visual-servo / AnyGrasp; add a no-LLM tool-sequence test that asserts we **do not** enter teleport when `is_simulation` is false.
- [x] **`aim_arm_at` → `take_ee_picture` chain**: EE capture gated on successful aim grant (consumed once). Real-robot dogfood after pick failures still open (wrist stream / Mars).
- [ ] **Paper figures**: keep regenerating `manip_figures` / chase-cam MP4s from scripted TAMP on the scene used in the paper; check in paths under `~/runs/emet/` only (not repo blobs).
## Extracted shared-agent review stack (2026-10-09)

- [ ] Complete exact-head readiness gates in `docs/plans/shared_stack_readiness.md`.
      Oct 12 EQA replay remains main 4/4 vs integration 1/4; do not promote upper slices.
- [x] Separate optional coverage into #198; move measured-head repair to #184 and
      isolated evidence / job-ordering repairs to #185.
- [ ] Rerun Molmo after preserving the configured venv interpreter; old wrapper
      discovery errors are infrastructure failures, not scored misses.
- [ ] Rerun #198 coverage ablation after flat-parameter dispatch repair; the old
      nominal coverage run actually used `local` and is not effect evidence.
- [x] Remove MuJoCo/EGL import from A* grid helpers; before/after one-frame probe
      reproduces exit 134 then successful Habitat rendering. Repair included in #182.
- [x] Publish Stretch body-up/XYZ and named actuator controls from the physics
      process through IPC/ZMQ; merged via #190 (`7f65dbef`). Four focused tests
      pass; controls use raw simulator units, not normalized joint targets.
- [x] Reconcile main through #181 → #182 → #184 → #185 after #190; telemetry is
      owned by main, not duplicated in #185's parent diff. 165 focused tests pass.
- [ ] Complete repaired-head route acceptance and EQA pairing before promotion;
      a startup fix is not an accuracy result. See `docs/TESTING.md` for job IDs.
- [x] Land #176 conversation preservation and #180 current-view grounding.
- [ ] Accept #181 → #182 → #184 → #185 on their actual extracted revisions;
      see `docs/TESTING.md` and `docs/plans/shared_stack_inventory.json`.
      Offline checks are not live acceptance. Keep providers opt-in.
- [ ] Complete serial paired main/candidate EQA and small OVMM; restore at least
      10 GB disk headroom with authorized cleanup before new simulations.
- [ ] Track existing main failure `test_voxel_sim_upgrades_full_frame_absent_to_present`
      separately; do not relax grounding just to turn it green.
- [ ] Review archived partial CHAT dispatch / arrival hooks separately, coordinating
      #176; no claim that this extraction delivers full CHAT pick/place.
- [x] Audit #169 residuals: wheel contact fix extracted to #182 (`c6809aa6`);
      full-runner Habitat preflight retained in #185. See `docs/plans/pr169_closeout.md`.
- [ ] #179: adapt staged-clearance/failed-lift and near/far workspace regression
      requirements from `31d1002b`/`6a1579e4` to the current manipulation planner;
      do not restore the obsolete experimental grasp executor wholesale.
- [ ] Revalidate physical navigation/OVMM after the wheel-material extraction;
      the pre-extraction 16-frame scan is not evidence for the new contact model.
- [ ] Habitat OVMM, Sourccey/Galaxea and real robots remain follow-ups; no full sweep
      required for the current EQA/small-room review gates.
