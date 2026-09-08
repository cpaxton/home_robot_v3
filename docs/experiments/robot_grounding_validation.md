# Robot grounding validation (2026-09-07)

Review branch: `fix/grounding-robot-validation`, based on navigation-contract
commit `18f11d11`. This is a bounded engineering pilot, not a paper sweep.
Use one common command/perception harness; robot assets and actuator mappings
remain robot-specific. Run heavy jobs serially with the exclusive GPU lock,
`--cpu-safe`, and one numerical-library thread.

## Fixes and evidence

- `c16e1d18`: the rby1 simulator uses the Galaxea R1 proxy. Its head camera was
  oriented through the visual mesh frame, sideways to the torso; pitch commands
  rolled the image. Give the optical camera an explicit torso-forward frame and
  use upper torso pitch for looking down. This is not hardware calibration.
- `4e45fc67`: Stretch dispatched trajectory waypoints twice and ignored final
  failure. Generic could overlap goals in nonblocking mode. Both now serialize
  one command per waypoint and require terminal success, including the final
  waypoint. No arrival tolerances were relaxed.
- `0b031566`: explicit episode trace paths now enable trace collection, including
  DynaMem. Preserve raw replies with no parsed router tools to diagnose malformed
  output before changing prompts or generation budgets.
- `16c3f62e`: camera masks enabled hidden geometry groups. MolmoSpaces group 4
  contains bright-green collision proxies, which obscured RGB and depth. Preserve
  MuJoCo's default visual groups for all cameras, then apply existing primary
  camera self masking. This is shared renderer behavior, not scene detection.

Focused command, camera, configuration and Mars kinematics gate: 72 passed;
deployment and standalone runtime parity: another 15 passed. These do not start
ROS or prove physical navigation. Live hardware tests were excluded.
An additional 29 wrist decoding, lazy commit, evidence policy, router and TAMP
interface tests pass. Two lazy-commit tests initially attempted real VLM startup;
they now use deferred clients to test memory behavior without loading a model.

## Pilot ledger

| Job | Configuration | Outcome |
| --- | --- | --- |
| `20260907_180003_eea62f` | Stretch/default table, DynaMem, 8 views, agentic 12 rounds/8 nav | 1/2 find phases; red localized via voxel, blue failed; 487 s; malformed router replies and navigation timeout |
| `20260907_180407_e8f695` | rby1/native iTHOR, camera fix, 8 turns | Nine captures, no command failures, upright camera; images exposed green collision-hull rendering, so not a valid perception gate |
| `20260907_180914_3a28e2` | Stretch/default table, DynaMem, 1 view, non-agentic | 0/2; insufficient evidence to establish a regression against the different eight-view setting |
| `20260907_181830_dc1d57` | rby1/native iTHOR, corrected visual groups, 8 turns | Nine captures, no failures; inspected views show textured surfaces rather than collision hulls; initial heading faces nearby wall |
| `20260907_181853_b41305` | Repeat eight-view Stretch agentic control after fixes | 1/2 scored phases, 343 s; blue prediction exists but misses by 0.367 m versus 0.3 m radius; both phases exhaust 12 rounds, no verified observation |
| `20260907_182113_564242` | Intended rby1/native iTHOR learned find | Invalid: zero episodes executed despite job status done; absolute episode path was replaced by benchmark path from another checkout |

Artifacts respectively reside under `/tmp/emet-stretch-grounding-control-20260907`,
`/tmp/emet-rby1-native-camera-fixed-20260907`,
`/tmp/emet-stretch-voxel-only-control-20260907`,
`/tmp/emet-rby1-native-visual-groups-20260907`, and
`/tmp/emet-stretch-waypoint-grounding-retest-20260907` on the development host.
Native learned-grounding output is `/tmp/emet-rby1-native-learned-grounding-20260907`.
Archive selected RGB, traces, result JSON and launch settings before publication;
temporary host-local paths are not a reproducible public artifact release.

The Stretch control uses the existing S0 parity settings, including perfect
simulator depth. It is not a sensor-only paper result. The earlier 0/2
query-driven result was **rby1/default table**, not Stretch or native MolmoSpaces.
Native rby1 grounding must use `molmo_rby1_ithor_s2_idx0`; the similarly named
`molmo_ithor_s2_idx0` row uses Stretch.

## Completion gates

1. Inspect actual corrected native rby1 RGB/depth, not only pose metrics.
2. Run native rby1 learned grounding and the paired Stretch agentic retest;
   retain failures and raw tool replies. Do not tune strong object-admission
   thresholds to compensate for bad views or treat weak search hints as objects.
3. For Mars, first verify deployed bridge version and fresh stationary head and
   wrist RGB/depth/poses. No base/arm motion. Reconnecting hardware is useful for
   that gate only; it does not depend on declaring every OVMM phase solved.
   The 116 focused tests and corrected live camera probe make this stationary
   hardware gate worthwhile now. Mars has not been contacted or moved in this run.
4. Keep EQA and GT-assisted integrated TAMP results separate from learned OVMM.
   Re-run bounded integration checks after shared changes before paper updates.

Outstanding independent issues include Robocasa torso physics, Sourccey/Molmo
solver failure and incomplete Stretch health telemetry. Neither mock tests nor
GT-assisted TAMP successes establish that these are fixed.

## Simulation-only continuation (2026-09-08)

Mars is explicitly deferred while the operator is away. No hardware contact.

The saved Stretch router replies repeatedly end at `"obs_id": -300000`.
Replaying the exact prefix with the cached Qwen3-VL-4B tokenizer gives 32 tokens,
not the 128-token limit. Five zero tokens trigger `RepetitionStop`. Numeric
blocks are now exempt from phrase-loop detection; generation remains bounded
by its existing token/time limits. Do not raise router budgets or relax grounding
thresholds to hide this bug. The recorded reply passes the corrected guard.

The batch runner now preserves explicit absolute episode files and fails before
worker startup if no episodes match. Relative CLI defaults still delegate to the
benchmark. This prevents reporting an empty selection as a successful pilot.
The focused selection, repetition, Qwen call-path, OpenAI adapter and router gate
passes 41 tests after these changes.

Serial retests at `3766381b`, retaining eight views and 12-round/8-nav budgets:

- `20260908_085100_ba4d01`: rby1/native MolmoSpaces, query-driven lazy graph;
  `/tmp/emet-rby1-native-router-retest-20260908`.
- `20260908_085121_79494e`: Stretch/default table, DynaMem;
  `/tmp/emet-stretch-numeric-router-retest-20260908`.

Both are bounded engineering runs, not a full sweep. Outcomes: rby1 0/2 in
129.5 s; Stretch 1/2 in 345.5 s. Router calls now parse, but Stretch investigation
was rejected because graph memory was absent. Blue localization missed the
scoring radius by returning a point 0.370 m from the GT bounds (radius 0.3 m).

### Investigation and exploration integration

- `49a62da3`, `f2ac713f`: remove graph requirements from voxel investigation,
  waypoint resolution and annulus sampling. Tests cover dispatch and actual
  waypoint resolution separately. Graphless capture/verification still has
  graph-observation assumptions and needs a shared view-evidence interface;
  these changes alone do not establish end-to-end DynaMem verification.
- `1cdedb36`: retain planner target metadata with visualization disabled,
  measure fallback exploration displacement, and block no-progress goals.
- `927bc54d`: apply blocked-goal filtering in the GraphEQA ranked override too.
  The live follow-up exposed that override bypassing the base picker.
- `f78cbcde`: allow third-person recording around explicit CHAT kinematic tool
  execution. The first attempt was rejected by the old script guard, not by
  the manipulation executor.
- `9d9aa364`: prevent a Stretch server worker joining itself during shutdown.

| Job | Check | Outcome |
| --- | --- | --- |
| `20260908_114752_8e542d` | Stretch graphless investigation dispatch | Stopped after retaining NO_WAYPOINT and navigation-timeout traces; waypoint graph requirement was fixed after launch; no completed score |
| `20260908_115153_923a59` | rby1 exploration progress | 0/2, 135 s, three graph nodes; exposed ranked override still repeating blocked frontier |
| `20260908_115320_e7c308` | Integrated TAMP + video | Failed at unsupported video/tool combination before execution; script fixed |
| `20260908_115716_c281de` | Stretch default table, four turns | Five captures, no motion failures; incomplete health telemetry; teardown self-join fixed afterward |
| `20260908_120238_f6731b` | rby1 with ranked override fix | 0/2, 130.5 s; graph fallback advances to space fallback, but frontier approach still uses object stand-off |
| `20260908_120415_6467fb` | Integrated TAMP CHAT with video support | Passed; 2.2144 m object displacement; 272-frame, 640x480 MP4 (22.7 s); GT-assisted, not learned manipulation |

Outputs follow their named `/tmp/emet-...-20260908` job directories. The initial
TAMP launcher `20260908_115237_690772` was cancelled before launch to correct an
unsupported output-path flag; it is not a test result.

See [simulation tryout](simulation_tryout.md) for exact workstation commands,
including the difference between scripted GT-assisted tools and learned models.

`29cc1bfa` passes the existing exploration mode through both navigation entry
points and distinguishes coverage goals from object stand-off targets inside
the sampler. Exploration picks the closest reachable, footprint-valid cell;
object stand-off distances, obstacle filtering and visibility checks remain.
Tests exercise both modes and an occupied target. The final combined gate is
121 passed (commands, exploration, waypoint planning, agent loop, renderer mask,
shutdown, decoder and batch selection), with two dependency deprecation warnings.

Final bounded learned retest: `20260908_121001_7d5280`, artifacts
`/tmp/emet-rby1-frontier-mode-retest-20260908`: **0/2**, 112.1 s, four graph
nodes, 899 explored cells. The robot reaches different navigation targets,
but verification reports `NOT_A_VIEW` with no station/view identity. Neither
phase produces a verified observation or localization. This is not evidence
that frontier changes solve learned grounding, nor a controlled speed comparison.

The next shared-harness blocker is capture/verification provenance: view evidence
needs an identity and verification interface independent of object-graph
insertion. Graphless capture currently has no graph observation to return, and
lazy captures can also lack the view required by verification. Fix this shared
interface before threshold tuning; do not manufacture object nodes merely to
give verification an ID. Then repeat the bounded Stretch and native rby1 checks.

The integrated recording is
`/tmp/emet-integrated-tamp-chat-video-20260908/third_person.mp4`.
Recording works, but an inspected frame is heavily occluded by scene geometry;
the chase-camera framing is not paper-ready.
