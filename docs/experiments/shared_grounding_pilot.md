# Shared grounding: bounded cross-task pilot

Interpretation: [environment acceptance progression](../environments/README.md),
[Habitat search scope](../environments/habitat.md),
[simple-sim controls](../environments/simple_sim.md).

This freezes the September 10 candidate, not a new default. The earlier verified
red/blue table finds are integration evidence, not OVMM or manipulation success.

Launch: managed job `20260910_213038_d86e1a`, frozen source `b194395a` at
`/tmp/emet-cross-task-20260910-frozen`. Artifacts:
`/home/cpaxton/runs/emet/shared-grounding-cross-task-20260910`.
This first launch is invalid for comparing strategies: all four OVMM cases and
six EQA cases encountered SAM2's missing `iopath` dependency during model
construction. OVMM serialized the exception despite exit zero. The absent-object
find then reached its 360-second cap while exploring; learned pick/place never
started. Retain these artifacts as infrastructure failures and a search timeout,
not a model-quality score.

September 11 retry: `20260911_185541_56a4c4`, frozen `eed1d868`, artifacts
`/home/cpaxton/runs/emet/shared-grounding-habitat-retry-20260911`. Installed
`iopath==0.1.10` and `portalocker==3.2.0` in Habitat, without changing Torch or
model weights. The runner now constructs SAM2 and executes synthetic box
inference before episodes, and supports `PHASE=habitat|sim|all` (default all).
This retry selects Habitat only, with the original cases, budgets and presets.
`process_status.tsv` records exits, not task acceptance. All ten retry processes
completed without infrastructure exceptions:

| Strategy | EQA q15 / q16 / q25 | OVMM object + receptacle, two scenes |
| --- | --- | --- |
| Hybrid | correct / correct / wrong (2/3) | 0/4 |
| Qwen-box | wrong / correct / correct (2/3) | 0/4 |

These three questions do not establish no regression against the earlier 3/3
smoke. The strategies disagree on two questions despite equal aggregate scores.
Hybrid scene 00025 returned an object localization 3.615 m from the target;
Qwen-box returned a receptacle localization 5.184 m from its evaluator target.
Neither is a task success. Other phases returned no localization.

Manual review of hybrid scene 00025's accepted support
`grounding-965b2a585dd64549851ab3ce1af623dd-surface-0.png` shows bedding/fabric,
not an identifiable lamp. The proposal query is `nearest bed`, while the support
selector receives the full question about the lamp nearest a bed. Qwen accepts
the bedding while claiming a small object on it resembles a lamp. This is a
concrete target/anchor confusion and false semantic acceptance; metric distance
alone would not reveal it. Preserve identity versus relationship as separate
verification concerns in the next fix, without handing context pixels back the
authority to validate an incorrect mask.

The pending learned Stretch pick/place diagnostic was launched separately as
`20260911_194952_1415c5` on the same `eed1d868`, with a 600-second cap, velocity
navigation, hybrid preset and no oracle scene plans. It does not rerun the known
absent-query timeout first. The tool failed after 195.9 seconds with
`Command 101 failed: terminal command outcome is immutable`; the agent reported
the failure rather than claiming success. No successful learned pick/place is
established. The command lifecycle exception needs diagnosis; process completion
is not physical task acceptance.

Lifecycle follow-up `efba8d7f`: reproduced the exception when a goal had already
failed with stop unconfirmed, then a later cancellation confirmed stopping and
attempted to rewrite the terminal status to cancelled. The fix retains the failed
outcome/reason and records later stop confirmation separately, releasing motion
ownership only on confirmed stop. Both core/deploy copies match; 37 focused
command, trajectory and adapter tests pass. Same-preset bounded sim retry:
`20260911_200332_5b6840`, artifacts `~/runs/emet/lifecycle-manip-retry/evidence`.
The retry reached manipulation without the lifecycle exception, but failed
pregrasp with arm extension -0.180 m and subsequently lost current-frame target
support. No pick/place success. This run did not necessarily exercise stop recovery.

Manipulation handoff follow-up `40288111`: the verified find pose faced the
object with the camera, while Stretch's side-grasp expects the target along -Y.
The grasp adapter now owns a measured target-facing side rotation and fresh
reacquisition; find remains unchanged. Failed/nonfinite pregrasp IK now returns
failure instead of falling through into visual servo. 23 focused tests pass.
Bounded same-preset sim check: `20260911_202359_853d5b`, artifacts
`~/runs/emet/grasp-handoff-retry/evidence`. The retry completed orientation and
reacquisition, then reached pregrasp (IK arm -0.039 m, clamped by the legacy 5 cm
allowance). Wrist tracking rejected absent/ambiguous current-frame support;
pickup failed and place was skipped. This is not manipulation acceptance.

`3b2ce750` additionally propagates arm-motion failure rather than treating valid
IK as execution success (25 focused tests pass). `a7c45096` retains wrist RGB,
depth, calibration and grounded world points on tracking rejection under the
existing episode evidence opt-in. Serial diagnostic `20260911_202818_77c40c`
uses that frozen source; artifacts `~/runs/emet/wrist-tracking-audit/evidence`.
No tracking-tolerance relaxation or oracle fallback was introduced.

The wrist audit completed with failed pickup, not a task success. Manual review
of `wrist_tracking/grounding-054937277d1849de892afab78332be86.png` shows the red
cylinder clearly. Its stored world median projects to wrist pixel (160.3, 216.1),
with expected camera depth 0.380 m versus observed 0.370 m. This argues against
a gross camera-frame error in this observation. Replaying the current expanded
world bounding box gives two components of 4,726 and 2,005 pixels, triggering the
intended ambiguity rejection. Bounding-box-only association is insufficient in
this close view; object-specific wrist support is needed. Neither largest-mask
fallback nor looser margins is justified by this evidence.

September 12 candidate fix shares `ground_vlm_frame` between head and wrist:
same configured provider and semantic selector, followed by the original
grounded-target spatial mask association. Wrist verification does not ingest a
new instance. Positive wrist depth is accepted below the navigation-map minimum
because grasp views can be closer than mapping views; zero/invalid depth remains
excluded. The former box-only connected-component gate is not a semantic mask.
63 focused tests pass, including semantic abstention and wrong-world-target
rejection. Added per-frame VLM work may increase servo latency and must be
measured; no real-robot or learned-manipulation acceptance is claimed.

Offline red-cylinder, blue-cube and absent-banana queries on the exact saved
wrist view were submitted as `20260912_091644_fa11a1`, then cancelled before
inference: NVML reports a driver/library mismatch (loaded kernel 595.84,
userspace 595.91). GPU repair is required before cached-model and live validation.
The three inputs all reference the previously listed wrist capture; they are
diagnostic queries on one view, not an expanded independent test set.

Post-reboot replay `20260912_153104_dafcde`, frozen `dd29d217`, completed after
kernel/userspace driver alignment to 595.91.07 and successful full SAM2 inference
preflight. Artifacts are `~/runs/emet/wrist-postboot-{proposals,baseline,support}`.
YOLOE→SAM2 produced one red-cylinder proposal, one blue-cube proposal, and zero
absent-banana proposals. Support-only Qwen accepted red and blue (selector times
1.23 s and 0.99 s respectively); banana abstained without a model call.
Manual inspection of `wrist-postboot-support/support_only/0-panel-0.png` confirms
the red support depicts the cylinder. Reprojecting the saved calibrated depth
and applying unchanged original-red-target association accepts 1,397 red pixels
and rejects the blue mask. This validates the intended distinction on this
saved frame, not robustness during motion or physical pickup. The focused
63-test suite also passes post-reboot. Live bounded pick/place retry
`20260912_153738_fb4a22` uses the same frozen source and preset; evidence goes to
`~/runs/emet/wrist-live-postboot/evidence`. The tool failed after 40.3 s:
wrist semantic/world association completed, then a legacy shape check accessed
`servo.semantic.shape`, despite head semantic labels being absent on this path.
No pickup or place succeeded. Process exit zero is not task success; teardown
also emitted multiprocessing broken-pipe/reset errors. Fix `2db1b757` validates
the selected wrist mask against wrist world geometry instead, with tests for
absent head labels and mismatched wrist resolution (11 handoff tests pass).
Same-preset bounded retry `20260912_201500_bb5284` retains all model and control
settings; artifacts `~/runs/emet/wrist-shape-retry/evidence`. The focused suite
passes 65 tests. The retry completes the wrist approach from 0.369 to 0.166 m,
closes the gripper, and advances to finding the blue cube. It then fails
placement navigation at the first waypoint: command 186, `motion deadline
exceeded`, `stop_confirmed: true`. Tool duration is 182.3 s; overall pick/place
fails. There is no independent confirmation of a held object. Post-grasp head
image `grounding/grounding-6b9317d388ee4aa09d068ca40238fbdf.png` shows the blue
cube and a small red region beyond the table, not a verified in-gripper cylinder.
Do not score gripper closure as physical pickup. Next isolate physical grasp
verification and post-grasp navigation; do not loosen tracking tolerances or
launch broader acceptance pilots yet. Multiprocessing teardown errors remain.

### Placement-navigation timeout: wheel transmission units

The failing placement waypoint is approximately an in-place 175-degree turn,
not a long drive. Isolated baseline `20260912_232620_453a49` (source `933fa725`)
reproduces it without perception models or manipulation. The nominal 10 s
command budget is stretched by the existing sim-time ratio: failure occurs at
12.27 wall seconds, measured yaw -2.369 versus goal -3.05 rad (0.681 rad short).
Pose samples show continued slow rotation, not a stationary obstacle stall.
The earlier diagnostic `20260912_232447_3fc9ac` never launched simulation because
its temporary route YAML was malformed; exclude it from navigation results.

Stretch's MJCF wheel velocity actuators use `gear=3`. The bridge previously
sent wheel-joint rad/s directly to actuator controls, and interpreted actuator
velocity as wheel-joint velocity in feedback. MuJoCo's [transmission semantics](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-general-gear)
scale actuator velocity by gear. Fix `76ebf9a1` multiplies commands by the model's
actual gear and reads joint `qvel` for wheel-speed telemetry. No MJCF friction,
limits, navigation deadline, arrival tolerance, or speed-boost changes.
15 tests cover command/feedback units with unit, positive non-unit and negative
gears; the broader adapter/command suite passes 44 tests.

Matched fixed diagnostic `20260912_232858_58901d` succeeds: first positioning
command 1.35 s, half-turn 7.01 s, return turn 6.46 s. Terminal half-turn error
is 0.055 m / 0.142 rad, within unchanged exploration tolerances 0.07 m / 0.15 rad.
These are exploration-contract passes, not precision-navigation acceptance.
Both runs use the existing `probe_rby1_camera.py` on seed-0 default-table Stretch,
with a temporary wrapper selecting the unnamed contract and nominal 10 s timeout,
and logging pose samples. Route: `(0.13, 0.0085, 0.006)`,
`(0.13, 0.0085, -3.05)`, `(0.13, 0.0085, 0.006)` in episode coordinates.
Artifacts: `~/runs/emet/nav-turn-{trace-v2,geared}`, including observations and
images; managed job logs retain timestamped `NAV_SAMPLE` and `NAV_RESULT` lines.
Integrated same-preset learned retry `20260912_233137_f012c2` uses frozen
`76ebf9a1`; artifacts `~/runs/emet/manipulation-wheel-units/evidence`. Initial
navigation completes without timeout, but fresh target grounding rejects
`target absent or ambiguous` before pickup (tool 17.7 s); place is skipped.
This run does not exercise post-grasp navigation, and does not establish
manipulation success. Keep that integration gate open despite the isolated
navigation fix passing. Existing multiprocessing/EGL teardown errors persist.

Follow-up `20260912_233819_e45145`, source `94149560`: one repeat of the existing
`configs/benchmarks/navigation_acceptance.yaml` using `probe_rby1_camera.py`
(`--route-repeats 1`), default-table Stretch, no perception models. All ten
commands succeed under the named precision contract (2 cm / 0.03 rad), including
±10° and ±30° turns and 20 cm forward/return translation. Maximum terminal
errors: 0.01939 m and 0.02953 rad; zero corrective redispatches. Artifacts:
`~/runs/emet/nav-geared-precision`. The probe summary is `incomplete_telemetry`,
not full acceptance: Stretch omits the requested base-up/actuator-target fields.
This is one route repeat on one robot/environment, not cross-robot robustness.

Manual comparison of the final two frames from `manipulation-wheel-units`
shows the red cylinder on the extreme lower-right edge of accepted frame
`grounding-4e5d1b8424ba4aeda2f56cb0e358c07f.png`, and no visible cylinder in
rejected `grounding-8a445c194f364a029016ffcaa02cb195.png`. The rejection has no
supported proposals; Qwen is not rejecting a clearly visible cylinder there.
The sweep permits an early return within 0.35 rad of commanded head pose,
even during motion. Residual camera motion is a hypothesis to measure before
changing verification. Camera-only trace `20260912_234053_eb1405` confirms the
mechanism: after the soft-wait handoff, saved camera rotation changes another
8.9145 degrees; base displacement is under a micrometre and yaw change under
0.003 degrees. `~/runs/emet/head-handoff-trace/handoff/head_0.png` shows the
cylinder and cube, while `head_3.png` shows neither. First sample is already
0.20 s after soft-wait return; this is a lower bound on total post-return drift.
The full observations have `image_timing: null`; do not infer capture latency
from joint-state sampling alone.

Fix `d9bb2ce3` uses the robot adapter's blocking head move and existing newer-frame
wait for verification callbacks only; ordinary mapping retains its soft sweep.
91 focused head-sweep/query/grounding/manipulation tests pass. This does not
upgrade the underlying head-motion API to a verified success contract; clients
that silently time out remain a follow-up. Serial job `20260912_234501_ffeb03`
checks the fixed camera handoff, then the same bounded learned command. Artifacts
`~/runs/emet/head-handoff-settled` and `~/runs/emet/settled-gaze-retry/evidence`.
Fixed camera handoff rotates just 0.0037 degrees across the corresponding saved
views after a 1.05 s head/frame wait (versus 8.9145 degrees before). Integrated
retry passes arrival verification and immediate fresh grounding, then rejects
after the subsequent side-grasp rotation/head movement: final
`grounding-930a1f12ee68467e86625b9a83089794.png` shows floor and table edge.
Tool duration 30.1 s; no pickup or place. The grasp adapter did not wait for the
RGB-D stream to advance after that second movement. Candidate `4f946d1d` adds
the existing post-motion frame wait there, without changing mask acceptance;
46 targeted tests pass, including head → frame wait → grounding order.
Bounded retry `20260912_234917_6b6fdf`, artifacts
`~/runs/emet/grasp-fresh-frame-retry/evidence`, completes with failed pickup
(tool 62.1 s). Both head-camera handoffs pass and wrist servo approaches from
0.368 to 0.218 m. Final calibrated wrist capture
`wrist_tracking/grounding-006bc3656f6440b19ab073405e0573ec` shows the cylinder
partly behind a gripper finger and clipped at the lower image boundary. Qwen
selects candidate 4, but its resulting instance-0 mask fails original-target
world association: only 1,339 / 3,385 valid pixels (39.6%) lie in the original
bounds plus the unchanged 5 cm margin, versus the required 80%. Selected support
median `(0.118, -0.465, 0.476)` differs from original target median
`(0.079, -0.539, 0.523)`. Do not relax the threshold on this example. Distinguish
object displacement/contact, mixed mask support, and camera/pose timing before
changing tracking. Several servo-center depths are zero and manipulation base
goals are repeatedly reissued; inspect these control paths next. No gripper
closure or place success in this retry. The full 91-test focused suite still
passes; these are correctness repairs, not long-horizon manipulation acceptance.

## Fixed comparison

- Hybrid: `query_detector_segmented_pilot.yaml`, YOLOE boxes → SAM2.
- Reference: `query_segmented_support_pilot.yaml`, Qwen boxes → SAM2.
- Both: support-only Qwen3-VL-8B-Instruct int4 verification, lazy query memory,
  unchanged fusion and task budgets. This tests proposal strategies, not SAM2
  versus YOLOE segmentation, and not lazy memory versus DynaMem.
- Habitat OVMM: nearest-v2 scenes 00006 and 00025, seed 0, 12 rounds/8 nav steps.
- EQA: questions 15, 16, 25, 20 planning/10 movement steps, no semantic sensor
  labels or enriched GT hints. The CLI does not pin EQA RNG seeds: these are
  matched questions/settings, not a deterministic causal no-regression test.
- Hybrid-only Stretch table: one absent-object find and one learned pick/place.
  Velocity navigation; no teleport or oracle manipulation. This is a multistep
  integration diagnostic, **not the PR #160 TAMP battery**. The oracle TAMP
  control previously passed; learned TAMP acceptance remains outstanding.

All twelve processes run serially. Stop after a timeout to inspect cleanup.
Each has a wall cap, not an extended task budget. The runner records process
completion separately from benchmark correctness. Report abstentions, grounding
errors, navigation failures and manipulation failures separately; do not turn
tool `ok` or exit zero into physical success.

## Reproduction

From a frozen checkout with the SAM2 small, YOLOE-L and MobileCLIP checkpoints
available at the usual loader paths:

```bash
OUT=/absolute/new-artifact-directory \
HABITAT_BIN=/absolute/habitat-env/bin/emet-habitat \
AGENT_PY=/absolute/agent-env/bin/python \
SAM2_SOURCE=/absolute/pinned/segment-anything-2 \
emet jobs run --name shared-grounding-pilot --need-mib 16000 \
  --cpu-safe --gpu-exclusive -- bash scripts/run_shared_grounding_pilot.sh
```

`EMET_CONFIG` already feeds the unified preset to both Habitat runners through
`get_parameters`; no new benchmark-only config API is needed. Preflight verified
that the EQA and OVMM profiles retain the selected query options and model.
SAM2 source is pinned to `2b90b9f5ceec907a1c18123530e92e794ad901a4` for this run.
Using it through `PYTHONPATH` avoids changing the shared Habitat installation.
Habitat's installed executable sets its required C++ library path.

Review the saved grounding RGB/masks/poses, agentic traces, navigation receipts,
and EQA map/video bundles. Inspect final accepted support manually, especially
any positive absent-query response. A new contamination or wrong-object
acceptance blocks promotion even if aggregate correctness improves. Retain
zero-result failures and infrastructure exclusions. No full sweep is authorized
by this pilot; use its failures to select the next shared-mechanism fix.
