# Shared-agent stack readiness — 2026-10-12

Clear slices independently. Non-draft means evidence is ready for review, not
permission to merge. Keep defaults unchanged; no full sweep or hardware gate.

## Acceptance review repairs — 2026-10-15

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
- Combined reviewed evidence branch `7c1bba8e` preserves repaired map epochs,
  actual-capture timing, bounded approach views and spatial labels. Its normal
  opt-in `configs/emet/evidence_context.yaml` is queued against main on the same
  EQA-12 seed: job `20261015_212434_f4fcff`, output
  `/tmp/emet-reviewed-evidence-profile-eqa12-s0-20261015`. Defaults unchanged.

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
negative Y. Upright telemetry and measured head tilt exclude a fall as the cause
of its dark assessment images. Standard eight-view S0 diagnostic
`20261015_083813_207ff7` is running on both arms. Do not infer perception failure
from an unobserved target direction. All jobs use the serial GPU lock.
Final-owner default EQA gate `20261015_084243_b8f836` is queued: main `801dce82`
versus integration `63016107`, the same 12 questions, seed 0, no experimental
selection overrides. Final task acceptance remains open.
The earlier candidate's EQA 26/36 versus main 25/36 is a modest development
result on different frozen revisions; prioritize reliable shared tooling and
do not present this as a meaningful accuracy gain or this PR's exact-head gate.

## Ownership and exact heads

| PR | Ownership | Current gate |
| --- | --- | --- |
| #181 `dc106273` | Opt-in grounding providers, shared assessment dispatch | Exact-head default-path EQA attribution and saved-view provider audit pending. |
| #182 `1319ea6d` | Measured navigation, clearance, wheel contacts | Exact-head route/EQA and short Molmo rby1 route pending. Integrated route result cannot clear this head. |
| #184 `48b56aeb` | Bounded head/floor recovery, measured head-arrival fix, executor delegation | 284 focused ownership-repair tests passed at the acceptance descendant; exact-head simulated recovery and paired EQA pending. |
| #185 | Frozen drivers, evidence isolation, queue ordering, environment discovery | 29 focused driver/metric/discovery tests passed. Live shared-driver pair and remaining failure-path checks pending. |
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
