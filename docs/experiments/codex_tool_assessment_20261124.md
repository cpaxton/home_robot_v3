# Codex-supervised manipulation tool assessment (2026-11-24)

Question: can a stronger visual supervisor execute these tasks using the same
camera requests and native manipulation tools, and which failures belong to the
tools versus the configured local Qwen3.5-9B?

## Protocol and limits

The file-backed `scripts/codex_observation_gateway.py` records the exact VLM
request text and PNG images. Interactive Codex reads these and writes the same
JSON schema accepted from Qwen. Depth, calibration, robot kinematics and native
encoder feedback remain downstream in the shared tools. No object CAD, object
pose, segmentation ID or evaluator success label is supplied to the supervisor.
The scorer independently uses simulator state after execution. Known robot hand
collision geometry is used to avoid asking the fingertips to pass through the
measured counter; that is robot geometry, not an object prior.

These are three familiar development fixtures (pear, garlic, tongs). Interactive
Codex has conversation/history and can inspect recorded failures; Qwen receives
stateless requests. This is a comparison of configured supervisor systems, not a
controlled claim about model weights or general benchmark performance. Routing
is the native diagnostic phase executor, not the production CHAT loop.
Environment collision certification remains explicitly bypassed in simulation;
native contact physics, object properties and gripper force limits are unchanged.
Contact counts are telemetry, not automatic task failures. The RGB-D is
rendered/calibrated simulation sensing; hardware depth noise and calibration
error are not qualified by these runs.

Artifacts are under `logs/codex-tool-trials-20261124/` in the repository root;
each batch preserves manifests, critical source hashes/copies, profile, request
images, decisions, native receipts and independent scores. Historical failures
remain intact. Gateway directories are `gateway/` (initial bridge) and
`gateway-v2/` (longer manual-response deadline). Five initial 85-second manual gateway
timeouts (0038, 0052, 0055–0057) are orchestration failures; they must not be attributed to perception.

## Demonstrated tool defects and repairs

- Cartesian approach imposed the final wrist rotation abruptly. Interpolate both
  position and SO(3) rotation; evaluate the two equivalent parallel-jaw wrist
  orientations before committing.
- Camera image directions were projected naively into world XY. Backproject the
  requested image direction onto the estimated horizontal grasp plane using
  camera calibration.
- Crop refinement stretched images to 640 × 480. Preserve aspect ratio and map
  image directions using actual crop dimensions.
- Visible surface centering biased the grasp toward the camera. Interpret the
  semantic grasp-center pixel at the estimated interior section height, while
  depth supplies the local object neighborhood and height estimate.
- Joint arrival could hide actual fingertip target error. Verify the requested
  Cartesian grasp endpoint, with explicit development-profile tolerances. The
  initial 0.8 mm endpoint limit later rejected a pear approach at 0.86 mm,
  before closure; the final matched profile uses 2 mm. This is an admission
  setting, not evidence of a failed physical grasp.
- Low grasp estimates could demand hand/counter penetration. Bound the hand
  bottom using known robot geometry and observed support height.
- A grasp image could become stale while inference ran. Include encoder readings
  in the visual query, and reinspect if occupied fingers become empty during
  interpretation. Never score success from the image alone.
- Long blocking visual calls while loaded let a marginal tongs grasp slide out.
  The atomic-transfer development profile performs visual planning before grasp,
  then uses native occupied-to-empty grip monitoring during motion. It does not
  silently wait for a model while carrying. This does not prove robust long holds.
- Open-sink transfer needlessly lowered the hand to a precisely estimated floor.
  An explicit development placement mode releases above a measured open basin.
  It uses source-counter height as a sink-rim proxy; this is not a general
  container-placement guarantee.
- Growing the selected destination into a full connected floor plane discarded
  the semantic placement region. The new tool chooses measured floor nearest
  that region, preserving the supervisor's goal rather than forcing the entire
  basin center. The discarded-region target caused tongs transport IK failure
  after a verified 9.7 cm lift; robot-only observed-input replay passed the
  nearer requested route.
- A native fixture-restoration crash occurred at a named joint view. Use raw
  joint-ID/qpos-address lookup for that restoration. Later startups passed;
  broader native-process crash robustness remains unqualified.

## Development sequence (all attempts retained)

| Batch | Scope | Result |
| --- | --- | --- |
| `pick-development` | Original tools, Codex supervisor | 0/3 retained picks |
| `pick-pose-repair` | Gradual wrist motion and corrected directions | 0/2 scored, 1 startup crash |
| `pick-grasp-repair` | Crop/center repair and Cartesian endpoint check | 2/3 retained picks |
| `verification-r1` | Feasible hand/support height | 2/3 retained picks |
| `verification-r2` | Tighter endpoint control | 2/3 retained picks; one planning timeout |
| `atomic-transfer-development` | Full transfer without loaded model pauses | 2/3 full tasks; tongs grip loss then recovery timeouts |
| `atomic-transfer-verification` | Repeat full tasks with longer manual deadline | 2/3 full tasks; tongs lifts/carries, then fails IK at the substituted basin-center goal |
| `atomic-transfer-region-verification` | Preserve semantic placement region, historical 0.8 mm endpoint gate | 1/3 strict tasks; pear refuses at 0.86 mm, garlic passes, tongs passes official success but fails full-bounding-box containment |

Tongs repeatedly lifted about 9.7–9.8 cm in retained-pick tests but dropped during
long loaded interpretation/hold. It actually returned to the counter; this was
not merely a strict scorer rejecting an otherwise retained object. Tightening
endpoint control alone did not resolve it. The first atomic-transfer tongs
attempt targeted a wider tip section but lost grip during initial lift. Its
recovery was interrupted by manual-response timeouts. These failures remain in
the denominator; later verification is reported separately.

## Frozen-tool verification and matched Qwen comparison

The tongs-only semantic-region development retry passed the full task with a
9.52 cm physical lift, zero contact frames, two grounding proposals and one
physical execution. The first proposal lacked usable support geometry.

Codex completed the final frozen six-trial batch: **6/6 official RoboCasa
successes**, qualifying physical lifts and completed/settled transfers, all with
one grounding proposal and one physical execution. The original stricter harness
score remains **4/6**, because both tongs runs fail its additional full
registration-bounding-box containment check. These are not rewritten as strict
passes. Pear and garlic each pass both restarts under both metrics. Tongs passes
the official predicate and settles in the sink, but that does not qualify the
extra whole-box placement criterion. The matched Qwen batch plus the explicit infrastructure-only retry produced
**1/6 official successes**, **3/6 qualifying physical pickups**, and **0/6 strict
harness successes**. Its one official success required physical recovery; no
Qwen trial achieved first-attempt official task success. The original simulator
import crash (before binding/model input) remains recorded separately. Both
batches and that retry have identical critical source hashes and profile bytes;
fixture IDs, hashes and specifications also match.

| Final matched metric | Codex | Qwen3.5-9B |
| --- | --- | --- |
| Official success + completed/settled transfer + physical lift | 6/6 | 1/6 |
| Qualifying two-pad physical pickup | 6/6 | 3/6 |
| First-attempt official full transfer | 6/6 | 0/6 |
| Original harness score including full-box containment | 4/6 | 0/6 |

The two Codex strict-score failures are tongs placements. The Qwen failures are
both tongs grounding refusals, both garlic transfers, and the retried second
pear transfer. Its successful pear transfer still fails strict containment.
Original history, explicit retry and resolved comparison rows are saved in
`logs/codex-tool-trials-20261124/matched-comparison.json`; the adjacent
`summarize_matched.py` regenerates that report without changing raw receipts.
The raw Qwen batch contains five scored trials and one startup crash; the
separate pear retry supplies the sixth scored trial rather than replacing any
scored model failure. Use `codex_atomic_transfer_repeat_development.json` and
`qwen_atomic_transfer_repeat_development.json` with the identical
`pandaomron_atomic_transfer_tolerant_development.json` profile (2 mm endpoint
tolerance). The grasp-site development profile also now uses 2 mm rather than
retaining the demonstrated 0.8 mm refusal setting; historical profile bytes are
preserved in each artifact directory. Each manifest restarts pear, garlic and tongs twice. Earlier
verification manifests retain the stricter historical profile. Full success requires
completed transfer, qualifying two-pad physical lift, official RoboCasa success,
basin containment and settling. Planning/execution retries and infrastructure
failures are recorded separately. Do not infer a model-only limitation from
Qwen results collected before these tool repairs.

## Reproduction

Run from the `agent-harness-expanded-testing` worktree with repository-root
artifact paths. Start the interactive gateway with a **new** output directory:

```bash
python scripts/codex_observation_gateway.py <new-gateway-directory> \
  --port 8767 --response-timeout-s 580
```

Run `scripts/eval_vlm_transfer_batch.py` with the Codex repeat-development
manifest, the pinned `robocasa-pinned-final` fixture root, a new output directory,
`--profile configs/eval/pandaomron_atomic_transfer_tolerant_development.json`,
`--endpoint http://127.0.0.1:8767/v1`, a free `--port-offset`, and the existing
pinned native simulator Python path. Set `EMET_VLM_REQUEST_TIMEOUT_S=600` for
interactive supervision; the normal client default remains 90 seconds. Read each
pending `observation.json`, inspect its PNGs and atomically write `decision.json`.
Do not inspect active grader/object-state traces to choose actions. Unavailability
and failures should be returned honestly; never manufacture a success response.

For Qwen, use `configs/eval/qwen_atomic_transfer_repeat_development.json` and
`http://127.0.0.1:8765/v1`, otherwise the same profile and source revision. These are fixture restarts, not identical
recorded-image replays; the capture-aligned robot views vary between restarts. Run one
simulator at a time. Each batch freezes its own source/profile copies and checks
fixture hashes. Compare these hashes before attributing differences to the
supervisor. The local model is BF16 Qwen3.5-9B, greedy decoding, thinking disabled,
revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

Current regression check: **662 tests passed**, including crop angle/aspect,
image/plane conversion, Cartesian arrival, known-hand support bounds, stale
visual evidence, native feedback and semantic destination preservation. This
validates the implementation controls; physical success still comes from the
separate native simulator receipts.


## Remaining tool contracts

The native phase diagnostic is not the full CHAT agent. Before broad or physical
qualification, preflight the complete transport route before closing the hand,
return measurable/retryable reachability feedback, and bind post-release visual
verification to the production tool result. Test absent targets and unfamiliar
objects/layouts with a frozen protocol. Long loaded holds require their own
retention checks even if atomic transfers pass. Audit registration-box containment
against the official benchmark and observed resting object; preserve both metrics
rather than silently redefining a failed historical task.


The initial sequential comparison wrapper received SIGTERM after all six Codex
receipts were finalized. Qwen was launched separately with the same source hashes,
profile and fixture specs. Wrapper/process cleanup is an orchestration issue;
all six completed Codex receipts remain in the comparison. No model result is
invented for the unstarted Qwen portion of that wrapper.


## Attribution from the matched receipts

The original tools were not a reliable upper-bound control: Codex initially
failed all three retained-pick cases. The camera/pose/arrival/goal repairs above
therefore address demonstrated tool defects, not imagined weaknesses of Qwen.
On the repaired tools, Codex's six official successes establish that these
observations and actions contain enough information for these familiar tasks.
They do not establish general object/layout coverage or a model-weights-only
advantage: Codex retained development history while Qwen's calls are stateless.

Qwen's first tongs repetition proposed the same box for object and support on
all three tries, despite instructions to use separate regions. Its destination
patch also lacked usable support geometry. The second tongs repetition likewise
stopped in grounding. Garlic's first repetition physically lifted, lost grip
on transport and failed IK during recovery. Its second repetition missed and
then produced a crop in which the model reported the target absent. Pear's
first repetition completed official success after an empty-grip-triggered
regrasp; it failed the stricter containment check and accumulated more contact
frames than the Codex pear runs. Preserve the prior execution receipt: a final
empty `events` list after recovery does not mean the entire trial was a
perception-only refusal.

The current local 9B supervisor is thus unreliable at this combined multi-view,
three-region argument-construction interface. That is not evidence that VLMs
cannot perform the task, nor proof that training is the only fix. The next useful
ablation is simpler model calls for object and destination separately, explicit
role-specific geometry feedback, and measured-depth refinement of rough grasp
proposals. The tool should preflight and recover whole motion sequences rather
than require the VLM to predict IK feasibility from an image. Test that on new
fixtures before adding a heavier object pose tracker or assuming the model needs
exact object CAD.


## Scoring clarification after the comparison

Scoring version 2 separates the benchmark result from full registration-box
containment. The pinned RoboCasa `PickPlaceCounterToSink._check_success` calls
`obj_inside_of(..., partial_check=True)` and requires the gripper to be far away.
Our extra predicate instead requires every rotated registration-box corner's
horizontal coordinates to fit inside a single basin, with no margin. Empty box
corners can extend beyond an irregular object's actual shape. The two criteria
are intentionally different; a strict-box failure alone does not establish that
the physical object escaped the sink, nor prove that its whole shape fits.

New `task_passed` and `physical_transfer_passed` use official success plus
verified lift, completed execution and settling. `strict_box_transfer_passed`
preserves the old additional requirement, while `whole_box_contained` reports
that geometry predicate separately. Contact diagnostics remain independent.
Historical raw score files above are unchanged; the matched 6/6 versus 1/6
official results were already computed and reported before this change.
