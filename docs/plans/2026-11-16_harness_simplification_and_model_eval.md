# Harness simplification and efficient model evaluation

Objective: one coordinator and shared tools for EQA, room-scale OVMM and TAMP.
Improve context and execution contracts so a better model can make better choices
without new task-specific routing rules. Keep model decisions and scored physical
outcomes independently inspectable.

## Implemented first slice

- Find returns the current task's independently admitted **last-observed object
  location** even when current visual verification is false. Query-memory
  invalidations/rejections/restored checkpoints and missing references still
  reject the selected location. Label/coordinate validation excludes camera
  nodes, another object and nonfinite XYZ.
- Result metadata separates last-observed location provenance from current-view
  verification. The semantic answer is unchanged. `prepare_query_target` still
  reacquires fresh geometry before manipulation; find success is not grasp/place
  authorization.
- Model-final advertises one investigate operation. The old navigate alias
  remains callable for compatibility. Legacy/reference tool packs are preserved.
- Find search memory omits count/MCQ instructions. Semantic EQA retains its
  existing memory instructions. Dialogue, measured receipts and image recall
  remain available; no extra coordinator or automatic retry policy was added.
- The existing serial room runner accepts explicit `--vl-endpoint URL#MODEL`.
  Ambient endpoint variables are still cleared. Its manifest records remote
  model identity and marks precision/decoding unverified rather than assuming
  that an endpoint name proves effective model settings.

First CPU gate: 386 tests pass; two simulator opt-in tests skip. Coverage includes
find/OVMM consumers, invalidation and checkpoint restoration, manipulation
reacquisition, EQA answer/room routing, memory rendering and runner configuration.
`/tmp/agent-simplification-contracts-20261116.xml` records the result.
Small live qualification remains a distinct gate; CPU success is not task success.
The first live check is now complete: bowl passes and microwave fails (1/2),
without OOM. Its audit and limitations are in
[the development report](../experiments/agent_simplification_20261116.md).

## Freeze before comparing models

Use the same committed harness and prepared context for every backend. Store the
source SHA, config/driver hashes, exact interleaved text/images and raw responses.
Treat a backend change as a whole-backend comparison unless all differences are
verified. In particular, record:

| Setting | Required evidence |
| --- | --- |
| Model | Served model ID plus checkpoint revision when available |
| Precision | Effective dtype/quantization; otherwise `unverified` |
| Decoding | Effective sampling/temperature/top-p/token cap; not just requested values |
| Vision | Identical source PNG hashes, image order/roles, request resizing/encoding; server preprocessing when available |
| Context | Same system text, dialogue, images and state; no scorer labels or future frames |
| Execution | Same simulator, robot, scene/start, physical controller and action limits for live pairs |
| Cost | Calls, input/output tokens when reported, wall time, errors/timeouts; simulator time separately when available |

Start with local Qwen3-VL-8B int4 and the existing Caliban Qwen endpoint. Caliban
HTTP is healthy at `192.168.1.55:8000`; its dtype/revision are not independently
verified. The checked-in Jetson server uses greedy generation, while the local
backend can inherit sampling defaults. Resolve/record that difference before
calling results a quantization ablation. See the
[complexity review](../experiments/agent_complexity_review_20261116.md).

A stronger supplied model, including a hosted model, is a separate optional
profile using the same OpenAI-compatible boundary. Do not change prompts,
detectors, graph policy or physical limits to help a particular model. Keep API
credentials outside configs/traces. Endpoint availability is not approval for an
unbounded sweep or a different model's result to replace a failed trial.

## Stage 1: fixed-context triage

Initially use **four packets × two backends × one trial = eight model calls**.
This stage proposes actions but executes none, so it requires no simulation.

| Packet | What to inspect |
| --- | --- |
| Task switch after bowl completion | Does the coordinator follow the new microwave request without confusing earlier bowl images/IDs with its target? |
| Microwave visible, geometry rejected | Does it distinguish semantic presence from metric admission and ask for a feasible closer view? |
| Motion rejected for unknown floor | Does it read the actual rejection, avoid claiming movement, and choose an appropriate sensing/motion tool? |
| EQA requiring a historical image | Does it retain the task/history and answer from evidence or request the missing image? |

Existing Molmo trace inputs are under
`~/runs/emet/simplified-find-20261116-r1/` and
`~/runs/emet/context-image-ids-find-20261115-r5/`. The earlier out-of-range
microwave examples are under `~/runs/emet/image-context-find-20261115-r1/`.
Freeze the EQA history packet from existing pilot evidence before inference;
record all selected packet paths and hashes in a single case manifest. These
correlated development examples diagnose interfaces, not benchmark accuracy.
The camera-turn handoff case is covered independently by deterministic contract
tests; retain it as a regression rather than spending model calls to validate
the same adapter rule again.

`scripts/replay_vlm_context.py` prepares/audits packets without model calls by
default. Its opt-in `--vl-endpoint URL#MODEL` mode now makes one bounded call on
the exact recorded packet, using the existing message client, with wire-body,
raw-response, usage and failure capture. It executes no tools and makes no scoring
calls. It does not introduce a second conversation implementation. Freeze the
prepared packet once and send the same packet to every model; do not separately
regenerate random image/message IDs for each backend.

Score schema/argument validity, referent correctness, whether the requested tool
is available, and whether the decision addresses the measured failure. Preserve
raw responses and manual assessment notes. Allow multiple reasonable next
actions; do not score one exact action sequence as the only valid plan. Count
every failed/timeout/refused call. Do not equate a plausible action with a safe
or successful executed action.

If sampling is enabled, repeat the **whole selected set** three times (24 calls)
under a predeclared policy, rather than selectively rerunning bad outputs. If
decoding settings cannot be matched, report backend differences explicitly.
Set token/time/call limits before launching; retain partial cohorts and failures.

## Stage 2: small closed-loop capability checks

Only after the fixed-context adapter works, test **three starts × two backends ×
three trials = 18 short episodes**, serially:

1. A visible target with valid depth; establish that admission and find handoff
   work without search. Turn away after admission in a separate deterministic
   contract test; this must not authorize stale manipulation.
2. A visible target initially outside the support range with a known feasible
   closer observation. Measure the model's selected action, actual displacement,
   range change, new image and subsequent grounding. Do not substitute a scripted
   recovery for the model's requested action.
3. A valid partial-floor scene with a rejected movement. Measure whether new
   sensing establishes floor support and enables a validated next action.

Use the existing simple scene and Molmo fixtures. Fixture reachability checks
may use simulator truth to qualify the test; agent inputs and decisions may not.
Capture before/after images, measured motion receipts, geometry admissions,
invalidations, final answers, and last-observed/current-verification status.
Keep find, metric localization and pick/place scores separate.

The explicit endpoint option is ready for the existing room find pilot:

```bash
# Execute under the existing emet jobs CPU/GPU admission wrapper, serially.
# Both checkouts must be clean; use fresh output directories for every trial.
PYTHONPATH=src .venv/bin/python scripts/run_shared_room_pairs.py \
  --main /path/to/frozen-control --candidate /path/to/frozen-candidate \
  --arms candidate --decision-policy model_final --grounding-backend vlm \
  --episodes molmo_rby1_ithor_s2_idx0 --seed 1 --scan-steps 4 \
  --max-rounds 12 --max-nav-steps 8 \
  --vl-endpoint 'http://192.168.1.55:8000/v1#Qwen/Qwen3-VL-8B-Instruct' \
  --output /path/to/fresh-output
```

This is a find-only complete-system run, not an isolated precision or placement
test. The endpoint supplies the shared VLM roles, including coordinator and
perception/grounding calls. Stage 1 isolates coordinator choices using frozen
tool receipts; live results must be labeled as a whole-VLM-backend comparison
unless role-specific clients are explicitly held fixed. A seed request does not
establish deterministic remote decoding. The
current runner timeout is 1,500 seconds per invocation; qualify remote latency
with packet replays first, and retain timeouts rather than extending limits after
observing a failure.

## Stage 3: shared-harness acceptance, then scaling

Use a frozen six-question EQA pilot spanning history, count/detail and spatial
questions; one small Molmo find task; one easier RoboCasa find/manipulation task;
and a minimal already-merged TAMP regression. Pair candidate/control and backend
conditions separately. Favor stable existing fixtures over the tiny/hard objects
previously identified as poor development controls. Habitat OVMM remains a later
long-horizon benchmark, not this slice's merge gate.

Record whether differences arise from context reading, requested action,
execution, grounding, handoff or final answer. If a better model succeeds at
fixed packets but fails live, inspect new views and measured execution before
changing the prompt. If models fail the same contract case, repair the common
interface. If valid contexts produce different choices, preserve those as model
outcomes. Do not accumulate model-specific exception rules.

Scale to the 12-question pilot and broader benchmark battery only after these
cases are understood. Report paired outcomes and denominators; repeated trials
of one scene are not independent environments. Paper claims should distinguish
better tools, better model inference and improved end-to-end task success.

## First scheduled cohort

The first four cases use three exact prepared requests from
`simplified-find-20261116-r1` (microwave calls 1, 10 and 5: task switch, rejected
geometry and rejected motion), plus the qualified `cap-dev-01` recorded recall
fixture from `capability-dev-render-colors-20261028`. The latter has 2,760 target
pixels in the earlier frame and zero in the later frame; it tests historical
evidence without another simulator rollout. Fixture qualifications and review
notes stay outside model messages.

Preparation freezes both PNGs/text and image IDs once. Every backend receives
the same serialized messages; transport hashes verify this after inference.
Budget: four calls per backend, 512 output tokens and 240 seconds per call,
zero retries. Failed/truncated calls remain failed rows. A timeout stops that
phase from submitting more work into a potentially still-running model;
unattempted cases remain explicit rows. This is development triage, not benchmark
accuracy, and the controlled recall fixture is not a natural-environment episode.

Local Qwen runs under the existing exclusive GPU lease on safe CPU cores. The
Caliban phase waits for the local supervisor to exit and then runs without the
local GPU lock, allowing other local GPU work to resume. No live agent, physical
tool or simulation is scheduled by this cohort. The durable protocol, frozen
inputs, runner, test report, model metadata, wire requests and results live under
`~/runs/emet/frozen-model-triage-20261116-r1/`; job IDs are recorded there at launch.

The first comparison and follow-up image-history audit are complete; see
[the audit](../experiments/history_context_audit_20261116.md). Historical pixels
reach local preprocessing, but current-turn presentation changes the response.
Recall can confuse current presence on local int4, while Caliban passes that
control. Neutral acknowledgement alone does not explain the original failure.
Any replay that appends recalled images must avoid falsely claiming all attached
images are in acquisition order. Keep source capture order separate from
presentation order and retain both historical and current-view negative controls.
Next compare packet placement on frozen trajectories before another live sweep;
do not interpret the first eight replies as a general precision comparison.

## Independent operator qualification

The requested external-operator trial is complete on the unchanged production
CHAT loop; [report and evidence](../experiments/independent_operator_trial_20261116.md).
It does not establish task success: head commands remained unconfirmed and a
failed FIND ended the coordinator turn with a misleading generic success summary.
Before another live accuracy sweep, qualify measured gaze and preserve detailed
mapped-tool feedback through the coordinator loop. Then repeat the bounded
operator case and small paired EQA/OVMM gates. The frozen packet-placement study
remains a separate model-context comparison; it cannot repair actuation failures.
