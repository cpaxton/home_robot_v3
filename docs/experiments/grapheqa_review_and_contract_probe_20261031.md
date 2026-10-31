# GraphEQA review follow-up and frozen planner-contract probe

This follows the [active qualification](grapheqa_active_qualification_20261030.md).
The current three-case diagnostic remains unqualified for active-agent claims.
This work addresses API review findings and tests one explanation for the empty
plans without changing navigation, the answer arbiter, or benchmark defaults.

## Review disposition

| Point | Resolution |
| --- | --- |
| Missing `cfg.vlm` during room enrichment | All in-tree enrichment callers provide it. Mapping-only construction works without it. Enabled enrichment now gives a clear configuration error before output creation when the model is missing; the provider client is created only on actual inference. No hidden GPT fallback. |
| Implicit null inheritance / profile mutation | `with_model_config` returns an immutable merge. Omitted/null `room_model` explicitly means the profile's primary model and clears old room overrides. Named overrides remain supported. |
| Non-identifier Enum IDs | The reported failure does not reproduce: functional Enum accepts hyphens and leading digits. Tests preserve those IDs through schema validation and target selection. Native IDs are prefixed; no renaming layer is added. |
| Strict JSON Schema portability | Documented the required provider/model contract and nested alternatives. SDK-generated schemas close all objects and require their properties in every planner variant. Empty plans stay legal. OpenAI-compatible transport is insufficient by itself; alternate modes stay explicit. |
| Full-image trace volume | Documented base64 storage overhead and compressing completed evidence. Images remain in requests/traces; changing inference inputs is not a logging optimization. |
| Editor swap files block source freezing | Added only dot-file `.swp`/`.swo` ignores. A real temporary-repository test proves swaps pass while untracked Python still fails. |

Validation: 40 fork tests pass, including ten tests against the real CPU scene
wrapper/configs. The reviewed mapping-only wrapper also passes the native
container replay: ten cached learned packets produce current state plus a
frontier; output `/tmp/grapheqa-reviewed-stream-9c3p5pm7/`. This is replay
validation, not a new segmentation/navigation episode.

The review text predates the latest fork: independent multi-image attachments,
active sensor plumbing and the corrected q2 scorer write-up were already added.
q2's dataset label is C; its old saved scorer correctly marked both guesses wrong.

## Predeclared narrow experiment

Hypothesis: the model's planning/stopping contract is buried in a large prompt
and mostly undescribed response schema. It may fill `steps: []` as a formatting
default rather than deliberately choose to stop observing. Unbounded graph
descriptions also caused one response to enumerate invented IDs to its token cap.

Replay all three native-controller requests from
`grapheqa-four-images-context-pair-20261030/paired/`, including any failed rows.
Compare the exact original request with one that changes **schema descriptions
only**, placing the existing action/stopping and incomplete-evidence semantics
next to their fields and requesting brief evidence summaries.

Constants: four independent images and their labels/order, graph, question,
choices, legal IDs, allowed empty plans, generation budget, loaded Qwen3-VL-8B
int4 checkpoint and 512-pixel per-image preprocessing. Six calls run serially;
variant order alternates across cases. No scorer or gold answer file is read.
The annotation helper has a unit invariant: removing descriptions yields the
identical original schema. History/images/model/budget also remain unchanged.

The probe invokes the real `LLMServeState.complete` inference/message conversion
in process. It does not use HTTP transport or mock model replies; transport was
qualified separately. Save complete input/output, measured token usage and
format failures. The source runtime planner remains unchanged.

Measure valid responses, model-chosen actions while uncertain, empty uncertain
plans and confident stops. Keep answers exactly as emitted. A selected action
would justify a fresh live integration check; it is not itself evidence of
arrival or correct EQA. A small positive result would need broader validation,
not a default flip or a paper-superiority claim.

## Description-only result and next ordering probe

Job `20261031_150511_b71068` completed all six calls at source `bd2a30cb`.
Both conditions produced valid JSON in all three cases, but neither proposed
any action. q2/q6 admitted insufficient evidence; q12 still treated missing
bedside-table labels as absence. These annotations are not promoted to runtime.
No gold labels were loaded or correctness improvement claimed.

The responses expose another testable ordering issue: `steps: []` is emitted
before the explanation concludes that further exploration is needed. The next
comparison changes only root property/required ordering to put `answer` before
`steps`. It adds no wording or constraints and leaves every valid answer/action,
including empty plans, legal. Images, all other messages, graph and model/output
budgets remain unchanged. The same six-call counterbalanced replay is used.
This is a hypothesis about the interface to an autoregressive model, not an
established explanation; no runtime change is made without a positive result
and a subsequent fresh live integration check.

## Ordering result

Job `20261031_151001_789df0`, source `332f63ad`, completed all six calls. The
original-order replays again selected no actions. Answer-first q2 and q6 both
reported insufficient evidence and proposed `frontier_0`, a valid observed
action. q12 emitted a step while still confident; the existing controller
would stop, so it is not counted as a usable navigation proposal.

This is a two-case integration signal, not an accuracy gain or a statistical
result. Explanations remain weak and q12's stopping error is not repaired.
The fork now orders the same Pydantic fields as answer, then steps in all four
response variants for a fresh q6 live check. No descriptions or constraints
from the failed first probe are added. Allowed empty plans, model answers,
action IDs and the controller's stopping rule are unchanged. The live check
uses the normal current-view input, so it is a wiring gate rather than another
matched replay or evidence of a causal task-performance effect.

## Fresh current-image check: active gate still open

Job `20261031_151544_fbdafa` used emet `6660b780` and fork `62ee538` on q6,
with two decisions available. Transport/typed-response validation completed, but
the model declared confidence 0.9, selected C (wrong), and also proposed
`frontier_0`. Its explanation incorrectly treated missing ladder detections as
enough evidence to answer. The existing confidence stopping rule took precedence;
zero motion primitives executed. The proposed step is **not** an executed action.

`interop.status: passed` only means the checker/episode completed; it does not
mean active exploration or task success. The saved outcome explicitly reports
`model_stopped`, `primitive_count: 0`, and `correct: false`. No answer or action
was replaced. Owned model server and native worker exited cleanly.

The positive frozen-input signal used four images; this check used one. The next
qualification explicitly supplies a bounded, task-independent uniform history:
CURRENT first, then three uniformly spaced earlier captures from the same
episode. For 17 captures these are array indices 16, 0, 5, 10. Each has its own PNG,
sequence, timestamp and estimated camera pose. Default current-image behavior
remains unchanged (`--max-images 1`); the check opts into `--max-images 4`.
The active adapter snapshots each assessment's packet and replaces it on the
same controller, preserving measured motion history. Tests exercise the real
planner/provider adapter across two captures, checking image bytes/order and
the motion receipt, and reject a packet whose first image is not current.

Run q6 first with two decisions. Keep the existing model stopping rule, motion
safety, generation budget, checkpoint and scorer separation. This still tests
the interaction gate, not a matched active comparison or task-quality gain.
Only widen the pilot after a model-selected action reaches a fresh assessment.

## Four-view live result and final disposition

Job `20261031_152415_f6d8a2` used emet `5be8cdcc` and fork `3eb0c82`.
All four independent attachments reached the model and the reply validated.
q6 again chose C (wrong), declared confidence 0.95 and proposed `frontier_0`.
It stopped immediately: one decision, zero executed primitives, 23.33 seconds
episode wall time. The server and native worker cleaned up normally. This
does **not** close model-selected-motion acceptance.

The request audit compared this run against the positive frozen ordering replay:
the schema, agent-role prompt and question were exactly identical. The current
image and first history image were byte-identical too. The replay used frames
17/1/7/15 with concise XY/yaw labels; the live uniform control used 17/1/6/11
with full estimated-pose labels and fresh timestamps. Consequently the live
check is not an exact repeat of the positive request, nor does it isolate the
effect of field ordering or additional images. No attachment was dropped. The
current image was inspected manually: a partial stove/plant/adjacent-room view,
insufficient to establish the ladder's room. The model incorrectly turns a
missing learned detection into a confident answer about the wider scene.

Do not promote the ordering effect as a reliable repair. Fork commit `c41439f`
restores the original default response-field ordering. The ordering-only helper
remains in the emet diagnostic, so later model comparisons can reproduce it.
The failed description additions also remain experimental only. The optional
bounded history adapter stays available with the single-image default unchanged.
No stopping/answer override, forced action, target-specific fallback or safety
relaxation was introduced.

Final focused validation: **42 fork tests** and **22 emet sensor/diagnostic tests**
pass. The reviewed native ten-packet replay also passes. These qualify API,
configuration and evidence plumbing; the active task baseline remains unqualified.
Keep the two failed live checks and all frozen replies in the evidence record.

Next gates, in order:

1. Establish model-chosen observation gathering under a frozen model/profile
   and useful sensor coverage. Compare identical recorded requests when testing
   prompt/schema changes; do not infer stability from one development replay.
2. Qualify the emet policy with the same estimated-pose basis, footprint,
   perception and movement budgets. Sharing executor code is insufficient.
3. Run the preselected paired active pilot and matched-history image-only
   controls, retaining errors and abstentions. Only then expand the paper row.

No full sweep or task-quality superiority claim is warranted by these results.
