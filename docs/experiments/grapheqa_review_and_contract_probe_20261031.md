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
