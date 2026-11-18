# Qwen image-history audit

The first frozen-context comparison returned `Unknown` on a historical color
question for local int4 and Caliban Qwen. This audit separates image transport,
image placement, dialogue structure and temporal interpretation. Source:
`af6c8f18`. Agent policy and runtime defaults are unchanged.

## Evidence and transport

Earlier frame, with a green box face:

![Earlier recorded view](figures/history_before_turn_20261116.png)

Later frame, after a camera turn:

![Later recorded view](figures/history_after_turn_20261116.png)

The controlled fixture has 2,760 target pixels in the earlier frame and zero in
the later frame. Answer labels/qualification data never enter model messages.
These synthetic views are diagnostics, not natural-room benchmark episodes.

Both backends receive identical wire-message hashes for every paired condition,
including image bytes and ordering. The decoded earlier image retains its green
region; neither source image is dropped by budgeting.

The real cached Qwen processor, using local server/client conversion on CPU,
produces two image grids `[1, 24, 32]`, pixel tensor shape `[1536, 1536]`, and
**575 input tokens**, exactly matching the original local inference receipt.
Both user turns and vision placeholders survive. This argues against local
transport/preprocessing loss; it does not expose internal model attention.

The checked-in Jetson converter produces the same layout with that processor.
Deployed server/processor revision, precision and effective decoding remain
unverified: health does not expose them and available SSH credentials are
rejected. Remote usage fields are zero placeholders, not measured zero tokens.
This remains a backend comparison, not an isolated int4/fp16 ablation.

CPU evidence: `/tmp/history-preprocessing-audit-20261116.json`, also copied into
the history-layout cohort. Wire requests and raw responses remain in each arm.

## Presentation results

Each cell is an actual response from one completed short model call.

| Condition | Local Qwen int4 | Caliban Qwen |
| --- | --- | --- |
| Earlier image alone: box color | Green | Green |
| Both frames in one user message: earlier color | green | green |
| Same ordered text/pixels across two user messages, no labels | Unknown | Unknown |
| Original two-user packet with image-label boilerplate | Unknown | Unknown |
| Neutral assistant acknowledgement, keep labels | Unknown | Unknown |
| Neutral assistant acknowledgement, remove labels | Unknown | Unknown |
| Existing recall rendering: earlier color | Green | Green |
| Existing recall rendering: visible in later frame? | Yes | No |

The plain one-user/two-user pair has identical flattened text and pixel sequence;
only message boundaries differ. Removing annotations does not rescue the split
case. A neutral `Acknowledged.` assistant turn does not rescue it either, so
consecutive user messages alone are not a sufficient explanation. An older
October replay with acknowledgement answered green, but its prompt/serving
revision differ and are not matched controls.

The first triage fixture omitted an assistant turn, making it less representative
of a completed interaction. The acknowledgement controls test that weakness
rather than dismissing the original result. Its raw negative outcome remains.
No colour fact is supplied in the acknowledgement.

Recall is queued **by the test**, using production `VLMConversation.recall_image`
and packet rendering. Earlier dialogue survives; historical pixels are presented
again in the current user turn. This is evidence for a usable representation,
not autonomous coordinator recall or physical task success.

## Temporal-order control

The diagnostic system text asserted `Views are in temporal order`. This is true
for the original chronological pair, but false when recall appends an older image
after the current one. That was a mistake in diagnostic construction; it is not
an instruction this audit added to the production agent.

The final local controls explicitly use **greedy decoding**, recorded in model
metadata. Earlier cohorts use checkpoint sampling defaults. Caliban effective
decoding remains unverified.

| Later-frame presence question | Local greedy int4 | Caliban |
| --- | --- | --- |
| Recalled pixels, original global order claim | Yes | No |
| Same recalled packet, replace claim with frame-label wording | Yes | No |
| Both frames chronologically in one user turn | No | No |

Correcting the false claim is necessary for truthful inputs, but does not alone
fix local temporal confusion. Local int4 mistakes historical presence for current
presence in the recall condition; Caliban does not in these controls. That is a
real observed backend difference in one fixture, not proof of a precision effect
or a general success-rate estimate. Repetitions are correlated development probes.

## Implication

Qwen can read the relevant pixels. In this example, availability somewhere in
earlier dialogue does not ensure reliable use. Presenting the selected evidence
in the current decision packet helps historical recall, but stale images can
then mislead current-view judgments. Acquisition identity and historical/current
roles must remain explicit, without losing the original dialogue.

The smallest next experiment is a bounded current-decision image bundle using
the existing view-selection machinery: source-order labels, a clearly identified
current frame, original assistant decisions/receipts preserved. Compare fixed
trajectories with historical positives and current-absence negatives before
small paired EQA/OVMM checks. Do not blindly reattach every old image, invent
assistant acknowledgements in real sessions, force answers or add another
navigation recovery policy.

No runtime default flip or merge-readiness claim follows from this audit. The
mixed real-scene choices in the first four-context triage remain diagnostics;
this color test does not replace them.

## Cohorts and resource accounting

Completed audit: **22 model calls**, no tools or simulation:

- Eight layout calls: `~/runs/emet/history-layout-audit-20261116-r2/`.
- Eight acknowledgement/recall controls:
  `~/runs/emet/history-presentation-controls-20261116-r1/`.
- Six ordering controls: `~/runs/emet/history-ordering-control-20261116-r1/`.
  Its `history-audit-summary.json` records raw replies, result/wire hashes and
  separately labeled infrastructure stops.

Initial layout, isolated recall and isolated acknowledgement attempts stopped
before inference at the 16 GiB reserve: **14 not-attempted rows, zero model
calls**. These remain under their original directories and are not model errors.

Offline uv pruning removed 39,322 unreachable files (4.1 GiB logical size; shared
links mean less filesystem reclamation). Only old unshared pip HTTP/uv-index
responses were subsequently removed. Environments, weights, compiler caches and
run evidence remain. Final small controls predeclared a **15 GiB model-only
reserve**, after the completed layout cohort wrote about 106 KB. Physical,
model, input and output-budget limits were unchanged; stopped attempts were not
relabeled as completed calls.

Regression checks: **39 passed** across HTTP serving, Qwen multimodal history
and exact-packet replay (`/tmp/history-audit-regressions-20261116-sockets.xml`).
The initial sandbox run passed 38 but could not bind the loopback socket; the
same suite passes with local-socket permission. No implementation change was
needed for that infrastructure restriction.
