# Context payload audit — 2026-10-17

## Question and scope

Does the shared harness present accurate observations, explicit uncertainty,
useful history and actual tool outcomes to Qwen? Do not replace its answer to
improve scores. Preserve safety gates. No full sweep or hardware run.

Pilot source `0873430d`, artifacts
`/tmp/emet-final-model-pilot-results-20261016`: EQA 6/12, S0 both find targets
localized, RoboCasa/Molmo neither. All episodes completed; aggregate exit 1 is
the failed task gate, not an infrastructure crash. This is not manipulation
acceptance or a matched comparison against the older 9/12 profile.

## Observed contracts and gaps

- Shared Qwen inference uses `reset_context=True`. It is not a persistent full
  dialogue. Router state explicitly renders at most six recent action summaries.
  This is distinct from the older shared-client context-clobbering bug.
- The final answer's HISTORY section is empty in all six diagnostic cases
  (q6/12/14/25/56/65), despite preceding exploration/investigation actions. Its
  history is prior EQA-query history, not the router/tool transcript.
- Original Molmo object search: four router calls choose inspect_graph; one
  deterministic inspection-no-progress fallback chooses investigate. Only that
  movement attempt produces an action-history record. Repeated graph inspection
  is therefore not represented in Recent actions. Do not assume that removing
  the fallback helps: in this trajectory it caused the only approach attempt.
- The failed Molmo approach is *not* described as arrival: subsequent state
  reports no_target, failed_move and insufficient_floor_coverage, including 116
  rejected unobserved-footprint samples and nearest reachable distance ~1.85 m.
  It is incorrect to attribute this directly to Qwen missing the object.
- RoboCasa state does preserve safety-rejected navigation and failed progress.
  Its starting assessed image is a close view of a paper-towel holder, sink and
  cabinets, not a clean jar view. The benchmark passes the literal goal `cab`;
  this abbreviated simulator name is a language-quality concern, not proof of
  the failure's cause.
- q14 final text declares bathroom/kitchen/living_room as question scope, although
  those rooms originate in competing answer choices. It also explicitly prefers
  option landmarks visible in Image 1 for location questions. These are potential
  misleading cues, not established causes of the wrong answer.
- Manual q14 inspection: Image 1 contains the standing fan and a plant, not a TV.
  Historical images 3/4 do show the TV/living area. The recorded model attributes
  a TV to Image 1. This is cross-image attribution trouble with competing spatial
  answers, not evidence that all context images lack a TV.
- Manual q56 positive control: current bookshelf close-up alone is poor room
  context; historical Image 2 clearly supplies the blue curtain next to it. The
  correct response explicitly uses Image 2. Historical context is useful here.
- Original router traces save state text but not the exact attached RGBs or full
  caption prefix. `_live_rgb` also has a historical-graph fallback whose output
  is captioned as current. That is a code-path freshness risk; the pilot traces
  do not establish that this fallback actually occurred.

## Frozen-image experiment

Six targeted cases (five failures and q56 control), four text arms, three seeded
repeats, same image pixels/order and model. Selected for diagnosis, not an
unbiased estimate of accuracy. Gold is used only by grading, never in prompts.

| Arm | Changed input |
| --- | --- |
| original | Reconstructed production system + saved user prompt |
| neutral_user | Remove inferred QUESTION_ROOM line and landmark-preference phrase |
| image_only | Keep exact question/options and image provenance; remove graph/memory/status text |
| neutral_system | Keep original user text/images; replace system with concise evidence/uncertainty/JSON contract |

The neutral-system arm changes a whole system prompt, not one isolated rule.
None of these arms changes views, navigation, stopping or scored answer authority.
They cannot demonstrate an online task gain or repair missing observations.

Source branch `test/context-audit-20261017`; script
`scripts/replay_final_context.py`; two focused CPU tests check input identity and
gold exclusion. Lossless gallery:
`/tmp/emet-final-context-gallery-20261017/index.html`.

Initial job `20261017_024105_9ba528` was cancelled and excluded. It loaded generic
EQA settings instead of Habitat's post-config hmeqa variant, and omitted the
384-token answer cap/JSON assistant prefill. Retained partial output has explicit
`disposition.json`; its scores must not be used. Corrected source `07e65e26`
matches the pilot's logged hmeqa/json/384/prefill settings and saves them.
Corrected job: `20261017_024513_ec837d`; output:
`/tmp/emet-final-context-replay-v2-20261017`. Completed: 72/72 parseable responses.
The subsequent CPU regression also exercises the reconstructed hmeqa variant,
384-token budget and JSON prefill without loading a model.

### Completed fixed-image results

Each entry is correct out of three repeats, not independent scene trials.

| qid | original | neutral_user | image_only | neutral_system |
| --- | --- | --- | --- | --- |
| 6 | 0 | 0 | 0 | 2 |
| 12 | 0 | 0 | 1 | 0 |
| 14 | 2 | 2 | 3 | 3 |
| 25 | 2 | 0 | 2 | 3 |
| 56 | 3 | 3 | 3 | 3 |
| 65 | 0 | 0 | 0 | 2 |
| total /18 | 7 | 5 | 9 | 13 |

The complete system contract matters more here than removing the two user-text
cues alone. Do not conclude general superiority: this is a small, deliberately
failure-heavy subset, source/config-reconstructed replay and a whole-system
prompt change. Original q14 is itself 2/3 correct, unlike the one recorded live
miss. This does not establish a deterministic packet regression.

Manual correctness checks qualify the apparent gain. Neutral-system q25 invents
a bathtub/rack in its explanation; its score is not a trustworthy grounding
certificate. Neutral-system q65 abstains once but twice correctly guesses off
because no light/airflow is visible, which is insufficient operational evidence.
q12 counting remains unresolved. The positive q56 control stays 3/3 in every
arm and genuinely uses blue curtains in a historical image. Therefore preserve
useful context, remove misleading instructions only with controlled validation,
and do not use answer accuracy alone to validate the harness.

Saved prompt.txt is not a complete wire trace. The replay reconstructs the system
from frozen code and runtime configuration. Input block serialization and
in-process cache/trajectory state are not claimed to be identical to the live run.

## Exact router payload capture

Trace-only source `c1987278`, branch
`test/router-context-capture-20261017`: save system prompt, full user text with
image captions, ordered PNGs, pixel hashes, round/call identity and token budget
before the router call. Boundary is client payload before model preprocessing,
not encoded tensor or wire capture. No prompt/policy/default change. Nine
trace/router tests pass. Capture failures are explicit rather than silently
presented as complete evidence.

Single Molmo rerun `20261017_024307_0a9c99`, output
`/tmp/emet-molmo-router-context-20261017`, queued/run under the same exclusive GPU
lock as replay. No simultaneous high-load experiments. Inspect captured requests
alongside actual executed tools, not just the model's requested tools.

### Completed live capture and router replay

The trace-only Molmo run completed without an infrastructure error and again
failed both find targets. Eight complete requests were captured. Object search
again chose inspect_graph four times; movement was a deterministic fallback.
The last object request contains a clear kitchen RGB showing a bowl on the
counter and a microwave above the stove. The system exposes verify_siglip on
the live camera, but its higher-level rule says to choose investigate or explore
and describes assessment after movement. Thus current-view verification exists
but is obscured by prescriptive routing instructions. This is not evidence that
the model cannot see the bowl; nor does visible recognition certify its geometry.

Fixed-router replay `20261017_024800_98700c` uses the last captured object and
receptacle requests. Arms: exact recorded text; same text plus factual earlier
tool-result rows (including non-motion graph reads); same user text with the
prescriptive Rules/examples replaced by neutral task/evidence/tool guidance.
Images, tool descriptions and 128-token budget stay fixed. Three repeats/arm.
Raw requested tools are retained; no actions execute and no task-success claim
is possible. Output: `/tmp/emet-router-context-replay-results-20261017`.
One-off replay source: `/tmp/emet-replay-router-context-20261017.py`; its hash and
client/config revision are recorded in output provenance.

Completed: 18/18 responses contain parsed tool calls. Final selections:

| Captured request | original | factual tool history | neutral rules |
| --- | --- | --- | --- |
| bowl search | inspect_graph 3/3 | investigate 3/3 | inspect_graph 3/3 |
| microwave search | inspect_graph 3/3 | inspect_graph 3/3 | inspect_graph 3/3 |

History changes a decision on the identical bowl view, but not universally.
Investigate still targets the existing bowl proposal; this offline experiment
does not show that another approach succeeds or resolves the missing floor.
Removing the prescriptive rules by itself is not a demonstrated router fix.
Version-controlled replay driver is now on
`test/router-context-capture-20261017` at `8216252f`; the exact original one-off
source/hash used for inference remains retained in the temporary input/provenance.

An additional API-feedback gap is visible in the replies: Qwen repeatedly sends
inspect_graph(query=...), but inspect_graph advertises no parameters and
`_handle_inspect_graph` ignores its args. It therefore receives no explicit
feedback that this requested query variation was not applied. This alone is not
proof of the search failure (the current question already names the same target),
but silently accepting unsupported arguments is misleading tool behavior.

## Next decision

### User-prioritized repair — before instruction changes

Branch `fix/agent-conversation-contract`, frozen live-check head `e68d8f8c`:
06cf1eba repairs explicit multimodal conversation transport; 4e4dd204 connects
episode dialogue, tool results, final answers, strict argument names and traces.
No routing-rule or system-prompt rewrite. Conversation lives on the agent under
episode/session + map-epoch scope, including both OVMM find phases. Perception
inference remains isolated; new worlds/episodes cannot inherit old messages.

Remote client and VL server now preserve assistant turns and prior user images,
rather than flattening the request. Both sides of an old emet remote deployment
must be updated. Undeclared arguments produce explicit recorded errors before
execution; inspect_graph(query=...) is rejected, not silently applied/ignored.
Automatic recovery's private budgets are not exposed as model-controllable args.

470 focused tests pass, plus 47 room/navigation tests with one pre-existing
SigLIP-upgrade fixture failure explicitly excluded and reproduced on unchanged
parent 8216252f. New tests cover native image encoding, remote round trip,
perception isolation, assistant/tool history, final-answer transfer, episode/world
isolation, cross-OVMM-phase continuity, rollback and no-execution argument errors.

Live contract check `20261017_075712_91ad9d`: EQA q2/6/14/56 then one Molmo rby1
find, serial exclusive GPU. Artifacts:
`/tmp/emet-conversation-contract-check-20261017`.

Live EQA completed cleanly: 2/4 (q6 and q56 correct; q2 and q14 wrong), all
`model_final` provenance. Final-answer request exports retain respectively
22/7/4/4 history messages, including 7/2/1/1 assistant turns, 8/3/2/2 actual
tool-result messages and 7/2/1/1 historical images. This verifies continuity,
not an accuracy gain. The Molmo receptacle phase's first request retains ten
messages from the preceding object-search phase, including assistant turns and
failed-navigation feedback. Molmo completed without infrastructure errors:
bowl localization succeeds with `grounded_object`; microwave localization fails,
so the complete two-phase find gate fails (1/2 phases). Qwen repeatedly recognizes
the microwave above the stove, but the trace defers completion with
`object_localization_required`; no grounded receptacle XYZ is admitted. This is
not a demonstrated semantic-recognition failure, nor a successful manipulation
test (`manip_mode=skip`). Query time is 81.9 s, episode time 132.2 s. The driver
returns 1 for the failed task gate even though the episode subprocess exits 0.
No unsupported-argument call occurred in these inspected live EQA packets;
explicit rejection/no execution/error feedback is covered by regression tests,
not claimed as naturally exercised online. A fresh focused rerun passed 43 tests.
Inspect context/memory cost before deciding whether instruction changes are still
needed; full history is intentionally not silently truncated.

### Full development pilot queued

Job `20261017_081733_952378` (`history-paired12`) runs candidate `aeda57c4`
then fetched main `801dce82` under one exclusive GPU lock, CPU-safe, seed 0,
questions 2/6/12/14/15/16/25/28/31/56/65/68, local Qwen3-VL-8B int4,
20 planning / 10 movement budget, lazy/query-driven, no GT semantic enrichment.
Candidate retains `pilot_model_final.yaml`; main uses its native default config
because it does not implement the candidate's final-answer/view-selector policies.
This is a complete-system comparison, not a history-only ablation. The previous
candidate's 6/12 is the closer repair reference; single-seed deltas remain noisy.
Requested later history ablations are now tracked in candidate TODO.md with
argument rejection held enabled in all arms.

Output: `/tmp/emet-history-paired12-20261017/{candidate,main}`.
External frozen driver: `/tmp/emet-run-history-paired12-20261017.sh`; driver and
config hashes plus per-arm source SHAs/exit codes are retained with the outputs.
Completed cleanly: candidate 8/12, main 9/12, 24/24 readable episodes, no reported
episode errors. Previous candidate was 6/12: q6 and q65 improve, no scored losses.
Against main, candidate loses q12 and q14 and gains q65; all other scored outcomes
agree. Single seed; no superiority or non-regression acceptance claim.

Raw-output audit matters: main q12's final model says None, but `mcq_debias`
substitutes the correct Two. Main q65's model says off, but `mcq_debias` substitutes
the incorrect on. Candidate q65 preserves off but explicitly reasons from absence
of a visible AC unit: this is a scored win, not grounded state evidence. Main q14
final model and earlier assessment agree on living room; candidate final model
says next to TV, so q14 remains a meaningful image/context discrepancy to inspect.
Candidate q12 stops after one round at One versus main's eight rounds; inspect
coverage/stopping and actual image packets before changing prompts or policies.

### Fixed-packet history diagnostic

Manual inspection: candidate q14 current fan view is tightly framed against a
wall; pre-turn view shows sofa/window, and earlier context images show a TV and
the wider living room. Main's current image directly includes fan + sofa/window.
Candidate's retained investigate result also asserts Next to the TV. Cannot
attribute the error to image selection or history independently from this pair.
Candidate q12 current image shows only part of the bed/bedside area, insufficient
to certify the room-wide count. Neither observation justifies semantic overrides.

Job `20261017_091335_a234d7` tests candidate's captured q12/q14 final packets,
full vs tool-results-only vs no history, three repeats (18 calls), fixed saved
system/current images/token cap/prefill. Offline diagnostic only, not task success
or a proposed default history deletion. No routing instructions changed.
Driver `/tmp/emet-replay-captured-history-20261017.py`; output
`/tmp/emet-captured-history-ablation-20261017`. Input request and driver hashes
are retained. Completed 18/18 parsed responses, no infrastructure error:

| Case | Full history | Tool-results-only | No history |
| --- | --- | --- | --- |
| q12 count | 0/3 (One) | 0/3 (One) | 0/3 (One) |
| q14 fan | 0/3 (Next to TV) | 0/3 (Next to TV) | 2/3 (living room) |

History removal does not fix q12 on this packet. Q14 is consistent with anchoring
on the earlier VLM assessment carried in the tool result, not merely too many
images/tokens: tool-results-only retains the wrong answer across all repeats.
This is two selected cases, not a task-success or universal causal claim. Preserve
dialogue; next isolate the presentation of earlier model interpretations from
measured tool facts, without dropping failures, altering physical admission, or
silently overriding the final model. A proposed formatting change needs replay
and live validation; none has been enabled by this diagnostic.

Provenance-only follow-up job `20261017_095915_addd8d`: q12/q14 plus successful
q6/q56 controls, unchanged full history vs identical history with explicit
prior-VLM provenance labels, three repeats each (24 calls). All original messages,
images, tool failures, semantic assessment fields and answer suggestions remain;
no production change or semantic override. Transform sanity check confirms input
immutability and preservation of message roles/count and failure/assessment values.
Driver `/tmp/emet-replay-captured-history-20261017.py --label-assessments`, output
`/tmp/emet-assessment-provenance-20261017`; output contains driver/input hashes.
Results pending.

Inspect completed replay and captured router requests before changing behavior.
Likely narrowly scoped follow-ups are complete, bounded tool-result history
(including no-op reads), explicit current-versus-historical image provenance,
explicit rejection/support of requested tool arguments, and simpler evidence
instructions evaluated on matched live episodes. Do not add a
semantic override, assume unknown floor is free, or infer object absence from
an incomplete image packet. New prompts must be evaluated online before promotion.
