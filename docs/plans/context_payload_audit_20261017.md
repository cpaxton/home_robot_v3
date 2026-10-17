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

Inspect completed replay and captured router requests before changing behavior.
Likely narrowly scoped follow-ups are complete, bounded tool-result history
(including no-op reads), explicit current-versus-historical image provenance,
explicit rejection/support of requested tool arguments, and simpler evidence
instructions evaluated on matched live episodes. Do not add a
semantic override, assume unknown floor is free, or infer object absence from
an incomplete image packet. New prompts must be evaluated online before promotion.
