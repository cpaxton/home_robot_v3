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
`/tmp/emet-final-context-replay-v2-20261017`. Pending at this checkpoint.

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

## Next decision

Inspect completed replay and captured router requests before changing behavior.
Likely narrowly scoped follow-ups are complete, bounded tool-result history
(including no-op reads), explicit current-versus-historical image provenance,
and removal of contradictory answer-pressure/location cues. Do not add a
semantic override, assume unknown floor is free, or infer object absence from
an incomplete image packet. New prompts must be evaluated online before promotion.
