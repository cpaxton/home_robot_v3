# Auditable image packets and historical-view recall

The seed-1 find repeat exposed an input-resource failure, not an established
model-quality regression. The last microwave request held 46 history messages,
28 historical image attachments (15 distinct pixel hashes) and two new images.
The server raised CUDA OOM. Historical tool results and positive evidence must
survive; resetting the conversation is not the repair.

## Three actual trace excerpts

Source candidate: `cd03fad7`, episode `molmo_rby1_ithor_s2_idx0`, seed 1,
12 rounds / 8 navigation calls per phase. Full artifacts:
`~/runs/emet/recovery-budget-high-20261115-r2/`.

1. **Bowl admission, round 0.** `investigate` reports failed navigation but a
   fresh captured view independently admits object observation 7:

   ```json
   {"tool":"ground_query_view","round":0,"view_obs_id":1099511627781,"ok":true,"obs_id":7}
   ```

   The next state says `Selected object geometry ... admitted, object_obs_id: 7`.
   The model nevertheless selects `rotate_base(degrees=90)`. Earlier pixels and
   the failed movement remain relevant facts, not mutually exclusive outcomes.

2. **Bowl continuation, round 6.** Another turn produces:

   ```json
   {"tool":"ground_query_view","round":6,"view_obs_id":1099511627790,"ok":false,"reason":"Too many proposals; request another view"}
   ```

   The next state has no selected geometry. The earlier object still exists in
   memory, but its selected-result pointer was cleared. The model later gives
   its semantic location while terminal localization fails. This deserves a
   separate evidence-retention repair; image budgeting does not manufacture a pass.

3. **Microwave, fourth coordinator request.** History alone contains 28 images;
   two more are attached to the new request. The packet includes old bowl-task
   dialogue and measured movement failures. The active question is explicitly
   `Where is the microwave?`. The 30 image occurrences contain 16 distinct pixel
   identities overall. The raw history text is about 78,000 characters, excluding
   the current request: image count is not the only eventual context constraint.

These are captured requests and selected actions, not invented successful plans.
Original requests live under the candidate artifact's
`router_context/find_object_agentic_trace/call_0002/` and
`router_context/find_recep_agentic_trace/call_0004/`.

## Packet contract

Current API names pixel identity `image_id` and uses `recall_image(image_id)`.
Historical prototype traces below retain their original `recall_view` name;
capture `view_id` and graph `obs_id` are separate identities.

`emet.llms.image_context` now makes a request copy for episode-owned
`VLMConversation`, shared by EQA and OVMM find phases:

- Keep every original system, assistant and tool-result turn. Original dialogue
  and image snapshots remain owned by the episode; provider perception calls
  cannot replace them.
- Keep all images in the latest request, including explicitly recalled images.
  Fill the remaining slots with recent distinct historical pixels, within eight
  attachments by default. If mandatory images exceed the limit, fail explicitly
  rather than silently discard current evidence.
- Add stable episode-scoped image IDs and local message image slots. An omitted
  image becomes a text reference at its original location. Pixel equality does
  not establish acquisition-time, pose, geometry or object-state equality.
- `recall_image(image_id)` queues exact retained pixels for the next turn, labeled
  historical. It performs no movement, no new capture, no visual verification
  and no geometry admission. Unknown/reset-episode IDs return `UNKNOWN_IMAGE`.
- Record the prepared packet, its selected/omitted image audit, raw pixel hashes,
  and interleaved current content before provider resizing/tokenization. This
  boundary is still not a token or HTTP-wire trace.

`mapping.eqa.agentic_context_max_images` is configurable; `null` keeps all
attachment occurrences for the full-context ablation. Eight is a conservative
resource setting, not a tuned optimum or a universal GPU-memory guarantee.
Original text remains retained; the separate working-text bound described below is a character proxy, not a tokenizer limit. The
independent CHAT/OpenaiClient path is not silently claimed to use this episode
adapter. The packet module is provider-independent and does not depend on a robot,
scene, object detector or answer rule.

## Offline examples and limitations

Replaying the saved requests makes no model calls:

| Request | Original image occurrences | Distinct pixels | Prepared attachments |
| --- | ---: | ---: | ---: |
| Bowl after admission | 3 | 3 | 3 |
| Bowl mid-search | 15 | 9 | 8 |
| Microwave before OOM | 30 | 16 | 8 |

![Bowl admission request images](figures/context_bowl_admission_20261115.png)

The first three images are nearly identical kitchen views, but their raw pixels
differ. The module intentionally does not infer that they are interchangeable.

![Bounded replay of the failing microwave packet](figures/context_microwave_packet_20261115.png)

The eight selected images still contain similar doorway/kitchen views. This is
an auditable recency baseline, not a task-usefulness claim. The model can request
an older view by its displayed ID. Optional pose/coverage/semantic ranking can be
compared later without changing dialogue ownership or physical permissions.

Prepared requests, selection audits and image sheets are at
`/tmp/context-audit-20261115/`. Reproduce one with:

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_vlm_context.py /path/to/request.json \
  --output /tmp/new-context-replay --max-images 8
```

Use `--full-context` for the attachment-count ablation. The script rejects an
existing output directory and never queries a model or reads gold labels.

Validation: **433 focused CPU tests pass**, including the real remote-client /
server translation boundary, retained tool text, unchanged caller state, recall
across inference failure, episode reset isolation, mandatory-image overflow and
prepared-trace equality. Report: `/tmp/image-context-tests-20261115.xml`.
Live memory and task acceptance remain separate gates; offline packet reductions
are not accuracy results.

## First live packet check

Frozen source `f822e696`, managed job `20261115_144905_3c165c`, reran the same
find episode with CLI seed 1, Qwen3-VL 8B int4, 12/8 action limits and unchanged
physical/geometry gates. All **15** coordinator requests have at most **8**
attached images, even when their original dialogue contains **30** occurrences.
Recorded prior-message counts match the complete retained dialogue. All 15
executed actions match model selections; there are no action rewrites.

The episode completed without an inference/OOM error: bowl passes, microwave
fails (**1/2** phases), 262.01 s query / 346.79 s episode wall time. This is not
an accuracy gain established against baseline; the earlier seed-1 candidate was
incomplete. Raw request text and action sequences differ stochastically.

The model chose `recall_view` for the bowl at round 1 using an exact valid ID.
That image was already in the preceding packet, so this demonstrates live tool
use, not adaptive retrieval of an omitted view. Contract tests separately cover
recalling a genuinely omitted image. The bowl then grounds and finishes at round 9.

At microwave round 3, the receipt reports `geometry.status=not_admitted`, with
microwave proposals at 3.366/3.450 m outside the unchanged 2.5 m depth limit.
At round 4 the model submits “Above the stove, mounted on the wall.” while only
three of eight navigation calls have been used. The answer is preserved; it
cannot supply the missing metric object geometry. The final request contains
the rejection and proposal bearings: the failure was not hidden from the model.

Artifacts: `~/runs/emet/image-context-find-20261115-r1/`. The compact packet/action
audit is `image-context-live-audit-20261115.json` in
`~/runs/emet/image-context-cohort-20261115-r1/`.

### Task-framing follow-up

The top of that request still says `Question: Where is the microwave?`, while
the metric-localization requirement appears later. The state now starts find
requests with the actual caller objective:

```text
Task: find and localize the requested object for downstream robot action.
Object request: Where is the microwave?
The requested result is an admitted object location, not just a semantic location description.
```

Semantic EQA keeps its `Question:` header. This changes task information, not
motion, answer selection or a stopping gate. The model can still choose its
answer/actions. The follow-up also makes occurrence pixel SHA-256 values match
the raw-pixel hash convention used by the trace images; dimensions remain part
of identity. **435 focused CPU tests pass**, including the find/EQA distinction.
Live task benefit of the header is still pending.

The header-only live follow-up at `a158a639` also completes without OOM and
scores **1/2** phases (107.32 s query / 158.39 s episode). Bowl finishes after
nine decisions; microwave receives an invalid-card rejection and then submits
a semantic answer after two decisions, with no navigation calls used. The
explicit objective alone did not establish a task improvement. Artifacts:
`~/runs/emet/image-context-goal-20261115-r2/`.

Auditing that run's **actual saved system prompt** also found a contradictory
legacy sentence in the `explore_frontier` capability description: “must
investigate unused voxel detections ... first.” The coordinator prompt says
there is no required sequence. The shared capability description now explains
frontier navigation and weak semantic bias without embedding routing priority.
Legacy execution policies are unchanged; no automatic navigation is added.
The actual assembled tool-prompt contract is covered by a regression test.
**436 focused CPU tests pass.** This consistency repair is separate from any
claim that Qwen will choose a better plan.

## Eight images alone do not bound working context

The final image-only consistency run at `353bb7af` completed the bowl but raised
CUDA OOM during the sixth microwave request. It is an **incomplete episode**, not
another completed 1/2 result. The attempted allocation was 822 MiB. All saved
requests respected the eight-image limit. The failed request's history contained
about **58,000 characters of older state snapshots**, **29,000 of tool results**
and **2,500 of assistant decisions**, before its current input and system prompt.
This invalidates any general claim that the eight-image limit solved memory use.
Artifacts: `~/runs/emet/image-context-policy-20261115-r3/`.

The episode now also has a **64,000-character working-text limit**. This is a
character proxy, not a tokenizer count or universal GPU guarantee. The policy:

- Keep the system prompt, current state, every assistant decision, and every
  plain-text/tool-result message unchanged.
- If the prepared packet is too large, replace the oldest multimodal user-state
  text with an explicit episode-scoped `message_id` reference. Retain its images
  subject to the independent image limit. The original full state stays in
  `VLMConversation.messages`; nothing resets the episode.
- `read_history(message_id, offset, max_characters)` returns the original text
  in pages, plus its image IDs. The model can use `recall_image` for those pixels.
  These information tools add no motion or geometry permission.
- If required current/system/action/tool text alone cannot fit, fail explicitly
  before inference, retaining the original dialogue. Do not silently discard a
  measured outcome to fit the budget.

Prepared traces record which state texts were archived and the actual working
character count. `source_user_text` also retains each original current state
before packet annotations, so the full state snapshots remain inspectable across
saved calls. The offline replay tool can bound both text and images;
`--full-context` disables both budgets **for content available in that input
file**. A prepared request alone cannot resurrect previously omitted pixels/text;
use original captures or the retained live episode archive for full ablations.

Configuration (set either value to `null` to disable that bound):

```yaml
mapping:
  eqa:
    agentic_context_max_images: 8
    agentic_context_max_characters: 64000
```

On a replay of the saved failing prepared request, the working text falls from
104,153 to **63,594 characters**, with eight images, without dropping any assistant
or tool-result text. The replay adds fresh annotations, so those character counts
are for the replayed packet, not the original server's token count. It makes no
model calls. **442 focused CPU tests pass**, including remote-wire/server text
bounds, exact history retrieval, paging/reset isolation and required-text overflow
before inference. A GPU resource probe and live follow-up are still required.

## Working-text validation and identifier correction

At frozen source `8ee6d898`, the resource-only probe replayed a bounded version
of the saved failing packet: eight images, 63,594 characters and **22,773 actual
prompt tokens**. With a synthetic **6 GiB GPU allocation** approximating competing
agent memory, generation completed using SDPA. Peak PyTorch reserved memory was
**18.61 GiB including that reservation** (17.47 GiB peak allocated). This is a
resource diagnostic; its proposed action was not executed or scored. Evidence:
`~/runs/emet/context-state-memory-probe-20261115-r1/`.

The subsequent physical-simulation find episode also completed without OOM,
again **1/2** phases (165.55 s query / 217.62 s episode). All 15 coordinator
requests stayed within eight images and 64,000 characters; the maximum measured
working text was **63,767 characters**. Up to eight earlier states were archived
by reference. A cross-request audit verified unchanged text for **31 assistant /
tool-result messages** and unchanged current inputs. All 16 executions belonged
to the model's selected action lists, without rewrites. Evidence:
`~/runs/emet/context-state-budget-find-20261115-r4/` and its sibling cohort's
`working-context-live-audit-20261115.json`.

The same audit caught an API naming collision introduced by the prototype. The
model requested `recall_view(view_id="voxel-frame:5")`, copying a real capture
receipt's `view_id`; the prototype pixel registry expected another ID namespace.
The request failed safely, but those names were misleading. The current API is
therefore **`recall_image(image_id)`**, where pixel IDs start `image_...`.
`read_history` returns `image_ids`. Capture `view_id`, graph/candidate `obs_id`
and archive `message_id` retain their distinct meanings. There is no guessed
compatibility alias and no change to physical permissions. Contract tests cover
successful image recall and rejection of frame IDs or the wrong argument name.

For example, the model can request:

```json
{"name":"recall_image","arguments":{"image_id":"image_<episode-scope>_0003"}}
```

or page an archived state:

```json
{"name":"read_history","arguments":{"message_id":"message_<episode-scope>_0001","offset":0,"max_characters":4000}}
```

Copy real IDs from the packet annotations; the placeholders above are not valid
IDs. Quoted historical tool calls preserve their prototype names. The image sheets
are offline replays using current pixel IDs; current inference receives the current schemas.

The memory improvement does not establish localization quality: microwave
geometry still abstained and the model chose a semantic final answer. Broader
paired EQA/another-room acceptance and retained-object selection remain open.
