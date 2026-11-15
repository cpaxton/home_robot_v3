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

`emet.llms.image_context` now makes a request copy for episode-owned
`VLMConversation`, shared by EQA and OVMM find phases:

- Keep every original system, assistant and tool-result turn. Original dialogue
  and image snapshots remain owned by the episode; provider perception calls
  cannot replace them.
- Keep all images in the latest request, including explicitly recalled images.
  Fill the remaining slots with recent distinct historical pixels, within eight
  attachments by default. If mandatory images exceed the limit, fail explicitly
  rather than silently discard current evidence.
- Add stable episode-scoped view IDs and local message image slots. An omitted
  image becomes a text reference at its original location. Pixel equality does
  not establish acquisition-time, pose, geometry or object-state equality.
- `recall_view(view_id)` queues exact retained pixels for the next turn, labeled
  historical. It performs no movement, no new capture, no visual verification
  and no geometry admission. Unknown/reset-episode IDs return `UNKNOWN_VIEW`.
- Record the prepared packet, its selected/omitted image audit, raw pixel hashes,
  and interleaved current content before provider resizing/tokenization. This
  boundary is still not a token or HTTP-wire trace.

`mapping.eqa.agentic_context_max_images` is configurable; `null` keeps all
attachment occurrences for the full-context ablation. Eight is a conservative
resource setting, not a tuned optimum or a universal GPU-memory guarantee.
Original text remains intact and is not token-budgeted by this change. The
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
