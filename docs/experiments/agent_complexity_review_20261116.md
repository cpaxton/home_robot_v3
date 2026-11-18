# Agent complexity review, November 16

Reviewed feature revision `420fb299` and the captured Molmo development run at
`~/runs/emet/context-image-ids-find-20261115-r5/`. This is a source/context audit,
not a new live evaluation or evidence that simplification improves accuracy.

## Assessment

The useful core is straightforward: assemble task, images and history; ask the
coordinator for a tool call; execute it with measured physical validation; return
the observation and result; repeat until the model finishes or its budget ends.
The implementation surrounding that core is too entangled. Fixing it should
reduce overlapping responsibilities before adding another recovery policy.

Measured scope:

- `agentic/*.py`: 12,228 source lines including comments, compatibility policies
  and trace code. Line count alone does not measure active runtime complexity.
- `init_executor` assigns 110 distinct `self` fields, before later lazy fields.
  `AgenticSession` is an open attribute container forwarded by the executor,
  rather than a declared state contract.
- The actual live system prompt is 9,839 characters and advertises 12 tools.
  `navigate_to_obs` is explicitly an alias for `investigate`.
- Final per-round raw state is 3,319 characters for bowl and 3,042 for microwave.
  It repeats room, memory, retrieval, visual and geometry summaries. These states
  accumulate alongside exact action receipts in the conversation.
- The episode makes 16 coordinator calls, six logged VLM assessments, two target
  extraction events and three grounding attempts. These are event counts, not
  an exhaustive accounting of every model forward or retries inside grounding.

## Current flow and findings

1. **Find completion mixes different facts.** `views.ground_confirmed_view`
   admits object-specific geometry; `AgenticEQAResult.grounded_obs_id` carries it
   independently of `verified`. `ovmm_agentic_find` then requires both to use
   the grounded-object position. In the live bowl run, geometry 7 survives the
   last camera turn, but visual verification becomes false and terminal find
   returns no position. The latest context clarification explains the existing
   rule; it does not repair that coupling. Remembering where an object was seen
   and authorizing a manipulation from the present state need separate contracts.

2. **Multiple representations describe semantic progress.** The executor holds
   `_verified`, `_verified_obs_id`, `_last_vlm_assess`, `_pending_answerable`,
   `_confirmed_answer_evidence`, `_grounded_obs_id` and an `EvidencePolicy`.
   These have distinct historical purposes, but consumers can accidentally treat
   one as authoritative for another. The failure above is a concrete example;
   the mere presence of several fields is not proof that each is redundant.

3. **Compatibility logic obscures the active loop.** `run.py`, `router.py` and
   `answer.py` contain legacy, grounded-v2 and model-final paths together.
   Model-final correctly skips documented action substitutions and preserves an
   explicit submitted answer. The 16 live executions match model selections.
   Thus much of the visible branching is inactive in this run, but it increases
   review burden. Keep historical policies reproducible while making the normal
   model-final dispatch path readable without tracing every legacy condition.

4. **The model sees overlapping APIs and names.** The navigate alias adds a
   choice without adding capability. `verify_siglip` now encompasses VLM visual
   assessment and geometry feedback despite its detector-specific name.
   `look_around` may capture a single heading, while `rotate_base` turns.
   Existing descriptions document this, but clearer canonical tools would reduce
   the explanation needed. Pixel/capture/object IDs must stay distinct: they
   refer to different things. Simplification must not collapse their semantics.

5. **Repeated state is expensive and sometimes irrelevant.** For a localization
   task, `CONFIRMED_MEMORY` still includes instructions about count and location
   MCQs. Current-view presence, retrieval score, retained geometry and completion
   requirements occur in different sections. The eight-image/64k-character
   adapter prevents the reproduced input growth while preserving exact history,
   but it is not a substitute for a concise current state. Its two information
   tools are memory access, not a new planner or an automatic recovery schedule.

The tool path also performs visual assessment after some motion/capture actions.
That is explicit tool behavior, not a second coordinating agent. It is a useful
ablation candidate: measure whether those extra model calls help beyond returning
RGB and the actual geometric/motion receipts. Do not remove them and claim the
result is a precision-only comparison.

## Smallest useful sequence

1. Repair the find handoff contract using the existing grounded-object reference.
   Return provenance-bound last-observed location separately from current
   visibility and downstream motion/manipulation authorization. Test admission
   followed by a camera turn; preserve failures for wrong query, wrong episode,
   missing geometry and explicit invalidation. Physical actions still use their
   current geometry, reachability and collision checks. Do not invent another
   confidence score or another persistent evidence registry.
2. Simplify the model-visible interface: advertise the canonical investigate
   tool, keep compatibility aliases callable for existing integrations, and
   render task-relevant current facts once. Remove count/MCQ instructions from
   find state. Preserve exact receipts, original history and accessible images.
   Check fixed captured contexts before attributing live differences to context.
3. Compare local int4 and Caliban on identical frozen image/text requests with
   explicit model revision, decoding, image preprocessing and output limits.
   Follow with small repeated live cases only after handoff regressions pass.
   Keep architecture and model changes separate. Source-module cleanup can then
   isolate old policies without changing the measured active behavior.

Acceptance for the simplification is behavioral: original tool arguments and
explicit answers survive; current images and historical evidence are identified;
rejected actions return measured reasons; remembered location does not authorize
stale manipulation; small EQA and room-find checks retain their failed rows.
No additional workflow engine is needed to test this.

## Caliban availability and comparison caveats

On November 16, `http://192.168.1.55:8000/health` reports ready CUDA multimodal
`Qwen/Qwen3-VL-8B-Instruct`; `/v1/models` advertises the same checkpoint name.
Hostname `caliban` did not resolve here. No generation, deployment, restart or
remote file modification was performed. SSH login with the available credentials
was rejected, so the running server's dtype and code revision are not verified.
The health response does not expose weight precision.

The existing native launcher defaults to float16 and the historical
[precision note](vlm_quantization.md) records fp16 experiments. However, the
checked-in Jetson server uses `do_sample=False`; the local path inherits model
generation settings unless overridden. Its wire path/image preprocessing may
also differ. Historical 9/30 versus 16/30 figures are motivation for a controlled
check, not sufficient proof that precision alone caused the difference.

`run_shared_room_pairs.py` intentionally fixes local int4 and removes remote
endpoint environment variables. Exporting `EMET_VL_ENDPOINT` alone will not
switch that driver to Caliban. An explicit generic model-endpoint option and
recorded effective server configuration are needed before labeling a new pair
as an int4/fp16 comparison; no Caliban-specific agent branch is needed.

Relevant source: `src/emet/memory/graph_eqa/agentic/` (executor initialization,
run, router, tools, views and answer modules),
`src/emet/eval/ovmm_agentic_find.py`, `src/emet/llms/{conversation,image_context}.py`,
`scripts/run_shared_room_pairs.py`, `docker/jetson_llm_server.py` and
`scripts/run_jetson_vlm_native.sh`.
