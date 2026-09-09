# OVMM grounding closeout

Implementation: `cfab4f97`, review PR #167. The 142-test focused suite and all
commit hooks passed. A subsequent CLI dispatch regression also checks that
`--seed` and query mode actually reach the runner (not merely appear in help).

Pilot submitted September 9 as managed job `20260909_120346_2b3550`, frozen
checkout `/tmp/emet-ovmm-grounding-cfab4f97`. Results/evidence are under
`/tmp/emet-ovmm-grounding-v2-cfab4f97`; supervisor log and recorded command are
under `/home/cpaxton/runs/emet/jobs_runs/ovmm-grounding-v2-pilot`.
Completed: query-driven 0/4, lazy-arrival 0/4, shared-agent DynaMem 1/4
(table in scene 00025). No merge acceptance or performance improvement claimed.

## Manual visual review and VLM audit

Follow-up job `20260909_132943_604841` (frozen `9420d190`) completed **0/2**
in 179 seconds; artifacts `/tmp/emet-ovmm-retry-qwen-9420d190`. The object phase
now performs seven exploration calls rather than eight rejected-handle no-ops.
However, router states on rounds 3 and 6 still advertise the rejected handle
`-3000000` under Investigate. The legacy renderer read raw `_hypotheses` instead
of the existing rejection-filtered `_investigate_hypotheses`. The renderer now
uses that shared eligible set; a regression asserts the rejected ID is absent
from the actual generated state message. The execution guard remains defense
against stale references, not a replacement for truthful router state.

Manual review of follow-up grounding `ff64a60d55f4473483df37cb39e7c9e4.png`
shows a close, oblique view of a lamp beside a red sofa. Its NPZ masks contain
only -1 (307,200 background pixels); this is a detector mask miss before depth
admission. Qwen correctly reports a visible lamp but cannot verify the bed
relationship. Frame `rgb_1099511627789.png` faces a wall/artwork/wooden surface;
`rgb_1099511627803.png` is mostly occluded by nearby dark geometry. Their negative
Qwen assessments agree with manual review. The latter follows a frontier move
reported reached at approximately (-8.48, 1.21); subsequent moves fail. This
establishes poor observation quality, not the exact collision/camera root cause.
Do not attribute these frames to VLM hallucination or claim the robot fell over.

All six logs identify local `Qwen/Qwen3-VL-8B-Instruct`, int4, CUDA/SDPA.
This establishes model identity, not perception quality. Do not describe the
remaining TAMP gate as training a motion planner: it is the shared task agent
using learned perception with conventional planning/execution, without oracle
object poses or teleport manipulation.

Manual review of query_00025 grounding RGB files ending `44ddf23cddaf497692404c2006611738`
and `616603915f564b3ca2f4c575fa020571` shows a sofa in a kitchen/living area, not
a clearly identifiable bed. The view assessor's bed claims are not supported
by this review. The region verifier rejected the lamp/bed relation. A high lamp
detection score alone does not establish the relation or prove the verifier bad.

The scene 00006 router selected the same rejected query handle on rounds 3–10.
Dispatch now redirects such selections to the existing exploration tool, under
its normal budget and safety gates. The trace retains selected and executed
actions and the rewrite reason. Rejection is not cleared; new-source candidates
remain eligible. This is not permission to revisit a rejected handle indefinitely.

For subsequent diagnostic runs, grounding JSON retains the exact verifier user
and system prompts, full raw response, ordered input-image references, detector
boxes/scores, and configured VLM identity/quantization/image-size settings. The
original RGB and numbered copy are saved alongside depth/masks. The numbered
copy is produced by the same helper as the actual verifier input. These are
client input images, not a dump of Qwen's internally resized image tokens.
View-assessment traces retain their full response, prompt, system prompt and
image count alongside the existing saved RGB reference. An image-unavailable
fallback is recorded as zero images, not silently represented as visual evidence.
Existing pilot caches are preserved; missing historical prompts are not invented.

Review order: image visibility → detector regions/depth → raw Qwen judgment →
parsed decision → admission → selected/executed action. Keep manual annotations
separate from agent evidence; they must not leak into benchmark runs. Model
quality/quantization remains a hypothesis to test on identical cached inputs,
not a reason to weaken admission thresholds or change the live pilot model.

Scope: land the shared grounding correction, then a bounded pilot. No new robot
models, threshold sweep, or paper performance claim is part of this change.

## Why the previous 0/2 is actionable

The September 9 post-merge episode (`20260909_083206_53a933`) detected a lamp
with score 0.223 and 21,689 mask points, but its verifier rejected “on the bed.”
The evaluator actually selected the lamp nearest any bed in the horizontal
plane. This was an instruction/scoring mismatch, not an admission-threshold
failure. The table phase verified a camera view without producing object XYZ;
its image also contains ambiguous cabinet/table furniture. Neither a positive
view judgment nor a camera pose should be scored as localized object geometry.

## Changes and preserved boundaries

- `habitat_find_phase_nearest_v2.yaml` explicitly requests the scorer's nearest
  relation. New episode IDs prevent pooling with legacy “on” runs. The old
  registry remains unchanged for reproduction. Category existence and any GT
  override are validated on the evaluator side before agent creation.
- A positively assessed current captured view can enter the same query-mask,
  semantic-verification, depth-admission and fusion path as a retrieval arrival.
  Historical frames cannot silently ground the current scene. A view is tried
  at most once per query; failed detection creates no search candidate or node.
- Results carry a separate grounded-object observation reference. This also
  handles receptacles, which the existing object-phrase voxel shortcut excludes.
  EQA image answerability remains distinct from localization and action readiness.
- Manipulation can reacquire a currently visible target without first retrieving
  it from memory. Existing candidates still require fresh reacquisition; ambiguous
  references, failed captures and oracle fallbacks remain rejected.
- The single-episode Habitat CLI exposes the existing run seed. Seeds reduce
  uncontrolled sampling differences; they do not promise deterministic GPU runs.

Nearest-to is not support/contact. A local image may be insufficient to prove a
global nearest relation; abstention remains legitimate. This proxy is not full
OVMM pick/place acceptance, and these changes do not solve semantic furniture
confusion by lowering thresholds.

## Bounded acceptance

Run scenes 00006 and 00025, seed 0, 12 tool rounds / 8 navigation steps per
question, Qwen3-VL-8B-Instruct int4, CUDA/SDPA, unchanged detector/admission
settings. Compare lazy-arrival, query-driven lazy, and DynaMem under the **same
shared agentic loop**. The DynaMem row is not the native DynaMem policy; native
policy results must remain separately labeled. One GPU-exclusive CPU-safe job
at a time; no sweeps. Freeze source before starting any episode.

The per-row command is:

```bash
emet-habitat run-ovmm-find-episode \
  --episodes configs/ovmm/habitat_find_phase_nearest_v2.yaml \
  --episode-id hm3d_lamp_bed_00006_nearest_v2 \
  --backend lazy_graph --query-driven-memory --seed 0 \
  --agentic-find --agentic-max-rounds 12 --agentic-max-nav-steps 8 \
  --output /absolute/new-run/query_00006.json
```

Set `EMET_EQA_EPISODE_DIR` to a distinct evidence directory for each run.
For lazy-arrival remove `--query-driven-memory`; for shared-agent DynaMem also
change `--backend` to `dynamem`. Repeat with scene 00025. Preserve commands,
source commit, result JSON, trace, RGB-D grounding cache and timing together.

Report FindObj and FindRec separately, including detection/semantic abstentions,
grounded-object provenance, graph size, and runtime. Runtime completion is not
task success. Do not compare v2 percentages directly with legacy 0/2 as a causal
improvement: language and grounding both changed. If the pilot is still red,
inspect the first failed transition before changing another policy.

TAMP uses the same fresh `prepare_query_target` handoff. Its unit acceptance
checks grounding, capability handoff, alias invalidation and observation after
execution. Learned end-to-end TAMP success still requires its own sim episode;
teleport/oracle execution controls are not substitutes. Paper updates wait for
these outcomes, not just passing implementation tests.
