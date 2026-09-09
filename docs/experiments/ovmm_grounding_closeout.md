# OVMM grounding closeout

## September 9 detector-free follow-up (not merge acceptance)

Final focused regression pack: **225 passed**, with one existing CLI test that
can load a real offline model excluded from the unit run. Commit hooks and
`git diff --check` also pass. Unit success is not task acceptance.

The new query-grounding backend uses Qwen to select a normalized image box and
an interior surface point. Connected, finite depth support supplies partial
visible-surface geometry; it is not a semantic segmentation, whole-object
extent, calibrated confidence, or a grasp plan. YOLOE remains an explicit
`query_memory.grounding_backend: yoloe` option, not a mandatory gate. Query mode
remains disabled by default. A geometry rejection permits at most one annotated
correction; semantic abstentions are not retried on the same image.
An additional same-model marked-point check was evaluated and withdrawn after
failing the cached audit below. It is not part of the final production path.
VLM surface grounding remains experimental: valid depth does not establish that
the point belongs to the requested object. Do not use this prototype to authorize
real-robot manipulation.

Arrival capture checks central depth for severe obstruction and permits at most
two head-only recaptures. It never substitutes a historical view or moves the
base as a recovery shortcut. This detects close/invalid depth, not semantic
coverage: an unobstructed wall is still an unhelpful view.

The shared command now exposes opt-in `--visual-servo`. Its current wrist-camera
grasp adapter is Stretch-specific. Query-driven chat cannot enumerate oracle
scene tasks, construct/execute oracle TAMP plans, or bypass fresh grounding via
`pick_place`. Controller completion is explicitly not independently verified
physical success. Other modes retain the existing oracle controls for isolated
mechanics testing, labeled accordingly.

Integrated startup also exposed a profile-precedence bug: the generic embodied
graph preset could overwrite `lazy_graph` with `use_instance_graph=true`.
`f899c120` applies the lazy backend invariant after those preset values, and
`3e973f36` adds `configs/emet/query_surface_pilot.yaml` with explicit verification
prerequisites. The first integrated attempt (`20260909_142931_1489b9`) failed
startup because verification was not enabled; it is not a task result.

The shared sim command is:

```bash
EMET_ALLOW_SDPA_ATTN=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYTHONPATH=src python -m emet.app.run_agent \
  --config configs/emet/query_surface_pilot.yaml --memory-backend lazy_graph \
  --robot stretch --start-sim --sim-config configs/sim/default_table_stretch.yaml \
  --headless --no-discord --llm qwen3-vl-eqa --eqa --visual-servo \
  --debug-tools --debug-llm \
  -c 'Use pick_place to put the red cylinder on the blue cube; report failures.'
```

This is an experimental diagnostic command, not a validated real-robot recipe.
Launch through the managed CPU-safe/GPU-exclusive job wrapper and set a bounded
timeout. Keep robot/simulator choices separate from the shared memory preset.

### Completed bounded runs

| Check | Frozen source | Outcome | Interpretation |
| --- | --- | --- | --- |
| Static table visibility | `385e6657` | red cylinder 2,099 pixels; blue cube 2,623 | Render/visibility control, not learned grounding |
| Live hold and two turns | `385e6657` | hold drift 0; turn errors about 0.030 rad | Partial route acceptance |
| Route translation | `385e6657` | unsafe posture; upright dot 0.97898; XY error 0.0928 m | Failed; safety threshold unchanged |
| OVMM nearest-v2 00006 | `81dcb92f` | FindObj 0/1, FindRec 0/1; 17 graph nodes; 122.9 s | Task acceptance failed |
| OVMM nearest-v2 00025 | `81dcb92f` | FindObj 0/1, FindRec 0/1; 21 graph nodes; 134.2 s | Task acceptance failed |
| EQA q15, q16, q25 | `81dcb92f` | 3/3 correct | Small smoke slice, not a paired no-regression claim |
| TAMP plan/execute control | `81dcb92f` | placement error 0.020 m | Oracle scene poses, kinematic attachment and nav teleport; not learned TAMP |
| Eight cached RGB-D queries | `1e632bee` | sofa/blue cube accepted; lamp false negative; red cylinder failed; four negative/relation cases abstained | Region correction did not recover the red cylinder |

All learned runs above use local `Qwen/Qwen3-VL-8B-Instruct`, int4, CUDA/SDPA.
OVMM uses seed 0, 12 rounds and 8 navigation steps. EQA uses 20 planning steps
and 10 movement steps, without HM3D semantic/enriched labels; its CLI does not
freeze a seed, so these runs are not a matched-seed causal comparison. EQA final
submission and initial raw EQA output can differ: inspect the agentic trace,
not just the correctness field, before claiming evidence quality.

The stationary `rby1` registry entry in this checkout resolves to the Galaxea R1
MJCF. This is not evidence of a genuine Rainbow RBY1 hardware model. The route
failure establishes unsafe measured posture, not its dynamics/collision cause.

Manual review of the cached correction overlay shows the red-cylinder point
above/left of the cylinder on background. Invalid depth correctly blocks it;
Qwen subsequently abstains rather than correcting it. The blue-cube point lies
on the cube and yields a surface centroid within 0.15 m of its evaluator center.
The close-up lamp remains a semantic false negative. A graph-size reduction
therefore does not demonstrate improved perception or task success.

Manual before/after review of receptacle-round-5 views `1099511627804` and
`1099511627806` shows a nearby textured window/exterior surface in both images.
The obstruction fraction falls from 1.00 to 0.71, but the head turns do not
reveal the requested table. Passing the depth check is not semantic progress.

Artifacts and managed jobs (one CPU-safe/GPU-exclusive experiment at a time):

- `20260909_141034_d5dea5`: `/tmp/emet-vlm-stationary`, `/tmp/emet-vlm-known-route`.
- `20260909_141308_fea283`: `/tmp/emet-vlm-habitat-acceptance`; per-phase traces,
  RGB-D caches, model settings and results. EQA figure/video bundles are under
  `~/.cache/habitat_eqa/episodes/vlm-led-eqa-{15,16,25}`.
- `20260909_141830_316009`: `~/runs/emet/jobs_runs/vlm-tamp-kinematic-control/job.log`.
- `20260909_142619_bacd38`: `/tmp/emet-vlm-feedback-sdpa-audit`; exact prompts,
  both attempts, original/correction overlays, support masks and calibrated XYZ.
- `20260909_143833_bb0d31`: `/tmp/emet-vlm-route-reacquisition-audit`, source
  `1e632bee`. Four queries after the two passed turns all had finite depth, but
  manual overlays show three points on the table behind the objects. Only the
  final blue-cube point is supported on the requested object. Do not report
  this as 4/4 successful reacquisition. This motivated marked-point verification
  in `273df5ba`, with a separate frozen audit rather than rescoring this run.
- `20260909_144248_f03625`: frozen `273df5ba`, outputs
  `/tmp/emet-vlm-point-verified-static` and `/tmp/emet-vlm-point-verified-route`.
  The added marked-point check still approved two cylinder proposals on the
  tabletop and rejected valid cube proposals with contradictory explanations
  (`verified:false` but prose asserting validity). It was withdrawn: no extra
  production inference gate or tuning knob is retained for this failed idea.
- Excluded launches: `20260909_141013_278915` was cancelled after detecting the
  wrong legacy model factory; `20260909_142412_a836bd` failed before inference
  because SDPA opt-in was missing. Neither contributes results.

Replay a manifest of `{arrays: NPZ, rgb: optional PNG, query, description: optional}`:

```bash
EMET_ALLOW_SDPA_ATTN=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python scripts/audit_vlm_regions.py \
  --manifest /absolute/frozen-inputs.yaml --output-dir /absolute/new-output
```

NPZ must contain depth and either RGB or an explicit PNG reference. World XYZ
is emitted only with `camera_K` and `camera_pose`; missing calibration is not
invented. Retain the manifest, frozen commit and managed-job command together.
These paths are local diagnostic artifacts, not an archival public dataset.

Remaining acceptance gates: reliable red-cylinder localization on cached and
live views, safe complete route, non-oracle shared-agent manipulation, and OVMM
localization on both scenes. Keep PR #167 draft while these remain red. Do not
promote this pilot to a performance table or silently change benchmark budgets.

### Integrated shared-agent smoke

`20260909_143340_91bef3`, frozen `3e973f36`, confirmed lazy initialization with
`instance_graph=False` and no YOLOE load. The production Qwen router selected
`pick_place(red cylinder, blue cube)`. Fresh observation then stalled in the
32-token image-caption call: first decode after 76 seconds, generation timeout
at 180 seconds, followed by an automatic retry. We cancelled the run and cleaned
up its exact agent/simulator process groups. No grasp or placement was validated.
Logs are under `~/runs/emet/jobs_runs/vlm-shared-preset-manip`.

The timeout worker can continue running CUDA work after the caller raises. The
client now rejects subsequent generations after a timeout, and voxel captioning
propagates timeouts rather than retrying or silently completing perception.
This is a safety/lifecycle fix, not a demonstrated fix for the original stall.
Investigate live caption/model-sharing latency before another manipulation run;
do not extend timeouts or add retries to turn this red gate into apparent success.

## Earlier detector-gated pilot

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

Detector-free surface selection now has a separate
[frozen perception pilot](surface_candidate_pilot.md). Clean/noisy red-blue
localization passes; lamp and textured-sofa failures remain. These diagnostics
do not replace the task-level acceptance below or change the prior OVMM 0/4.

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
