# OVMM grounding closeout

Implementation: `cfab4f97`, review PR #167. The 142-test focused suite and all
commit hooks passed. A subsequent CLI dispatch regression also checks that
`--seed` and query mode actually reach the runner (not merely appear in help).

Pilot submitted September 9 as managed job `20260909_120346_2b3550`, frozen
checkout `/tmp/emet-ovmm-grounding-cfab4f97`. Results/evidence are under
`/tmp/emet-ovmm-grounding-v2-cfab4f97`; supervisor log and recorded command are
under `/home/cpaxton/runs/emet/jobs_runs/ovmm-grounding-v2-pilot`.
Status at this documentation update: running, simulator/agent initialized;
task outcomes pending. No merge acceptance or performance improvement claimed.

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
