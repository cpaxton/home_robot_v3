# Live caption latency and candidate rejection follow-up

Source and runs remain frozen; heavy jobs are GPU-exclusive and CPU-safe.
No real robot is involved and no model, motion tolerance or timeout is relaxed.

## Live caption diagnosis

The prior failed integrated view was recovered from voxel-memory debug data,
not the query-grounding cache. It shows mostly dark background. The original
RGB-D is now retained in
`/home/cpaxton/runs/emet/jobs_runs/surface-shared-agent-handoff/memory_debug`.

`da1bd216` adds worker-stack output to existing VLM heartbeats only when model
debugging is enabled. Reproduction `20260909_181427_c7024c` again times out at
180 seconds; worker stacks are in Qwen's language-model forward pass. The
runtime has a background thread consuming about 89% of one CPU core.

Code inspection identifies a disabled-visualizer busy loop: `NullVisualizer`
is truthy, so Stretch starts its Rerun thread and continuously invokes its no-op
`step` after observations arrive. The generic client already requires
`visualizer_is_enabled`. `7baece68` applies that contract to Stretch startup and
the loop guard. Tests cover None, NullVisualizer, and an enabled visualizer.

Exact offline caption replay `20260909_181721_d121a0`, source `b09b4114`, uses
the same RGB array, prompt, 32-token limit, Qwen3-VL-8B int4 and CUDA/SDPA.
It completes in **0.53 seconds**, saying no objects are available. This excludes
the image alone as the reason for the live timeout. The standalone replay
script is `scripts/audit_vlm_caption.py`; it preserves its input and timing.

Integrated rerun `20260909_182006_4a45b8` uses source `7baece68` and the same
Stretch `pick_place` command, model and 180-second caption timeout. The process
has a separate 360-second outer limit. Candidate rendering also changed on
this source, but that code runs after captioning and cannot explain a caption
latency change. Task success remains distinct from fixing the busy loop.

The integrated rerun completes normally: live captions take **0.19–0.66 s**,
the pick/place tool returns in 28.7 s, and the whole turn takes 30.1 s. It reaches
exploration and query manipulation, then rejects an absent/ambiguous target and
skips placement. The late view shows only the table edge, without the targets.
Thus the latency regression is fixed in this run; pick/place is still not a
success. A separate observation-coverage ablation enables the existing
`EMET_FORCE_HEAD_SWEEP=1` option (`20260909_182507_85fb3a`), leaving defaults
unchanged and retaining explicit query-grounding artifacts.

## Candidate-only presentation

`67fcbc9e` retains only the measured support pixels in candidate panels, placing
black outside the mask. The separate original image still supplies context.
This removes the dimmed target that previously appeared outside a wrong table
candidate and was falsely described as selected object evidence.

The first replay (`20260909_181556_1e6420`) covers the same 15 stress queries.
It accepts nine target surfaces, abstains on the farther cylinder, and abstains
on all five absent queries. However, the freshly generated search box changed:
the farther-cylinder case has no supported candidates. This is **not** evidence
that changing presentation alone caused rejection.

Therefore `b02871c6` adds `--replay-boxes PRIOR_RESULTS_JSON` to the audit CLI,
restricted to the candidate strategy and exact matching input manifests.
Replayed localization is explicitly labeled in results, not presented as new
VLM inference. Run `20260909_182134_d9e871` holds the original boxes fixed,
including the 102-pixel table-only candidate. It changes only candidate
presentation/prompt relative to that original selector, not proposal geometry.
This is one known wrong-only proposal set, not calibrated rejection reliability.

The fixed-box replay completes with identical boxes and proposal masks in all
15 cases. Nine target queries retain 100% selected-mask purity and pass the
15 cm diagnostic gate; the farther cylinder now abstains, and all five absent
queries abstain. In the wrong-only case Qwen still calls the patch red in its
explanation, but sets `target_unambiguous=false`, so the strict selector rejects
it. This removes the known incorrect acceptance **on this controlled replay**,
not a claim that VLM explanations or confidence are reliable. The missing target
proposal remains unrecovered; more rejection cases are needed before promotion.

The focused shared-agent regression pack passes 247 tests, with one existing
model-loading CLI test deselected. Production defaults remain unchanged.
