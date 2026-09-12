# Segmented grounding in the shared agent

Purpose: improve measured support without introducing an OVMM-specific agent.
Qwen chooses a box, SAM2 proposes a mask, depth validates measured pixels, and
Qwen verifies identity. No simulator labels or detector scores grant acceptance.
This remains an opt-in research path, not a production manipulation guarantee.

## Controlled offline comparison

Reuse the exact 60 RGB frames and Qwen boxes from `best-local-grounding-20260910`.
No new localization calls, box expansion or GT input to segmentation. SAM2.1-small
selects its highest predicted mask-score proposal; that score is NOT semantic
confidence. The final selector still has to accept the measured support.

Source `3727fba6`, job `20260910_085137_6a7944` (completed, serial).

- Mask cache: `/home/cpaxton/runs/emet/sam2-proposals-20260910`.
- Initial Qwen selection: `/home/cpaxton/runs/emet/sam2-grounding-20260910`.
- Matched isolated/context verification: `/home/cpaxton/runs/emet/sam2-context-20260910`.
- SAM2 checkout: `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- Checkpoint SHA256: `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`.

| Mask provider / verifier | Pure selections / 31 visible | Impure accepted |
| --- | --- | --- |
| RGB-D components / context | 9 | 9 |
| SAM2 / matched isolated | 13 | 4 |
| SAM2 / context | 13 | 5 |

Context SAM2 breakdown: development 7 pure / 4 impure, previous held-out 0 / 0,
supplementary close views 6 / 1. Missing proposals in the seven visible held-out
rows remain missing. This does not establish held-out generalization. Context
does not universally outperform isolated selection; keep both options recorded.

Bowl purity is about 99.9% across the original three views. The close-view sponge
improves from a 4.2%-recall pure patch to 98.8% purity / 82.1% visible recall.
Bottle, close sugar cube, and broccoli also improve. But SAM2 can follow an
incorrect/partial prompt onto another surface. Context still accepts zero-overlap
masks in original sponge/paper-towel views and the close paper-towel view. Other
impure acceptances include sugar-cube views (91.5% and 1.3% purity). Cleaner
boundaries are not proof of correct object identity.

Exploratory post-hoc comparison found that agreement with the old measured mask
would reject three disjoint wrong-surface selections while retaining 13 pure
ones, but two impure cases remain. This was NOT a preregistered threshold test
and is NOT implemented as an acceptance rule. Do not tune or claim safety from it.

## Shared integration

`configs/emet/query_segmented_pilot.yaml` extends the existing query-memory pilot:

- `grounding_backend: vlm`
- `region_strategy: depth_candidates`
- `mask_backend: sam2`
- `surface_presentation: context`
- `whole_object_box: true`

The shared lazy/query controller owns one lazily initialized SAM2-small instance.
No robot-specific perception branch is added. The segmented mask may extend
beyond the prompt box, but remains subject to finite/range-valid depth checks.
Context rendering/prompting are shared with offline replay and exact context
panels are included in grounding debug records. Default RGB-D/point behavior and
normal agent presets remain unchanged. Blind identity matching is not wired in.

The existing SAM2 wrapper loaded two model copies and created the first before
device selection. It now shares one model for box and automatic-mask predictors,
uses inference mode, validates boxes and handles empty prompts without inference.
This is tested without model downloads. SAM2 remains an optional dependency for
users who do not enable the segmented path.

This supports the shared architectural boundary: low-confidence search candidates
are separate from grounded observation-local support, and manipulation still
requires fresh target evidence. It does not make a partial visible point cloud a
complete object model, nor does it establish execution success across tasks.

## Integrated simulation check

Initial find-only job `20260910_085837_0fd406` failed before connection with
`KeyError: agent`. Recursive preset inheritance skipped the grandparent defaults.
Fixed in `b5dab32e`: resolve every extends hop, compose defaults after merging,
and detect inheritance cycles. Tests cover the segmented preset retaining robot
client motion/agent parameters. Do not copy missing defaults into each task preset.

Retry `20260910_090227_c33e2c` uses source `b5dab32e`, Stretch default-table sim,
local Qwen, existing opt-in head sweep, and one `find_objects` request. No pick or
place is requested. Grounding evidence goes to
`/home/cpaxton/runs/emet/shared-segmented-find-v2-20260910`.
The process exited cleanly, but localization FAILED: legacy voxel verification
did not localize the cylinder and frontier execution aborted on a waypoint
timeout. No segmented-grounding artifacts were produced: this find route never
called the query verifier. This is not an online test of SAM2 mask quality.

The executor also incorrectly reported `find -> ok`. The fix records failed
finds in `_last_exec_ok`, makes dispatch relay failures rather than generic Done,
and prevents navigation budget exhaustion from returning an intermediate search
point as a localized object. Focused tests cover all three contracts.

At that point, remaining integration work was to connect query-tier retrieval and fresh view grounding
to the shared find/navigation loop, preserve the distinction between a candidate
approach and verified arrival, and diagnose the waypoint timeout without relaxing
safety limits. Then rerun find-only simulation before manipulation. EQA tool
grounding and pre-manipulation grounding alone do not establish that the legacy
find route uses the same verifier.

### Shared find wiring (2026-09-10 follow-up)

Implemented in `06cf9c9b` and `a0906b82`: query-driven find bypasses legacy
`localize_text`/detector acceptance. It first checks captured views through the
shared verifier, prefers measured target geometry when available, and otherwise
approaches a bounded query-tier voxel candidate. Neither an approach nor a
completed frontier is object-find success. Arrival requires new captured evidence
through the same verifier, and the verified gaze is retained. Manipulation still
reacquires independently. Rejected candidates do not create ambiguous active
manipulation references. No default perception settings or safety tolerances changed.

Job `20260910_175826_6a9e85` (source `06cf9c9b`) reached the query-candidate
planner, but base execution failed before arrival verification. Crucially, the
agent reported `find -> failed`, not success. The navigation wrapper labels any
false trajectory result as a waypoint timeout; the underlying terminal command
receipt is needed to establish the actual cause.

View-first retry `20260910_180138_325416` used source `a0906b82` and exposed a
missing callback forwarding in `GraphEQAController.look_around`. Fixed in
`e210adbc`, with tests through the actual lazy-controller inheritance chain and
the Habitat body-scan equivalent. Retry `20260910_180646_32d5e5` uses that source,
simulator subprocess output, and terminal command-receipt logging. Do not count
it as passed until localization and navigation evidence have been checked.
Heavy jobs use one exclusive GPU lock. The combined focused suite passes 153
tests (controller, graph/query memory, manipulation handoff, tools and transport).

The `e210adbc` run reached live grounding. Qwen detected the visible cylinder in
two views, but SAM2 masks were rejected as non-boolean. Its public predictor
casts thresholded masks to float32; offline caching had explicitly cast them
back, concealing the live/cache contract mismatch. `41731f4b` normalizes and
validates binary masks in the shared wrapper, including a float-output test.

The same run's navigation receipt established a separate failure: the controller
stopped 0.0983 m from the target while measured arrival required <=0.07 m. For
unnamed commands, the Stretch controller had retained its looser default tolerance.
`e3fee320` applies the existing exploration policy's 0.07 m / 0.15 rad limits to
the motor controller as well; acceptance is not relaxed. Explicit precision
settings remain supported. Retry `20260910_181120_10fd62` freezes these fixes.

That retry produced admitted, visually correct cylinder support (319 measured
pixels in the first accepted view) and completed base motion, but could not sample
an object approach. The 2D obstacle ray treated the supporting table as a visual
occluder. Cancelled the repeated-search diagnostic after identifying this issue;
its frozen evidence remains available. `4737f79f` lets query-driven approaches
use fresh RGB-D arrival verification instead of planar visibility. Reachability,
footprint, standoff and path collision checks remain intact; legacy and frontier
visibility behavior stays unchanged. Retry `20260910_181633_5f6cd7` freezes this
change. No end-to-end find success is claimed yet.

The `4737f79f` run completed navigation and reported success, but manual review
INVALIDATED that success: final record `grounding-15224831f69c4c99bfd774df08b89751`
selected brown table pixels beside the red cylinder. Qwen's context response
incorrectly attributed the nearby red object to that support. This is precisely
why process/tool success is insufficient for the pilot gate.

### Support-only identity ablation

Source `87fc920c`, job `20260910_182151_546179`, replays that live failure and
both 60-view proposal caches with the original scene omitted from final selection.
The model receives only isolated measured-pixel panels, without prior reasoning.
Qwen correctly rejects the live table patch as brown and not a red cylinder.

| Provider / support-only selector | Pure selections / 31 visible | Impure accepted |
| --- | --- | --- |
| SAM2 | 12 | 1 |
| YOLOE-L | 8 | 10 |

SAM2's remaining impure case is the close sugar-cube view (91.5% purity, 100%
visible recall). It loses one pure selection relative to context but rejects four
impure ones. This does not transfer uniformly across providers: YOLOE does not
improve. Neither recovers the held-out slice. This is development following a
manually found failure, not a new independent held-out result.

`configs/emet/query_segmented_support_pilot.yaml` separately enables the shared
support-only selector; context and production presets remain unchanged. Source
`3d609747` is frozen for red-cylinder job `20260910_182453_bb940d` and blue-block
job `20260910_182511_c74c26`, serial with identical settings. Both require manual
support inspection before claiming find acceptance. No pick/place is requested.

These Qwen-box live runs were stopped after repeated unsuccessful verification.
The red run also admitted another brown patch during initial search, explicitly
reinterpreting the requested color in its reasoning. Support-only is an improved
offline selector, not a reliable semantic guarantee. The blue run did not finish
a verified find either. Do not report these cancelled diagnostics as successes.

### Detector boxes, SAM2 refinement, Qwen decision

`configs/emet/query_detector_segmented_pilot.yaml` changes only the proposal
provider: YOLOE-L supplies query-conditioned boxes, SAM2 supplies masks, then the
same Qwen support-only selector and measured-depth admission decide acceptance.
YOLOE labels/scores never grant an instance. This addresses small-object VLM boxes
landing beside the object without adding a robot-specific search agent.

Source `bf268a7c`, red run `20260910_183408_ddbe85`: **manually verified find
success**, tool runtime 21.6 s. Initial source observation 3 and fresh arrival
observation 5 contain the cylinder; final support is 479 measured pixels on the
red cylinder, not the table. XY is [0.0599, -0.5387] initially and
[0.0620, -0.5343] at arrival (world metres). The planned base XY was already
near the start; this proves nearby localization and turning, not long-range
navigation or grasping. Evidence:
`/home/cpaxton/runs/emet/shared-hybrid-find-red-20260910/grounding/grounding-f06652c1b27140488557f275a800dd79.json`.

The same frozen provider/settings are used for blue run `20260910_183539_bed3e4`
and paired-cache job `20260910_183607_080dcc` (YOLOE proposals with versus without
SAM2 refinement). Both are serial. The combined focused suite passes 195 tests;
this is not a new EQA/OVMM/TAMP benchmark result. Simulator shutdown emits
multiprocessing teardown errors after task completion; preserve those logs rather
than mistaking them for successful or failed perception.

Blue run `20260910_183539_bed3e4`: **manually verified find success**, tool runtime
25.0 s. Source observation 4 and arrival observation 6 show the blue block; final
support has 802 measured pixels, XY [-0.0462, -0.5094] m. The final isolated panel
shows the block, not supporting furniture. These are two nearby-object smokes in
one simple scene, not independent evidence of broad-environment or OVMM success.
Final record:
`/home/cpaxton/runs/emet/shared-hybrid-find-blue-20260910/grounding/grounding-0a073cdb65884c768f937bbb851632c0.json`.

Matched-proposal cache job `20260910_183607_080dcc` completed: SAM2-refined YOLOE
proposals give **15 pure / 3 impure** with support-only Qwen, versus raw YOLOE's
**8 / 10** under the same selector. Development: 9 pure / 3 impure; supplementary:
6 / 0; held-out: 0 / 0. All 29 zero-visible views are rejected. Remaining errors
include zero-target-overlap selections for sponge (index 9) and paper towel
(14), plus sugar-cube contamination (16, 93.8% purity). These are real failures,
not just threshold rounding. Scores and exact inputs are in
`/home/cpaxton/runs/emet/hybrid-paired60-support-20260910/support_only/scores.json`.
The improvement supports a prototype, not broad reliable perception or safe grasping.

To reproduce a bounded, find-only run with the optional perception dependencies
installed (use the repository's configured environment):

```bash
PYTHONPATH=src EMET_ALLOW_SDPA_ATTN=1 EMET_FORCE_HEAD_SWEEP=1 MUJOCO_GL=egl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m emet.app.run_agent \
  --config configs/emet/query_detector_segmented_pilot.yaml \
  --memory-backend lazy_graph --robot stretch --start-sim \
  --sim-config configs/sim/default_table_stretch.yaml --headless --no-discord \
  --llm qwen3-vl-eqa --eqa --debug-tools \
  -c "Use find_objects once to locate the red cylinder. Do not pick or place anything."
```

Set `EMET_EQA_EPISODE_DIR` to a fresh artifact directory to retain grounding
records. For CPU affinity and serialization, wrap the command with
`emet jobs run --cpu-safe --gpu-exclusive`; do not launch parallel heavy jobs.
This remains a pilot preset, not a changed
production default. Next acceptance should hold this harness/model fixed across
cluttered find/OVMM, EQA and learned TAMP; task-specific success criteria differ,
but the query, geometry, freshness and navigation contracts should not.

### Paired mask-provider comparison

Job `20260910_175926_11c501` runs YOLOE-L on the identical 60 cached inputs,
followed by the same local Qwen isolated/context verification used for SAM2.
Compare against `sam2-context-20260910`, scoring against the same isolated GT
files. This compares complete proposal providers: SAM2 uses the saved Qwen box,
whereas YOLOE uses the query vocabulary. It is not a fixed-box segmentation-only
ablation. Record missing proposals and impure acceptance alongside pure support;
neither detector confidence nor a clean mask grants semantic acceptance.

Completed paired results (31 visible, 29 zero-visible queries):

| Provider / verifier | Pure selections | Impure accepted |
| --- | --- | --- |
| YOLOE-L / isolated | 8 | 10 |
| YOLOE-L / context | 8 | 9 |
| SAM2 / isolated | 13 | 4 |
| SAM2 / context | 13 | 5 |

Both providers reject all 29 zero-visible views in the context comparison, and
neither recovers the seven visible held-out queries. SAM2 improves broccoli
boundaries and recovers supplementary bowl/sugar views; YOLOE recovers a sponge
view (index 44) where SAM2 has no accepted support. The next sponge view (45)
illustrates the tradeoff: YOLOE purity/recall 94.2%/99.3%, SAM2 98.8%/82.1%.
The 95%-purity gate is an evaluation convention, not physical grasp validation.
Exact masks, Qwen requests, panels and per-row scores are retained under
`/home/cpaxton/runs/emet/yoloe-paired60-{proposals,grounding,context}-20260910`.

Focused checks so far: 51 perception/grounding tests, 69 shared query/manipulation/
tool-outcome tests, and 24 config/query-config/replay tests. Suites overlap; do not
sum these as a unique-test total. No new EQA, OVMM benchmark or learned TAMP success
is claimed from these unit checks or the offline results.
