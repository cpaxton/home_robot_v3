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

Remaining integration work: connect query-tier retrieval and fresh view grounding
to the shared find/navigation loop, preserve the distinction between a candidate
approach and verified arrival, and diagnose the waypoint timeout without relaxing
safety limits. Then rerun find-only simulation before manipulation. EQA tool
grounding and pre-manipulation grounding alone do not establish that the legacy
find route uses the same verifier.

Focused checks so far: 51 perception/grounding tests, 69 shared query/manipulation/
tool-outcome tests, and 24 config/query-config/replay tests. Suites overlap; do not
sum these as a unique-test total. No new EQA, OVMM benchmark or learned TAMP success
is claimed from these unit checks or the offline results.
