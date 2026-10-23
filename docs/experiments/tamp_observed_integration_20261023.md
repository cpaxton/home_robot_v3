# Observed integration: acquisition and motion epochs

Branch `feat/tamp-observed-integration` starts at merged main `ef4c1bfa`.
Implementation source: `7c07fe6a`. This first slice guards evidence and plan
lifetime; it does not enable observed manipulation.

## Implemented path

Head-camera timing now passes through controller perception, voxel ingestion and
scene-graph processing to the RGBD task provider. Repeated publication of a capture
retains its first local receive time. Older capture stamps, malformed timing and
changed clock domains cannot produce fresh task evidence. Capture stamps are
identities within an advertised stream; they are not compared against client
monotonic time or treated as synchronized cross-host clocks.

CHAT navigation/view-motion clears observed proposals and stored multi-step,
primitive and native plan handles before and after execution, including failure.
Inference started in the old epoch cannot publish afterward. A capture seen during
motion cannot be replayed after motion to renew proposals. Legacy untimed input
cannot satisfy the post-motion capture-replacement requirement. Independent
controllers and capture/pose synchronization still need provider qualification.

Validation job `20261023_164229_7356c5`: **796 passed, zero failures/errors/skips**,
162.962 seconds. The entire TAMP test directory and integrated regression set
include new real-dispatch and real RGBD/scene-graph tests. This is not the entire
repository suite. Ruff and whitespace checks pass.

## Native controls

Two CPU-only live simulators use the actual CHAT CLI, Caliban inference and the
same privileged stationary fixtures/regions as the earlier integration controls.
They are regressions, not observed acceptance or new benchmark denominators.

- Two-goal: job `20261023_164229_512b98`.
- Buffered swap: job `20261023_164230_303bb5`.
- Artifacts: `.tamp-runs/observed-integration/{two-goal,wide-destination-swap}/`.

Both controls passed independent containment, support and settling checks:

| Case | Completed transfers | Audit commands | Forbidden commands |
| --- | --- | --- | --- |
| Two-goal | 2/2 | 449 | 0 |
| Buffered swap | 3/3 | 750 | 0 |

Neither trace contains a stop-boundary violation. These are individual control
repeats, not observed success rates. [Machine-readable evidence](tamp_observed_integration_20261023.json)
retains final predicates, source/job identities and protocol/summary hashes.

## Camera and visual-region diagnostic

A worktree-local Ubuntu OSMesa library enables software rendering without the
occupied local GPU. Jobs `20261023_164338_8a0f41` and `20261023_164442_018b58`
render actual simulator RGBD. The initial head view misses the table. Four offline
head-pose views identify a table-facing view; these are FK configurations, not
executed head motions or navigation acceptance. The upright frame uses the
existing head-camera rotation convention.

Job `20261023_164628_cbdbcd` calls the existing visual region selector through
`OpenaiVLLMClient` on Caliban (`Qwen/Qwen3-VL-8B-Instruct`). Only the rendered image
and requested names are supplied; no object pose or body IDs are provided.
All three responses select a region with connected depth support, but that is
**not successful object grounding**:

| Query | Inference seconds | Depth-supported pixels | Fixture color pixels inside selected mask |
| --- | --- | --- | --- |
| Red cylinder | 9.451 | 147 | 1 red-dominant pixel |
| Blue cube | 9.241 | 366 | 0 blue-dominant pixels; 3 red-dominant pixels |
| Table | 9.067 | 9,415 | Not an object-mask certification |

The image contains 311 red-dominant and 553 blue-dominant pixels. The simple color
check is a diagnostic for this colored fixture, not a general semantic evaluator:
channel > 30 and > 1.5 times each other channel. It exposes background selection
that connected-depth validation alone accepts. The VLM also describes objects
as stacked; the scene is not certified by that prose. Model checkpoint/sampling
are not independently pinned.

This is an offline visual diagnostic, not live observed task success. Nine-second
model calls also exceed the current two-second live evidence window. The next
provider must track/reacquire valid object evidence in fresh frames after semantic
selection and validate geometry. Do not renew timestamps, enlarge the freshness
window to hide latency, infer a full object pose from its visible centroid, or
replace uncertain geometry with simulator truth.

## Next gate

Connect qualified tracked object geometry, conservative collision checks and
visual effect verification to the task session. Then exercise actual navigation
and post-motion reacquisition in one CHAT episode. The existing provider still
returns `discovery_only`; task planning/execution remain unavailable in observed
mode. [Implementation sequence](../plans/2026-10-23_observed_tamp_integration.md).
