# Observed surface grounding diagnostic — October 23, 2026

This is a recorded Stretch RGBD grounding diagnostic, not an agent episode,
tracking acceptance, object-pose estimate or native manipulation result. Caliban
served Qwen3-VL-8B-Instruct; no local GPU inference was used. All four candidate
selection attempts are retained in [the result record](observed_surface_grounding_20261023.json).
Job: `20261023_194152_4bf30e`, source `79ae048e` plus the ignored diagnostic script.

The earlier point-selected regions had only 1 red-dominant pixel out of 147 for
“red cylinder” and 0 blue-dominant pixels out of 366 for “blue cube”. Valid depth
at the guessed pixel did not establish correct object identity.

| Selection | Query | Selected pixels | Target-color pixels | Latency (s) |
| --- | --- | ---: | ---: | ---: |
| Existing measured candidates | red cylinder | 29 | 29 | 14.64 |
| Existing measured candidates | blue cube | 115 | 115 | 16.50 |
| Search box expanded by 50 normalized units | red cylinder | 204 | 204 | 16.70 |
| Search box expanded by 50 normalized units | blue cube | 336 | 333 | 18.73 |

Color counts are an independent fixture-specific diagnostic, not production
identity logic or a calibrated semantic score. These four selections reuse one
view of two targets, so they do not establish generalization or a task success
rate. The expanded diagnostic used the existing external-mask seam after bounded
RGBD candidate generation; a separate run checks the new direct API path.

## Opt-in implementation

`LazyGraphController.ground_vlm_frame` forwards `query_memory.search_padding`
into the existing `ground_vlm_region` / `select_supported_region` path. Use
`region_strategy: depth_candidates`, `mask_backend: rgbd`, and `search_padding: 50`
under `mapping.query_memory` in nested configuration. Padding is in normalized
0–1000 image coordinates, applied on each side and clipped to the image bounds;
it is an integer from 0 through 100, default zero. It only applies to RGBD
component proposals, not SAM or external masks. Defaults remain unchanged.

The coarse VLM box is an untrusted search region. Padding does not modify the
model response or repair invalid boxes. The audit separately records the actual
search region. All candidates still require measured support and final semantic
selection; the eight-candidate limit, whole-mask association filters and semantic
abstentions remain in force. The implementation never selects by fixture color.

## Next gate: fresh-frame association

The measured component selector improves initialization but still finishes after
the two-second evidence freshness limit. Do not extend that limit, assign the
inference completion time to the source image, or reuse a cached observation as
new evidence. Observed execution remains unavailable.

A subsequent implementation must associate the verified reference surface with
new captures using current calibrated RGBD geometry and appearance, reject
ambiguous/occluded/replaced surfaces, and discard its state across motion epochs.
Test delayed inference, repeated/out-of-order frames, missing depth, camera motion,
object motion, same-appearance distractors and target removal. Every emitted
position must retain the actual receive time of its measured frame. Visible
surface position is not a full object pose or conservative payload volume;
those remain independent requirements for native observed execution.

Production-path probe: job `20261023_194604_0ac11e`; output
`.tamp-runs/observed-tracking/production_results.json`, log
`.tamp-runs/observed-tracking/production-job/job.log`. It includes both known targets
and the absent query “green mug”, using `search_padding=50` directly.

The direct API probe completed: red cylinder selected 204 pixels (204 red),
blue cube 336 pixels (333 blue), and absent green mug was rejected with zero
selected pixels. Latencies were 16.84, 18.76 and 4.92 seconds respectively. All
three attempts are retained in the same result record; none dispatched motion.

Validation: the 99-check grounding/candidate/query/epoch regression selection
passed; a subsequently added controller-forwarding check plus the two existing
tracking-configuration checks also passed. Ruff and `git diff --check` are clean.
These checks overlap; they are not independent experiment trials.
