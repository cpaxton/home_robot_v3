# Coordinator-visible current-view grounding

`ground_object(query)` now exposes the existing LazyGraph RGB-D admission path
to CHAT. It captures an observation, requests object-specific grounding and
returns admitted world geometry or an explicit abstention to the next model
round. It does not move, invoke FIND, use a retrieved viewpoint as an object,
or authorize manipulation. EQA's existing confirmed-view admission uses the
same underlying `ground_query_view` method.

Use a complete identifying object phrase:

```json
{"name":"ground_object","arguments":{"query":"nearest black stool in the foreground"}}
```

The coordinator receives object/view references, coordinate frame and units,
geometry source and point count. Failed capture or admission clears the current
grounding reference; it never returns the previous object's XYZ as a fallback.
Acquisition timing is preserved when available, with freshness explicitly
unverified by this tool. Geometry belongs to the recorded captured view, which
can differ from the next live camera attachment. Existing navigation validation
and manipulation reacquisition checks remain independent.

## Where the geometry comes from

The configured VLM selects a visually supported region, while RGB-D geometry
providers and masks determine which measured points are admitted. The pilot
uses Caliban Qwen3-VL-8B, `depth_candidates`, and `yoloe_sam2`; model deployment
precision/revision remain unverified. YOLOE can propose query-conditioned masks;
SAM2 and depth support refine proposals, and the final VLM selection determines
semantic acceptance. The configured bounded SAM2 proposal recovery remains
available when initial candidates overflow or lack support. Other providers
remain configurable; this tool does not switch models silently.

XYZ is the median of the admitted finite masked world points, obtained from
the depth image, camera intrinsics and measured camera pose. It is not a VLM
coordinate prediction, whole-frame depth median, or voxel retrieval anchor.
The recorded example uses renderer sensor depth; estimated depth can be selected
by other configurations and is a separate provenance/quality question. No
simulator object poses or ground-truth segmentation masks enter these checks.
VLM acceptance is binary semantic admission, not calibrated confidence.

The opt-in `configs/emet/shared_find_acceptance.yaml` now explicitly enables
verified instance admission. Generic defaults remain unchanged. LazyGraph still
records viewpoints on ordinary updates; explicit successful grounding creates
the object reference. See [tool documentation](../agent_tools.md#current-view-object-grounding).

## Bounded recorded-frame qualification

Both checks use the same archived frame:
`logs/memory_2026-11-16_16-30-53/frames/{rgb_0009.png,depth_0009.npy,pose_0009.npz}`.
The runner replays historical RGB-D through the real coordinator dispatcher,
controller method, perception providers and graph admission. Its capture adapter
does not acquire a live frame. There is no simulator, robot action, manipulation
or physical task-success claim. Depth limits stay 0.25–2.5 m.

| Run | Source | Managed job | Result |
| --- | --- | --- | --- |
| r1 | `2c0cecd1` | `20261116_231404_7c8614` | Generic stool admitted; intended nearest-object constraint was ignored by the prototype's separate description channel. Cropped microwave abstained. |
| r2 | `15a0d9e5` | `20261116_231942_73063b` | Complete object phrase reaches the VLM; the foreground stool is admitted. |

Artifacts: `/home/cpaxton/runs/emet/coordinator-grounding-replay-20261116-r{1,2}/`.
Each directory retains its exact runner, source pin, frozen RGB-D/calibration,
parameters, model-call prompts/images/raw responses, admission cache and
coordinator-visible receipts. r1 makes three VLM calls for two tool calls; r2
makes two VLM calls for one tool call. Both managed jobs finish. No thresholds
or model decoding settings were tuned between these checks.

The initial API accepted `query="stool"` plus a separate nearest-object
description. The VLM grounding helper intentionally ignores EQA question context
when a short object query is present. Consequently both proposal and final
selection prompts said only “stool”. Manual inspection of the accepted mask
confirmed the farther stool, despite an intended nearest-object constraint.
The raw admission success is preserved, with intended-target failure recorded
separately; it is not counted as a successful nearest-object localization.

The corrected API has one complete object phrase. Unknown extra arguments fail
explicitly. EQA question-context semantics remain unchanged. In r2, both model
prompts contain “nearest black stool in the foreground”; parent inspection of
the accepted context/isolated mask confirms that foreground stool. The admitted
surface has 20,454 depth points and median world XYZ approximately
`[0.794, 0.734, 0.750]` m. This is a partial observed surface, not full object
geometry or a grasp-clearance certificate. Candidate proposals also change with
the object query, so this is an interface qualification, not an isolated model
prompt ablation or evidence of a general accuracy gain.

Reviewed evidence:

- [Prototype: farther stool selected](/home/cpaxton/runs/emet/coordinator-grounding-replay-20261116-r1/grounding/grounding-53052f272d4b401b8cfea34fbd110ada-surface-0-context.png).
- [Complete query: foreground stool selected](/home/cpaxton/runs/emet/coordinator-grounding-replay-20261116-r2/grounding/grounding-810629c28cad4a9dbec7fe27cdf2591f-surface-0-context.png).
- [Accepted foreground support pixels](/home/cpaxton/runs/emet/coordinator-grounding-replay-20261116-r2/grounding/grounding-810629c28cad4a9dbec7fe27cdf2591f-surface-0.png).

The r1 microwave response explicitly says it is not visible; the object is
mostly cropped from this particular frame. No XYZ is admitted. Raw model reasons
remain in the cache; the subsequent metadata-only follow-up also exposes parsed
model reasons with geometry failures so a coordinator can distinguish semantic
abstention from failed depth support without changing admission decisions.

## Validation and next gate

The final focused CPU pack passes 140 tests: grounding/admission, VLM region
selection, skill/schema membership, config admission, real CHAT/HTTP follow-up,
dialogue retention and extracted acceptance tooling. A real VLM prompt-path
regression checks that the complete constrained object phrase reaches inference;
the separate EQA question-context tests stay intact. Ruff and commit hooks pass.

The new tool has not yet been qualified in a fresh live coordinator episode.
Next run the bounded operator case with `ground_object` exposed, then small
paired EQA/OVMM acceptance checks. Preserve failure rows and safety rejections;
do not launch a broad sweep or infer manipulation success from this replay.
