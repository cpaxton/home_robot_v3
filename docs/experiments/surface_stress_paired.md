# Paired surface stress diagnostic

Question: does measured-surface selection improve grounding beyond the original
point method when objects touch, occlude one another, share appearance, or are
farther away? This is a component gate for the shared harness, not a substitute
for EQA, OVMM or TAMP episodes.

Frozen source: `c830592e`. Managed serial job: `20260909_170157_1e9f64`.
Qwen3-VL-8B-Instruct, int4, CUDA/SDPA; no model or prompt tuning between rows.
Fixture: `configs/ovmm/surface_stress.yaml`, using the existing kinematically
held Galaxea R1 proxy registered as rby1, not physical RBY1 hardware.

Five layouts each have two target queries and an absent green-mug query. Both
methods see identical rendered RGB-D. Simulator IDs enter only the independent
scorer, never the model prompt, masks proposed to the model, or search boxes.
For visible targets require mask purity >=95% and surface-centroid distance to
object center <=15 cm. Report recall separately: pure tiny patches need not be
useful grasps. Absent/fully hidden targets must abstain. Partial visibility is
reported rather than silently dropping difficult cases.

The same job then replays the existing eight-case room/static manifest through
both methods. Repeated room images are controls, not new independent scenes.
Manual room review must distinguish semantic misses, unsafe false acceptance,
and proposal overflow. A successful static result cannot cancel a room failure.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  EMET_ALLOW_SDPA_ATTN=1 MUJOCO_GL=egl PYTHONPATH=src \
  python scripts/eval_surface_stress.py --output-dir /absolute/new-output
```

Artifacts: `/home/cpaxton/runs/emet/surface-stress-paired-20260909` contains
original RGB-D, overview figures, visibility reports, model responses, exact
candidate panels/masks, manifest, and independently scored `scores.json`.
Room controls are in sibling `surface-room-point-20260909` and
`surface-room-candidates-20260909` directories.

Decision rule: improvement on the paired object cases supports further work,
but false acceptance under occlusion or same-appearance contact is a blocker
for promoting the method. Persistent room failures require a general proposal
or viewing improvement, not target-specific prompts. End-to-end acceptance must
still demonstrate fresh grounding through the shared OVMM/TAMP handoff, then a
small paired EQA no-regression check. No full sweep is required.

## Completed perception results

| Layout (two targets each) | Point pass | Surface pass |
| --- | --- | --- |
| Separated | 1/2 | 2/2 |
| Touching | 1/2 | 2/2 |
| Partially occluded | 2/2 | 2/2 |
| Farther | 0/2 | 1/2 |
| Same-color touching | 0/2 | 2/2 |
| Total targets | 4/10 | 9/10 |
| Absent-query abstentions | 5/5 | 5/5 |

Point has three incorrect acceptances and three misses on targets. Surface
has one incorrect acceptance and no target abstentions. All nine passing
surface masks have 100% simulator target purity. These are ten correlated
object cases in one asset family, not evidence of a general 90% success rate.

The remaining surface error is important: the farther red-cylinder box misses
the cylinder. The sole candidate is 102 pixels of table, and Qwen confidently
describes it as red cylinder. Purity is zero and center error is 69 cm. Thus
candidate selection improves this battery but **does not reliably reject a
bad proposal set**. Keep the strategy experimental; do not treat its binary
acceptance as calibrated confidence or authorize physical grasping from it.

The eight-case replay control again misses the visible lamp in both methods;
the candidate method abstains on the textured sofa because it exceeds eight
proposals, while point accepts a region (not independently scored as a correct
sofa mask). Candidate recovers both static colored targets; point selects
background for both on this replay. The point method's varying results across
calls further caution against interpreting a single successful frame as robust.

Next evidence priority: improve general search-box/candidate coverage and
test rejection when **none** of the candidates belongs to the target. Recover
room-scale texture without discarding possibly relevant proposals. These are
two distinct problems; increasing the candidate cap cannot fix a missed box.
No target-specific color heuristics or lower admission thresholds were added.

One bounded integrated Stretch simulation is recorded separately as managed
job `20260909_170512_6f6e56`; it is a shared `pick_place` handoff diagnostic,
not an independently scored OVMM benchmark episode. It runs only after the
paired perception job releases the GPU lock, with a 360-second process limit.

Integrated outcome: router selected `pick_place(red cylinder, blue cube)` and
entered fresh capture, then the 32-token image-caption call timed out at 180 s
(first decode token only after 74.4 s). Grounding and manipulation were never
reached. The timed-out shared client refused reuse; the process emitted an
abort message but remained alive, so the managed job was cancelled for cleanup
after recording the tool failure. Cancellation is not a successful episode.
No grounding RGB-D bundle was produced before this early caption failure; the
managed log retains the command, model, tool arguments, timings and traceback.
Save pre-caption evidence and diagnose this latency before repeating task runs.

Relevant fixture/proposal tests: 15 passed. Production defaults remain unchanged
and PR #167 remains draft. This batch supports a narrower design hypothesis,
not a cross-benchmark success or no-regression claim.
