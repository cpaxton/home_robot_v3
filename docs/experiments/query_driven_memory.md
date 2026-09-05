# Query-driven memory implementation and pilot

The objective is a shared EQA/OVMM harness with query-driven instance creation,
image-supported annotations, and fresh geometry at the manipulation boundary.
Retrieval candidates are not confirmed objects. No full sweep is authorized.

## Landed prerequisites

- Foundation draft PR #162: ingestion controls, DynaMem EQA fixes, isolated-run tooling.
- TAMP PR #160 merged into main at `4ed75850`; incorporated here. Its documented
  Innate Mars navigation and Sourccey cabinet placement failures remain open.
- Learned Habitat OVMM clients no longer receive semantic labels or placement
  metadata through observations/session data; the evaluator retains ground truth.
- Candidate lifecycle API and additive localization-result fields have focused tests.

## Connected prototype (opt-in; not a validated benchmark row)

The lazy controller now uses the same query-candidate path in agentic EQA and
OVMM find. Voxel retrieval records its source frame ID; repeated queries reuse a
stable handle without adding object nodes. After investigate captures a new voxel
frame, the controller runs detection on demand if streaming masks are absent,
then applies existing depth admission and fusion. Successful promotion exposes
the arrival image as the station evidence; it does not label everything at the
navigation target. The tool result and trace include the grounding outcome.

The graph runtime checkpoint stores candidate references. Reload always revokes
grounding, including when the resumed observation counter matches the old counter.
Old checkpoints without candidate state remain supported.

Enable these parameters before constructing `LazyGraphController` (backend
`lazy_graph`) for either task:

```yaml
query_driven_memory: true
eqa:
  agentic_verify: true
graph_object_fusion:
  enabled: true
  use_instance_nodes: true
```

Keep the other model, mapping, admission, fusion, and budget settings frozen.
The checked-in default remains `query_driven_memory: false`. This policy requires
the agentic loop; the classic arrival-label path is disabled when it is enabled.
OVMM must use agentic find, not the one-shot find adapter.

Promotion currently requires one exact normalized detector-label match. A generic
`mug` detection does **not** verify `red mug`, and shared words do not establish
identity. Unsupported attributes remain ungrounded until semantic verification is
connected. Ambiguity is checked before the admission candidate cap, so truncation
cannot turn two possible objects into a unique match. Candidate capacity rejects
new proposals without silently evicting live references.

## Remaining acceptance work

- Connect attribute verification to observation-linked annotations, independently
  of geometric grounding and without weakening target identity checks.
- Pass grounded targets to existing manipulation tools; refresh immediately before
  execution, invalidate after moving an object, and observe again after placement.
- The current manipulation wrapper reacquires by text, not by this candidate's
  geometry/identity. It is **not** yet an end-to-end learned pick/place handoff.
- Controller tests cover on-demand detection, mask-derived geometry, absent and
  ambiguous targets, missing depth, instance opt-out, stale captures, checkpoints,
  and shared EQA/OVMM retrieval. Run live simulator acceptance before claiming
  task success; these tests do not establish answer or manipulation quality.

## Limited acceptance rotation

- Resume the interrupted EQA random-16, retaining completed artifacts. The old
  no-instance row is diagnostic because its switch bypass was subsequently fixed.
- Compare DynaMem, arrival-only lazy memory, and query-driven memory on those 16
  questions. Freeze source, model, initial states, budgets, and row settings.
- Define six matched OVMM cases before outcomes: visible, occluded/cluttered,
  repeated objects, initially unseen, absent, and moved/revisited target.
- Run TAMP table pick/place, mixed-grasp rejection, rby1 relocation, and kinematic
  agent-tools gates; include at least one learned retrieval-to-manipulation sequence.
- Report per-episode outcomes, errors, grounding success, duplicate candidates,
  stale-target rejection, model calls, runtime, and memory size. Maintain explicit
  known-failure rows rather than silently treating them as passing coverage.

The prototype is incomplete until both EQA and OVMM exercise the shared lifecycle
and a learned target completes pick/place followed by fresh observation.
