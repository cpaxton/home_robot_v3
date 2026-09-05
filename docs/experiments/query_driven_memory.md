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

## Implementation still required before enabling the prototype

- Connect the candidate store to existing graph runtime save/load. The candidate
  serializer preserves references, but restoring geometry always requires reacquisition.
- Connect existing EQA and OVMM query/investigate tools to the same candidate store.
  Candidate creation requires source observation identity; do not fabricate one
  when voxel retrieval returns only a point. Distinct instances cannot share an
  identity merely because they match the same phrase.
- Add a single opt-in policy on the existing lazy backend. After a fresh capture,
  run existing detector/depth admission and fusion before promoting a candidate.
  Navigation completion or a high retrieval score is insufficient for promotion.
- Keep annotations linked to observations, independently of geometric grounding.
- Pass grounded targets to existing manipulation tools; refresh immediately before
  execution, invalidate after moving an object, and observe again after placement.
- Add integration tests for lifecycle callers, persisted references, missing depth,
  object motion, and learned-agent ground-truth isolation through the runner.

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
