# WBC coverage process: cases, gates, and voxel observations

Status: process + registry scaffold. The free-space PandaOmron teacher and
DAgger baseline are running; shelf/carry/occupancy cases are the next stages.
This document is the authority for what "good enough" means and where a case
enters training.

## Goal

One controller stack is "good enough" only when it passes a **held-out,
family-partitioned case suite** without privileged inputs. Teacher results and
in-suite DAgger results are diagnostics, not promotion evidence.

The process has four artifacts:

1. **Case registry** — every case family, its source, split, and gate.
2. **Training distribution** — how each case family enters teacher/BC/DAgger.
3. **Evaluation gate** — per-family held-out success, collision, tracking, and
   no-GT audit.
4. **Promotion/regression** — a checkpoint is promoted only when all active
   families pass and older families do not regress.

## Case families

| Family | What it tests | Initial representation | Entry point |
|---|---|---|---|
| `free_space_reach` | arm/base/combined tracking with no obstacles | proprio + target | current default suite |
| `hard_posture` | recorded failure postures and targets | proprio + target | `wbc_hard_cases.py` replay |
| `shelf_entry` | reaching into an opening with an obstacle/frame | proprio + **occupancy** + target | obstacle-conditioned subtask |
| `carry_payload` | holding a payload while moving/reaching | proprio + **occupancy** + payload state | carry subtask |
| `contact_recovery` | blocked/contact-rich corrections | proprio + **occupancy** + contact history | contact subtask |
| `base_arm` | base and arm coordination | proprio + target | combined suite |
| `articulation` | door/drawer opening/closing | proprio + occupancy + articulation state | later adapter |

Each family has `train`, `validation`, and `test` cases. **Test cases are never
used for teacher collection, BC, DAgger, hard-case injection, or checkpoint
selection.** In-suite evaluation is labeled as such; held-out is the gate.

## Case record

Every case is a JSON object (or one entry in a registry):

```json
{
  "id": "pandaomron.sink.failure-019",
  "family": "hard_posture",
  "robot": "pandaomron",
  "split": "test",
  "status": "active",
  "source": "manip_failures.jsonl#19",
  "occupancy": "none",
  "gate": {
    "min_success": 0.8,
    "max_collision_rate": 0.0,
    "max_tracking_error_m": 0.05,
    "max_duration_s": 8.0
  },
  "notes": "recorded tracking failure; held out"
}
```

- `status`: `active` cases gate promotion; `planned` cases are in the registry
  but do not gate until their teacher/provider lands.
- `occupancy`: `none`, `optional`, or `required`. `required` cases cannot run
  against a policy/checkpoint without the occupancy layout.
- `source`: failure record, simulator case id, or generator name. Provenance is
  required.
- No body IDs, object poses, contact labels, or other GT fields may appear in
  the actor observation. Occupancy is **geometry**, not object identity.

## Data provenance

A coverage case is only as good as its record:

- A failure attempt must carry the **pre-motion** `initial_q`. The post-motion
  trace is not the reset state.
- The record must identify the operation (`op`) so place/approach planning
  failures are not replayed as the previous grasp attempt.
- Semantic fields (`object_query`, `receptacle_query`) are always present;
  GT bodies are allowed only under `EMET_GT_TEACHER` and are tagged
  `privileged`.
- The case-library load/add/save is file-locked; concurrent jobs cannot
  overwrite each other.
- Hard cases are split train/test before any replay; the test case is never
  injected into DAgger.

The semantic/GT audit fixes in #267 provide the pre-motion and per-op record
shape; the hard-case exporter uses `initial_q` when present.

## Training stages

1. **Teacher ceiling** per family. A family with no teacher above its ceiling
   does not enter actor training.
2. **BC warm start** on teacher states. Hold out the test split.
3. **DAgger** on actor-visited states. Hard-case probability is reported; hard
   cases are excluded from evaluation resets.
4. **Obstacle/occupancy stage**: the teacher uses privileged scene geometry to
   label; the actor sees the initial occupancy crop. Ablate map/no-map and
   unknown-space masking.
5. **Carry/contact stage**: payload and contact are teacher-side only; the actor
   sees proprio, target, and occupancy. Contact history is a separate channel
   only if the contract adds it.
6. **Whole-task stage**: TAMP plan + WBC execution, evaluated on the same
   family split.

## Voxel-map observation

The contract already carries:

```text
WbcObservation.occupancy          # (H, W, D) uint8, base frame
WbcObservation.occupancy_origin   # world/base-frame origin
WbcObservation.occupancy_resolution
```

`wbc_observation_layout(..., occupancy_shape=...)` reserves a fixed flattened
block. The missing pieces are a provider and an encoder.

### Representation

- **Initial map**: a static scene occupancy crop at reset, in the base frame.
  It is generated from observed/known scene geometry, not object IDs or GT
  poses. For direct MuJoCo training cases the crop comes from the case JSON.
  For RoboCasa it comes from the live session's voxel/map crop.
- **Resolution**: start with 0.05 m, crop 1.6 x 1.6 x 0.8 m -> `(32, 32, 16)`
  (16,384 cells). A 0.10 m `(16,16,8)` baseline is the cheap ablation.
- **Frame**: the crop follows the base; `occupancy_origin` is transformed when
  the base moves. The map is fixed for the episode in v1; live depth updates
  are a later ablation, not a prerequisite.
- **Unknown space**: explicit `unknown` value (e.g. 2), not free. The policy
  must not treat unseen space as traversable.

### Encoder

The current `WbcActorCritic` is a flat MLP. Replace it with a multimodal
encoder:

```text
proprio vector ──► MLP ──┐
                         ├─► shared features ─► actor / critic
occupancy (1,H,W,D) ──► 3D CNN ──► global pool ┘
```

- 3D CNN: `Conv3d(1,16,3,stride=2) -> ELU -> Conv3d(16,32,3,stride=2) -> ELU ->
  global average pool` (small; the crop is tiny).
- The layout/checkpoint stores `occupancy_shape` and encoder kind; a checkpoint
  without occupancy cannot be silently loaded into an occupancy policy.
- The initial map is observed-only; the teacher may use privileged scene
  geometry for labels, but the actor branch receives only the map.
- Ablations: no map, map with unknown/free dropout, 0.1 m vs 0.05 m, static vs
  live crop.

### Providers

- `WbcCaseOccupancyProvider` (training): load a case JSON/npz crop and expose it
  through the backend observation.
- `RoboCasaVoxelProvider` (sim/agent): crop the session voxel map into the base
  frame; fail closed if the map or origin is unavailable.
- Both return the same `WbcObservation` fields; no GT body IDs.

## Gate and promotion

`wbc_coverage.py` loads the registry and a results JSON:

```json
{
  "case_id": {
    "episodes": 20,
    "success_rate": 0.85,
    "collisions": 0,
    "tracking_error_m": 0.03,
    "duration_s": 4.2
  }
}
```

Rules:

- Every active family has at least one `test` case.
- Every active test case has results and meets its gate.
- A family passes only if all its active test cases pass.
- Promotion requires **all active families** to pass and a no-GT audit of the
  observation encoder.
- `planned` families are reported as `not_gated`, not silently passed.
- A failing family blocks promotion; the failure list is written to the report.
- The same gate runs on every checkpoint in CI/GPU jobs; results are stored next
  to the checkpoint.

Recommended starting thresholds:

| Family | Success | Collision | Tracking |
|---|---|---|---|
| free_space_reach | 0.90 | 0.0 | 0.03 m |
| hard_posture | 0.80 | 0.0 | 0.05 m |
| base_arm | 0.85 | 0.0 | 0.04 m |
| shelf_entry | 0.75 | 0.0 | 0.05 m |
| carry_payload | 0.75 | 0.0 | 0.05 m |
| contact_recovery | 0.70 | 0.0 | 0.06 m |

Thresholds are starting points; they are raised only with evidence, never
lowered to make a checkpoint pass.

## Current state (2026-11-26)

- PandaOmron teacher: 1.00 in-suite; DAgger baseline reached 1.00 at iteration
  5 (10 evaluation episodes). Not a held-out promotion.
- Six Panda failure postures exported; five train, one held out.
- Hard-case fine-tune queued from the baseline `best.pt`.
- Occupancy contract/layout exists; no provider or CNN encoder yet.
- Shelf/carry/contact families are registered as planned.

## Next milestones

1. Case registry + gate (`wbc_coverage.py`), split hard cases train/test.
2. Occupancy case provider + 3D CNN actor; no-map ablation.
3. Shelf-entry and carry cases generated from RoboCasa placements (teacher
   privileged; actor map-only).
4. RoboCasa voxel provider and adapter; run the family gate.
5. Whole-task TAMP+WBC gate; promote the first checkpoint.
