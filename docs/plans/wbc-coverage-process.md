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
3. **Evaluation gate** — per-family held-out success, collision, tracking,
   measured contact availability, and no-GT audit.
4. **Promotion/regression** — a checkpoint is promoted only when all active
   families pass and older families do not regress.

## Case families

| Family | What it tests | Initial representation | Entry point |
|---|---|---|---|
| `free_space_reach` | arm/base/combined tracking with no obstacles | proprio + target | current default suite |
| `hard_posture` | recorded failure postures and targets | proprio + target | `wbc_hard_cases.py` replay |
| `shelf_entry` | reaching into an opening with an obstacle/frame | proprio + **occupancy** + target | obstacle-conditioned subtask |
| `carry_payload` | holding a payload while moving/reaching | proprio + **occupancy** + **wrist wrench/finger force** | carry subtask |
| `contact_recovery` | blocked/contact-rich corrections | proprio + **occupancy** + **wrist wrench/contact history** | contact subtask |
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
  "contact": "required",
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
- `contact`: `none`, `optional`, or `required`. `required` cases must report
  measured contact state on every step (`contact_state_rate == 1.0`); a missing
  or constant-unavailable contact channel fails the gate.
- `source`: failure record, simulator case id, or generator name. Provenance is
  required.
- No body IDs, object poses, GT contact labels, or other GT fields may appear in
  the actor observation. Occupancy is **geometry**, not object identity.
  Measured wrist wrench/finger loads are plant state, not GT; privileged
  contact labels remain teacher/critic-only.

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
5. **Carry/contact stage**: the actor sees proprio, target, occupancy, measured
   wrist wrench, and per-finger loads. Payload identity/pose and privileged
   contact labels stay teacher/critic-side. Hard postures are replayed with the
   same measured contact channels.
6. **Whole-task stage**: TAMP plan + WBC execution, evaluated on the same
   family split.

## Force and contact state

The actor is mostly blind while the arm is in motion, so force/contact is a
first-class observation, not only a safety check:

```text
WbcObservation.arms[i].joint_torque       # measured/commanded joint torque proxy
WbcObservation.arms[i].contact            # ContactObservation
WbcObservation.arms[i].gripper.force      # per-finger normal load
```

`ContactObservation` carries:

- `in_contact`: a measured contact event (external hand/object/environment).
- `wrench`: `(fx, fy, fz, tx, ty, tz)` in the **EE link frame**. For Panda it
  comes from the wrist `gripper0_force_ee` / `gripper0_torque_ee` sensors,
  transformed from the F/T site frame to the EE frame and sign-corrected to be
  the external load on the distal chain.
- `source`: measurement provenance, e.g. `ft_sensor`, `contact_normal`, `none`.

The raw F/T value is **not tared**: it can include the gripper's own
gravity/inertial load as well as payload and environment contact. That is what a
real wrist sensor reports; the policy learns the load. The separate
`contact_flag` and per-finger forces come from solved contacts, so contact
presence does not depend on a gravity model.

Layout fields (appended per arm, layout version 3):

```text
arm.<name>.has_contact_state   1
arm.<name>.contact_flag        1
arm.<name>.contact_wrench      6   # normalized by 50 N / 10 Nm, clipped
arm.<name>.gripper_force       m   # normalized by 20 N, clipped
```

The encoder receives a presence flag, so a robot without a force sensor reports
`has_contact_state = 0` and zeros rather than silently inventing contact.
PandaOmron currently reports the F/T sensor and per-finger loads. The hard-case
replay and held-out gate require `contact_state_rate == 1.0` for the
`hard_posture`, `carry_payload`, and `contact_recovery` families.

No GT enters this path: the sensor reads simulated force/torque and solved
contact loads. Object names, object poses, and privileged contact labels are
never consulted. A teacher/critic may use those to label or critique; those
records are tagged `privileged` and excluded from the actor observation.

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

- **Initial map**: a static scene occupancy crop at reset. It is generated from
  observed/known scene geometry, not object IDs or GT poses. For direct MuJoCo
  training cases the crop comes from the case JSON/npz; for RoboCasa it comes
  from the live session's voxel/map crop.
- **Target-centered crop**: center on the active EE target (reach) or the
  receptacle support target (shelf/carry), not only the base. This keeps the
  shelf opening, frame, and payload clearance in the same local view.
- **Resolution**: start at 0.04 m, crop 1.28 x 1.28 x 0.64 m -> `(32, 32, 16)`
  (16,384 cells). A 0.10 m `(16,16,8)` baseline is the cheap ablation.
- **Channels**: occupied (0/1), unknown (0/1), normalized ESDF/clearance.
  The ESDF channel is the key feature for contact-aware torque control; raw
  occupancy alone makes the policy learn clearance from scratch.
- **Frame**: the crop follows the target; `occupancy_origin` is transformed when
  the base moves. The map is fixed for the episode in v1; live depth updates
  are a later ablation.
- **Unknown space**: explicit `unknown` channel, never free. The policy must not
  treat unseen space as traversable.

### Encoder

The current `WbcActorCritic` is a flat MLP. Replace it with a multimodal
encoder:

```text
proprio vector ──────────────► MLP ──┐
                                     ├─► shared features ─► actor / critic
occupancy (3,H,W,D) ─► 3D CNN ───────┘
```

Recommended primary:

- `Conv3d(3,16,3,stride=2) -> BN -> ELU`
- `Conv3d(16,32,3,stride=2) -> BN -> ELU`
- `Conv3d(32,64,3,stride=2) -> BN -> ELU`
- global average + max pool -> 64 features
- proprio MLP `2 x 128`; concat -> 192; actor/critic heads.
- ~100k parameters; the 32x32x16 crop is small enough for a dense 3D CNN.

Baselines/ablations:

- **2.5D**: BEV heightmap + occupancy in 3-4 z bands + ESDF; 2D CNN. Cheaper,
  but it loses overhangs/shelf cavities; useful as a sample-efficiency baseline.
- **Tri-view**: BEV + xz + yz projections with a shared 2D CNN; captures shelf
  cavities more cheaply than dense 3D but needs a projection convention.
- **Sparse voxel/PointNet**: later option if the crop grows or resolution
  increases.

Pretraining:

- Capture many initial maps from **procedural RoboCasa scenes** (layouts,
  styles, seeds) and train the encoder with a masked-occupancy/clearance
  auxiliary (predict held-out voxels or ESDF from a partial map). This is
  self-supervised and uses no policy, object IDs, or GT poses.
- Then fine-tune the encoder end-to-end with DAgger on free-space, hard-posture,
  shelf, and carry cases. Freeze the encoder for the first policy BC stage if
  data is limited; unfreeze for the family fine-tune.

Checkpoint:

- The layout/checkpoint stores `occupancy_shape`, channel order, encoder kind,
  and crop frame. A checkpoint without occupancy cannot be silently loaded into
  an occupancy policy.
- The initial map is observed-only; the teacher may use privileged scene
  geometry for labels, but the actor branch receives only occupancy/ESDF.
- Ablations: no map, map with unknown/free dropout, 0.1 m vs 0.04 m, target vs
  base crop, static vs live crop.

### Procedural capture

Add a capture tool that:

1. Generates RoboCasa kitchen scenes over a fixed layout/style/seed grid.
2. Places the robot at the family's base reference (counter/sink/cabinet).
3. Computes the initial target-centered occupancy/ESDF crop from scene
   collision geometry.
4. Saves the exact encoder input, origin/resolution, target/base, family, and
   train/validation/test split to npz/JSON.
5. Runs the encoder offline to cache features and evaluate reconstruction,
   clearance prediction, and price/latency.

Test scenes are held out by layout/style/seed; they are never used for encoder
pretraining or policy DAgger. The same captures feed the coverage gate.

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
    "duration_s": 4.2,
    "contact_state_rate": 1.0
  }
}
```

Rules:

- Every active family has at least one `test` case.
- Every active test case has results and meets its gate.
- A family passes only if all its active test cases pass.
- A case with `contact: "required"` fails if `contact_state_rate` is missing or
  below 1.0; the contact channel must be present on every step, not only when a
  contact happens.
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
- Hard-case fine-tune completed from the baseline `best.pt`: teacher 1.00,
  BC 0.97, DAgger iterations 1.00/0.97/0.93/1.00 in-suite. This is a
  pre-contact layout-v2 checkpoint; the held-out `failure-019` gate and a
  layout-v3 retrain are still pending.
- Measured contact state is implemented for PandaOmron: wrist F/T wrench,
  per-finger normal loads, contact flag, and self/external contact separation.
  Observation layout version is 3; older checkpoints must be retrained.
- Occupancy contract/layout exists; no provider or CNN encoder yet.
- Shelf/carry/contact families are registered as planned; their gate requires
  `contact_state_rate == 1.0` once activated.

## Next milestones

1. Case registry + gate (`wbc_coverage.py`), split hard cases train/test.
2. Occupancy case provider + 3D CNN actor; no-map ablation.
3. Shelf-entry and carry cases generated from RoboCasa placements (teacher
   privileged; actor map-only).
4. RoboCasa voxel provider and adapter; run the family gate.
5. Whole-task TAMP+WBC gate; promote the first checkpoint.
