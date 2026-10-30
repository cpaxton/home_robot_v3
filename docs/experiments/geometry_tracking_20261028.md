# Geometry tracking — October 28

Geometry registration now works on the smooth cylinder that failed RGB material
feature admission. The successful live repeat passed **5/5 geometry fits and 5/5
observed discovery refreshes**, using CPU OSMesa and Caliban. This is a stationary
sensor diagnostic with a **declared shape prior**, not observed pick/place,
contact validation, an LLM task score or hardware qualification.

![Live RGBD and fitted declared geometry](figures/geometry_tracking_20261028.png)

## Implementation

`GeometryTracker` fits fresh segmented 3D points against a fixed geometry model.
Bounded multi-start optimization retains distinct poses and numerical nullspaces;
all original points, including unsampled outliers, must pass surface validation.
Lost surfaces, replay, stale receive times, scene changes and exhausted work
budgets clear the current fit. Previous poses seed search rather than substitute
for new measurements.

The known-model debug tier uses generic sphere/cylinder/box dimensions, explicitly
labelled `known_model_prior`. The cylinder fixture declares radius 20 mm and
half-height 40 mm; runtime fitting does not query simulator object poses, contacts
or attachment state. The semantic object name selects a measured component and
does not choose the geometry family. The prior JSON is an explicit CLI input.

`MeasuredSurfaceModel` supplies the same fitting interface from measured reference
surfaces plus a separate free-space enclosure certificate. Static multi-view
fusion preserves capture identities, scene revision and the oldest original
receive time. This tier has synthetic registration coverage; it has **not** passed
a live enclosure test. Unseparated support contact still leaves the recorded
single-view object volume unbounded. No visible-point AABB bypass was added.

Continuous shape ambiguity remains explicit. A cylinder's fitted centre and axis
can be locally observable while yaw and axis sign remain unresolved. Model
candidate bounds and numerical rank tests are diagnostics, not certified pose
uncertainty or complete search coverage. Conversion to the native executor's
unique rigid pose still rejects that ambiguity. Position/rotation bounds must be
independently calibrated; surface residuals cannot supply them.

`check_geometry_comotion` compares every retained pose pair with measured EE
motion, accounting for declared shape symmetries. Tests reject dropped/static
objects and reused captures. Its result always has `execution_qualified=false`:
geometry agreement alone does not establish encoder alignment, contact, repeated
temporal evidence or gripper clearance.

## Both live attempts

| Managed job | Geometry fits | Discovery refreshes | Combined diagnostic |
| --- | --- | --- | --- |
| `20261028_182510_a31d20` | 5/5 | 1/5 | Failed: semantic reference expired |
| `20261028_182754_73e62d` | 5/5 | 5/5 | Passed |

The first attempt interleaved support grounding before object refreshes. Three
remote semantic calls plus registration pushed the original object reference
past its existing 30-second lifetime. The provider correctly returned
`surface_reacquisition_stale`; the attempt is preserved as failed. The harness now
measures the independent support candidate after object refreshes. It retains the
30-second semantic and two-second current-observation gates and every original
capture time. It does not renew the reference timestamp.

The repeat made three Caliban calls total (two object selections, one support
seed), with no inference during five geometry updates. Fits took **0.97–1.28 s**
and had approximately **0.30 mm surface residual**. This is residual to the prior,
not independently measured pose accuracy. Each fit retained one centre/axis
candidate together with its unresolved yaw. The support diagnostic also produced
a fresh partial footprint; its margin remains provisional. Default manipulation
continued to refuse an uninstalled observed execution source, and the admission
checks issued no additional commands. Both attempts share one fixture/seed;
they are not independent held-out task episodes.

Raw files are under `.tamp-runs/geometry-tracking-20261028/{live,repeat-live}`,
including RGBD arrays, masks, calibration, original timing metadata, fit poses and
nullspaces. The [JSON evidence](geometry_tracking_20261028.json) preserves both
attempts, script/prior hashes and validation counts. Trials ran from the working
tree based on merge `1844e4be` (`main` at `cd169673`); test and registration
results are kept separate from task denominators.

## Validation and next gate

Managed CPU job `20261028_182521_e0619a` passed **748 checks**, zero failures,
errors or skips, spanning motion, TAMP, observed source contracts and the merged
agent dialogue/VLM paths. The final focused geometry suite passed **9 tests**,
including additional measured-model registration and motion updates. These
counts overlap and are not summed. Ruff and whitespace checks passed; the paper
build remains 52 pages without undefined references or citations.

Next: qualify pose/alignment bounds against an independent evaluator and implement
symmetry-aware source/executor consumption before enabling real observed
transfers. Moving-object identity/mask production, complete observed payload
extent and phase-specific robot/target/contact occupancy remain source gates.
The current live harness only reacquires a stationary reference surface. The
existing privileged native controls and prior task-success metrics are unchanged.
