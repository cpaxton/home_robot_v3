# Collision-map-assisted native contact, November 17

The stationary Stretch development control completes public-tool pick/place
with a real rendered RGBD collision map, including release and retreat. This
is **provided-map, native-state-assisted simulation**, not observed-only or
LLM-agent acceptance. No physical robot, VLM, GPU, attachment latch, or runtime
object pose writes are used. In-process physics advances on controller sleeps;
these runs do not measure real-time latency.

The runner is `scripts/eval_mapped_native_contact.py`. It uses the same
`scene_tasks`, `plan_pick_place`, and `execute_pick_place_plan` tools. Normal
actuator commands operate the physical gripper. Native state supplies robot,
passive-joint and object geometry. An explicit simulated overhead background
scan is acquired before inserting the target and robot, with calibrated
pinhole depth and a selected wood support patch. This is a provided map prior,
not an extra sensor claimed on Stretch. Geometry outside the map, including
whole solids straddling its boundary, retains the native scene check. The
reported scope is therefore **whole-solids-contained map checks plus native
scene checks**, not complete observed-map coverage of the robot.

## Repairs exercised

- Exact original depth endpoints can now reach tracking, and raw depth/K/pose
  are retained for carving. Registered RGB pixel centres do not replace the
  winning original sensor rays. This source diagnostic passes 17 captures and
  two paused-consumer windows (`20261117_154213_9604c2`); no task was attempted
  in that diagnostic.
- Canonical payload material and pose/attachment uncertainty are separate.
  Transport counts both EE captures once; retention counts each relative-pose
  error once. All grasp alternatives contribute to the lever reserve, checked
  again against actual attachment/retention geometry. Region certification,
  release and hypothetical-state restoration retain physical uncertainty.
- Original selected support rays can prove a subvolume free even when its
  coarse cell is unknown. A finite backing-layer prior spans voxel boundaries
  only for an eligible payload contact. Other/mixed occupied cells still win;
  unknown shadow, missing pixels, excessive penetration, and material behind
  the declared layer remain blocked. The 1 mm contact bound is unchanged.
- Native checkers support conjunctive scene guards. Held-pose, opening and
  falling-object/retreat clones preserve the map guard and supply their own
  private geometry. An opaque wrapper was correctly rejected by bounded
  release; the fix preserves its exact native-type contract. Invalid guard
  results reject, and collision reasons are cleared between queries.
- `ObservedEnvironmentAdmission` queries the map without robot ownership or
  fabricated sensor evidence. Observed execution still requires the stricter
  `ObservedMaterialAdmission`. Exclusive owner sets are cached on immutable
  backgrounds; replacing a background builds a fresh cache.

## Results and accounting

The compact [JSON index](mapped_contact_20261117.json) distinguishes completed
tasks, expected negative controls, construction failures, cancellations and a
native runtime crash. Full inputs, commands, traces, map rejection geometry,
source hashes and job wrappers remain under
`.tamp-runs/observed-map-pick-20261117/`. Negative controls are **not successful
pick/place tasks**, and development retries are not a benchmark denominator.

The 2 cm cylinder control and repeat complete grasp/lift/place/release/retreat
with 52,482 map queries each. The repeat measures 0.1322 m lift and 0.0750 m XY
relocation; final object centre is approximately `[-0.02020,-0.62495,0.51991]`.
Native-only references also pass. The trajectory figure is available at
`.tamp-runs/observed-map-pick-20261117/figures/mapped-transfer.png` and `.pdf`.

The after-grasp evidence-loss control returns `partial`, `code=collision`,
`execution_started=true`, `completed_ops=["grasp"]`, `failed_stage=preplace`,
and `recovery=inspect`. Command count stays **54 → 54** after the fault. This
counterfactual explicitly removes map and ray-witness qualification; it is not
a claimed new camera measurement. The occupied-map control rejects before any
command. These verify stop/receipt behavior, not autonomous recovery.
The final 5 mm/wider-region repeats also pass these checks
(`20261117_163143_857f43`, `20261117_163143_b5200c`), with the same 54 → 54 and
zero-command accounting respectively.

The cube exposes a real coverage limitation at 2 cm: a voxel at the table edge
mixes selected support with unclassified edge returns. Those returns are kept.
The finer trial uses a **5 mm map over the same workspace**, and a 2 mm measured
support footprint. Its source bounds remain inside measured pixels. A separate
10 cm-deep requested region cannot accommodate the cube's enclosing footprint
plus clearance for viable grasps; that diagnostic is retained and cancelled
before execution. Prospective paired cube/cylinder trials instead request
`[[-0.18,-0.73],[0.18,-0.57]]`. No collision/contact threshold is relaxed.

Both prospective 5 mm cases complete the full physical task through retreat:

| Development case | Map queries | Native-only queries | Final object centre (m) |
| --- | ---: | ---: | --- |
| Cube | 75,113 | 82,252 | `[-0.02045,-0.62275,0.51989]` |
| Cylinder | 52,547 | 64,804 | `[-0.02020,-0.63495,0.51991]` |

The job IDs are `20261117_162701_773ecc` and `20261117_162701_40a297`.
The cube requires bounded release correction before opening; that correction,
opening, falling-object verification and retreat retain the map gate.
These two development successes establish integration, not a general success
rate, held-out robustness or sensor-only execution. The full workspace remains
`[[-0.4,-0.9,0.44],[0.4,-0.3,1.04]]` at both voxel resolutions.

The [paired trajectory figure](figures/mapped_contact_20261117.pdf) shows these
two final cases. Cube/cylinder lift is 0.1327/0.1322 m; XY relocation is
0.0728/0.0850 m. Only the figure's voxel display is subsampled; collision
queries retain the full map. The figure labels all state/map assistance.

One parallel recovery attempt crashes natively (`exit=139`) before fault
injection. Its launch accidentally permits CPUs 8–11, documented unstable turbo
cores on this workstation. A serial safe-core repeat passes. The runner now
intersects its allocation with the existing safe CPU selection and records the
actual mask; this is **not a proven root cause or native-runtime fix**. The crash
remains a separate reliability incident. The larger prescan also exposes an old
native snapshot; the runner acquires current actual state after construction,
without renewing map acquisition times.

The broad motion/TAMP batch passes **1,044 tests**, zero failures/errors/skips
(`20261117_160946_caa24c`). The subsequent environment-only extraction passes
192 focused checks (`20261117_162321_cd78e8`); immutable-owner caching passes
129 (`20261117_162702_655cf6`). These overlapping suites are not task counts.

## Remaining observed gate

This control isolates map/refinement/execution integration. It does not qualify
an onboard map, measured base/floor contact policy, passive finger geometry, or
hardware sensors. Strict observed execution still needs a complete source and
robot-only refiner, genuine free evidence around the robot's uncertainty shell,
and independently justified **relative** payload/support and gripper bounds.
The existing 3 mm object-position debug budget alone exceeds the 1 mm support
contact limit; a successful privileged control does not resolve that inequality.
No read-only discovery default is promoted to execution, and no paper agent
success score is inferred from these controls.
