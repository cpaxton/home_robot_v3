# Shared navigation delivery and review ledger

## Frozen starting point

Implementation starts from `fix/grounding-robot-validation` at `4cb7abde`.
Remote `main` is `4ed75850` (PR #160 already merged); local `main` is stale.
The three-dot review diff is 72 commits, 162 files, +8430/-857 lines.
Do not include local virtualenv or third-party symlinks in a PR.

## Review boundaries

| Layer | Base | Head | Status / evidence |
| --- | --- | --- | --- |
| Evaluation foundation (#162) | main | `44c7710e` | Existing PR; ingestion switches, fusion controls, baseline/evaluation tooling |
| Query memory (#163) | foundation | `4c869e03` | Extend published `903b0a87` by seven cohesive memory/evaluation commits; exclude later transport work |
| Command contract (#164, draft) | query memory | `18f11d11` | Includes prerequisite posture/health probes, idempotent dispatch, receipts, cancellation and deployable runtime |
| Grounding corrections (#165, draft) | command contract | `4cb7abde` | Camera/targeting/evidence corrections, honest evaluation selection, stationary and live probes |
| Navigation acceptance (#166, draft) | grounding corrections | `fix/shared-navigation-acceptance` | In progress; opt-in completion policy and repeated cross-robot simulation acceptance |
| Integrated pilot | accepted navigation | pending | Frozen OVMM/TAMP/EQA rows and recorded outcomes |
| Paper alignment | pilot artifacts | pending | Methods, limitations, results and reproducible figures |

The historical boundaries are ancestors, so this stack retains the complete
starting tree without cherry-pick loss or rewriting published history. Inspect
each incremental diff and run its focused tests before marking ready. #161 is
still open: compare its fusion behavior/configuration/docs against #162 before
closing it as superseded. Do not infer coverage from similar PR titles.

## Acceptance order

1. Fix the live hold discrepancy. Preserve planar pose but allow passive base
   support dynamics; do not mask camera error with calibration offsets.
2. Introduce capability-advertised opt-in measured completion, noise-aware
   settling, bounded absolute-goal correction and confirmed-stop failure paths.
3. Five serial hold/turn/translation repeats for Galaxea proxy, Sourccey,
   Stretch and Mars in simulation. No teleport. Then fixed-seed disturbances
   and camera/embodiment checks in RoboCasa and MolmoSpaces.
4. Run the bounded paired pilot in `shared_agent_paper_update.md`: DynaMem,
   arrival-only and query-driven rows; six OVMM scenarios on Sourccey/Galaxea,
   PR #160 TAMP controls plus learned sequences, and development random-16 EQA.
   All rows share agent/tool/navigation semantics. Keep oracle controls apart.
5. Update runnable docs and results after every batch; revise paper claims and
   figures from frozen manifests, including failures. No full sweep.

Keep legacy navigation defaults until acceptance. Mars hardware is stationary
only, with previously authorized head tilt; no base/arm/wrist/gripper motion.
Sourccey simulation is required now; physical bridge support remains unvalidated
until hardware commissioning. No hardware connection is needed for these gates.

## Current results

- Initial strict live gate failed: camera tilt error 0.10873 rad, no route run.
- Reproduction: a 21 micrometre base-height offset under six-DOF pinning changes
  the settled optical tilt from approximately -35.8 to -28.6 degrees.
- Holding only planar pose while leaving height/roll/pitch dynamic passes the
  isolated production-hold test at three starting heights. This is not yet
  cross-robot or integrated acceptance.
- Job `20260908_200752_bedca4` at `73ce4552`, artifacts
  `/tmp/emet-live-settled-precision-route-20260908`: settled ten-second hold
  passes (max pitch error 0.002866 rad), +10-degree turn and return pass
  (goal errors 0.02986/0.02979 rad). Translation fails: the chassis tips, even
  though planar XY enters its 2 cm tolerance. No remaining route or integrated
  pilot was run. Camera/body posture must be checked alongside SE(2) arrival.
- The asset has **no wheel collision geoms**: only its chassis contacts the
  floor, and visual wheels extend about 4 cm below the configured floor plane.
  Adding wheel collisions alone in a local physics experiment did not establish
  reliable translation; that experimental asset edit was not retained. A matched
  wheel-support/actuation model needs validation, not a tolerance adjustment.
- Passive height/roll/pitch support is now an explicit constructor opt-in
  (`passive_base_support=True`, enabled by the diagnostic). Legacy base hold
  and velocity behavior remain the default until translation acceptance.
- New navigation policies reject unsafe base posture through the shared failure
  path. Stretch simulation uses timestamped episode-frame measurements; correction
  preserves the resolved frame rather than unconditionally treating it as world.
  ROS hardware policy support is not advertised until its adapters are validated.
- Job `20260908_201851_522c32` at `627438f4`, artifacts
  `/tmp/emet-live-posture-fault-stop-20260908`: hold/turn/return pass again;
  translation returns `failed`, reason `base posture unsafe`, with
  `stop_confirmed: true`. Final stopped XY error is 0.09364 m, not a successful
  arrival. The robot settles upright after stopping. The old probe summary
  then waited for settling at the unreachable goal; the updated probe records
  the failed receipt immediately instead. This is a **fault-stop validation**,
  not a nominal route pass.
- Focused navigation/load/controller tests: 72 passed, one skipped. Existing
  Sourccey model and MolmoSpaces merge tests: 15 passed, one skipped. Hardware
  was not contacted; no integrated learned benchmark has been launched.

## Remaining gates (not complete)

### September 8 continuation

- ZMQ PR #135 merged separately as `b2db951e` after 113 focused tests.
  The navigation branch/pilot below does not yet include that transport merge.
- Habitat EQA can run independently of the failing MuJoCo embodiment gates.
  First development unit: job `20260908_224906_e81fd9`, DynaMem question 1,
  frozen source `30cf14f8` at `/tmp/emet-pilot-30cf14f8`, artifacts and paired
  plan under `/tmp/emet-eqa-pilot-20260909`. Shared agentic/router enabled,
  no GT semantics/enriched labels, 20 planning / 10 movement budget, Qwen3-VL
  8B, map/video requested. Onboard DINOv3/H.264 explicitly disabled. Remaining
  rows/IDs are planned, not launched. This is not a locomotion acceptance run.
- Sourccey job `20260908_214330_b65b89` failed its initial image-orientation
  gate before motion. Its extra `flipud` inverted the asset's upright front
  cameras. Removed it and added an optical-frame/pixel-transform regression
  test; rendered health/precision-route acceptance remains pending.
- Only one heavy experiment runs at a time. Sourccey debugging during EQA is
  source inspection and lightweight geometry checks, not a second simulation.

- Validate the posture-fault live stop, repeated routes and disturbances.
- Repair/validate Galaxea proxy wheel-supported translation without declaring
  idealized direct base motion to be a physical locomotion result.
- Run Sourccey, Stretch and Mars simulator acceptance and scene-health checks.
- Complete timestamp/controller-policy integration for hardware bridges using
  offline transport tests, before stationary hardware commissioning.
- Run the frozen integrated pilot, then update paper results/figures. No new
  learned OVMM, TAMP or EQA result is claimed by these navigation changes.
