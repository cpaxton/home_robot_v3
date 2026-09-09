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
| Command contract | query memory | `18f11d11` | Includes prerequisite posture/health probes, idempotent dispatch, receipts, cancellation and deployable runtime |
| Grounding corrections | command contract | `4cb7abde` | Camera/targeting/evidence corrections, honest evaluation selection, stationary and live probes |
| Navigation acceptance | grounding corrections | `fix/shared-navigation-acceptance` | In progress; opt-in completion policy and repeated cross-robot simulation acceptance |
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
  cross-robot or integrated acceptance; live rerun is pending.
