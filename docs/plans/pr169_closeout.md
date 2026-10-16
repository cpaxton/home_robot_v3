# PR #169 disposition audit — 2026-10-11

Close #169 as a superseded review container, **not** as a claim that room OVMM
or every manipulation experiment passed. Keep its source branch and immutable
commits. Do not merge its 240-file inherited experiment diff wholesale.

Audit references:

- Source: `09033bfc286325402c400e903051960bdcad9ad1` (`fix/room-wheel-contact-profile`).
- Current main: `61aef83a`.
- Active replacement reviews: [#181](https://github.com/cpaxton/home_robot_v3/pull/181),
  [#182](https://github.com/cpaxton/home_robot_v3/pull/182),
  [#184](https://github.com/cpaxton/home_robot_v3/pull/184),
  [#185](https://github.com/cpaxton/home_robot_v3/pull/185).
- Manipulation owner: [#179](https://github.com/cpaxton/home_robot_v3/pull/179),
  audited at `7a190fc0`. Its current kinematic/TAMP implementation is not the
  old experimental `GraspObjectOperation` implementation.
- [Frozen inherited-file inventory](shared_stack_inventory.json) remains the
  archive for inherited #167/#168 changes. This report covers #169's 31 net
  changed files relative to its old PR base, and its later commit/revert history.

## Changes retained or migrated

| Source change / files | Disposition |
| --- | --- |
| `52bebc6d`: `stretch.xml`, `test_stretch_wheel_contacts.py` | Missing from main and extracted stack. Extracted to #182 as `c6809aa6`, with original attribution. 17 wheel-contact, arrival-monitor and import-boundary tests pass. It changes contact physics; rerun physical navigation/OVMM before promotion. The preceding 16-frame scan is **not** evidence for this newly extracted physics. |
| `4c08782d`: `navigation_result.py`, `test_arrival_monitor.py` | Inner-approach progress represented in #182. Current-main reconciliation retains main's measured-time progress window; do not overwrite it with the older source file. |
| `5d299c9d`: `voxel_arm_collision.py`, `test_navigation_import_boundary.py` | Import isolation represented in #182, with the additional grid-coordinate isolation repair on the reconciled branch. No old whole-file replacement. |
| `fdb441a9`: `check_habitat_runtime.py`, preflight lines in `run_shared_grounding_pilot.sh` | Missing acceptance utility extracted to #185 and wired into the current `run_extracted_acceptance.sh` EQA phase. Retain native/import failures before model startup; do not revive the obsolete pilot driver. |
| `963d9e32`, `73ab9204`: `vlm_region_grounding.py`, `test_vlm_region_grounding.py` | Object-only localization prompt and tests represented in #181. Semantic relation verification for support selection remains a separate responsibility; do not conflate it with object localization. |
| `75bf27b5`, `d640a4b0`: `agentic/explore.py`, `test_ovmm_rby1_find_episodes.py` | Narrative/room-glue retrieval filtering represented in #184. Retention does not imply improved accuracy: repeated EQA losses still require attribution. |
| `0ecc0aa9`: `controller_lazy_graph.py`, `test_query_grounding.py` | Old `verify_query_arrival` API absent from main/current extraction. Its forward-view-before-sweep principle is represented by #184's `agentic/investigate.py` capture/aim/verification flow. Do not restore the old controller executor alongside it. |
| Seed plumbing: Habitat `cli.py`, `runner.py`, `metrics.py`; `eval_ovmm.py`, `ovmm_batch.py`; `test_cli.py`, `test_ovmm_batch_selection.py` | Already in main through the seeded evaluation changes. Preserve newer main code. |
| `test_emet_config_loader.py` | Recursive inheritance tests already in main, including additional list-inheritance coverage absent from #169. Do not roll that coverage back. |
| `dev_sets.yaml`, `run_dev_loop.sh`, `score_eqa.py`, `score_ovmm.py` | Seeded dev loop, direct interpreter invocation and explicit errored/clean scoring already in main. #185/reconciled branch adds isolated evidence directories. |
| `TODO.md`, `eqa_restoration_handoff_20260915.md`, `shared_grounding_pilot.md`, remaining historical driver edits | Preserve at source SHA and the existing inventory. Keep negative room results and small-sample caveats. Current acceptance instructions belong to #185; historical results are not transferred to new heads. |

## Explicit manipulation handoff, not a blind cherry-pick

`31d1002b` and `6a1579e4` change `grasp_object.py` and `test_grasp_handoff.py`.
Their required `solve_pregrasp` / `ensure_grounded_grasp_workspace` experimental
path is absent from current main and #179. Restoring their parent implementation
would silently reintroduce geometry-servo and manipulation assumptions outside
the reviewed stack. #179 owns any adaptation to the current planner:

1. **Swept approach clearance:** a reachable endpoint does not authorize a
   diagonal raise-and-extend motion through a counter. Preserve the source
   regression scenario: stage a measured lift before extension where appropriate,
   or use the current collision-checked path planner. Failed lift must not command
   extension. This is a requirement, not a claim of a general clearance solution.
2. **Viewing versus manipulation workspace:** test both too-near and too-far
   viewing poses; relocate only with collision-checked navigation, recheck the
   measured arrival, and reacquire target geometry. Preserve reachable low-target
   cases. Do not transplant the old fixed radial proxy as a universal robot bound.

Original source/test commits remain available for adaptation. These are deferred
requirements assigned to #179, not fixed or accepted by closing #169.

## Rejected/reverted experiment

`7650cc3e` changed close-map resolution based on grounding. `09033bfc` explicitly
reverted it. The final source tree has no such change; do not resurrect it while
extracting commits.

## Gates and closure

- Close #169 only after publishing #182's wheel extraction and #185's preflight
  and audit, and linking the two manipulation requirements to #179.
- Do not delete the source branch. No merge to main is authorized by closure.
- Keep #182/#184/#185 draft acceptance gates: unit passes are not room-task,
  EQA accuracy, manipulation, or real-robot acceptance.
