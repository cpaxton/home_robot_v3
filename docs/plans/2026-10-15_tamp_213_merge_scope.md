# PR #213: agent orchestration merge scope

PR #213 provides the bounded CHAT route from task discovery to planning and
execution, including measured outcomes and a hard dispatch stop after failed or
partial TAMP execution. Its successful live control uses privileged stationary
Stretch inputs with Caliban inference. Read-only observed discovery is an
experimental integration boundary; its real detector diagnostics do not pass
observed task acceptance.

## Merge criteria for this PR

- Preserve the public versioned JSON tools and opaque task/plan handles.
- Allow successful TAMP sequences within the configured round budget. The
  continuation prompt retains original goals and instructs rediscovery after
  execution and respect for plan-only requests. Controlled-loop tests cover those
  sequences; semantic goal interpretation/coverage remain separately unassessed.
- Failed, partial, malformed, and discovery-only outcomes stop the remainder of
  a precomputed batch and terminate dispatch for
  the user turn. Any later tool dispatch, including a second failing tool, is a
  stop-contract violation until a new explicit user turn.
- Log explicit user turns for both scripted and interactive CHAT. The independent
  failure-boundary scorer consumes those rows; stripped logs cannot establish
  the same turn-boundary evidence.
- Keep privileged live controls, held-out physical failures, observed discovery
  diagnostics, and native runtime failures visible with their original scope.
- Pass focused orchestration, geometry-alignment, scoring and CLI regressions.

Perception latency/accuracy, held-object control, navigation/exploration acceptance
and hardware readiness are follow-up gates. They are not conditions for merging
this bounded orchestration milestone. This PR does not enable observed planning
or execution, promote a default robot preset, or relax a freshness threshold.

## Review dispositions

1. **Whole-snapshot validation before filtering:** intentional. `object_filter`
   narrows the returned proposals; it does not relax evidence validation or hide
   a stale/privileged member. The provider must supply one coherent snapshot.
2. **Scene-graph geometry input:** deliberate consistency correction. Voxel mapping
   can resize depth/intrinsics before constructing `world_xyz`; segmentation uses
   the original RGB/depth shape. Passing the resized cloud with original masks is
   inconsistent. The processor now unprojects original depth with original
   intrinsics/pose. This changes scene-graph geometry inputs. The regression runs
   actual voxel mapping at half image resolution, then verifies every graph point
   against original-resolution depth projection.
3. **Second failing dispatch:** it is still unexpected after the first failure in
   the same turn. Keep the scorer's conservative semantics and test them explicitly.
4. **User-row boundaries:** the real CHAT loop logs each explicit user request.
   A two-turn regression verifies a partial first turn, a newly authorized second
   turn, and no false failure-boundary violation when scoring those actual logs.
5. **Region argument errors:** validate both finite increasing XY regions before
   model/simulator startup. CLI input errors use argparse exit code 2 without a
   traceback. Direct Python episode callers still receive a ValueError.

Validation on October 15: 84 targeted CPU tests passed across native-agent CLI
inputs, CHAT continuation and logging, batch dispatch boundaries, independent
scoring, resized voxel mapping, scene-graph geometry and detector contracts. Ruff and diff checks passed. This
follow-up did not rerun or rescore physical episodes. GitHub reported no check
runs on the PR at the time of inspection; local test evidence is recorded here.

## Incoming stack boundaries

Open PRs inspected October 15:

| PR | Workstream | Relationship to #213 |
| --- | --- | --- |
| [#181](https://github.com/cpaxton/home_robot_v3/pull/181) | Qwen depth and SAM grounding providers | Candidate source for observed grounding; reconcile through the provider contract after review. |
| [#182](https://github.com/cpaxton/home_robot_v3/pull/182) | Measured navigation and map-clearance contracts | Required before claiming navigation-assisted TAMP acceptance. |
| [#184](https://github.com/cpaxton/home_robot_v3/pull/184) | Bounded inspection/recovery | Review how inspection results enter a later authorized TAMP turn. |
| [#198](https://github.com/cpaxton/home_robot_v3/pull/198) | Measured coverage scanning | Future exploration/discovery input and its partial outcomes. |
| [#199](https://github.com/cpaxton/home_robot_v3/pull/199) | Assessed camera evidence/context | Future source of observation provenance; do not turn historical retrieval into fresh execution evidence. |
| [#202](https://github.com/cpaxton/home_robot_v3/pull/202) | Auditable view selection | Future observation-acquisition policy. |
| [#214](https://github.com/cpaxton/home_robot_v3/pull/214) | Reactive RL manipulation | Owns low-level grasp/control improvement; reuse frozen physical regression cases. |

These are integration points, not newly imposed merge dependencies for #213.
Keep its draft status until orchestration review is complete; observed detector
success is not the criterion for removing draft status. Continue future observed
perception work with the owners of the relevant incoming providers, then validate
camera-grounded discovery before observed motion.
