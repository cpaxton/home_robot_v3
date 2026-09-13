# Agent task benchmark: quality before scale

Implementation starts from `e8564ac6` on `feat/agent-task-preflight`, in an isolated
worktree. The old Sourccey branch was squash-merged as PR #160; its tree is identical
to `4ed75850`. Existing `home_robot_v3` worktrees are owned by the active agent effort.

## First release

1. CPU configuration, geometry, scoring, trace-integrity and CLI checks.
2. One existing live rby1 CHAT integration control.
3. Three repeatable multi-room **assisted** fixture cases, each reset/replayed three times.
4. Visual review of success and negative controls; offline HTML/Rerun/video/figure export.
5. Shared-agent/learned-memory acceptance after the other effort's interfaces are ready.

The three cases are cross-room delivery, two-object collection, and a moved-object
revisit. The initial scene is a deterministic three-room MuJoCo fixture: only
single-room iTHOR assets were installed when development began. This fixture
validates the benchmark machinery without downloading a new environment corpus.
It is not ProcTHOR/BEHAVIOR, a realistic household benchmark, or an official task.

## Ownership and interfaces

This effort owns scene/task definitions, evaluator predicates, researcher tools,
evidence recording, certification and paper exports. The active agent effort owns
perception, grounding, navigation and the semantic-memory lifecycle. This branch
adds an opt-in bounded task mode around the shared loop; integration back into
the active agent worktrees should reconcile that small loop change.

Use the existing CHAT `plan_pick_place` / `execute_pick_place_plan` registry and
one-shot plan handles. Never interpret absence of the word "fail" as success.
The scorer independently checks measured object positions and held state.

The bounded task execution entry is now `emet.agent.task.run_agent_task`, shared
with `run_agent.py --task-mode` and the benchmark runner. It consumes action outcomes,
continues beyond three decisions and reports model finish, cancellation, timeout,
protocol failure and budget exhaustion. `--task-suite` launches fixtures directly
from the app. Actual local-model diagnostics are separate from assisted witnesses.

Record immutable event-time `policy` state separately from private `evaluator`
state. Attach images and observation/command IDs. Learned integrations must supply
their actual memory snapshots and model-input observation IDs; empty fields stay
empty. Do not derive an agent graph from evaluator coordinates.

Background and agent-triggered memory updates must share the upstream lifecycle:
grounding, provenance, stale-location invalidation, and evidence-backed revision.
Missing observation is not proof of absence. Room membership must respect walls.

## Boundaries and later stages

The fixture uses rby1 geometry, footprint-checked kinematic routes, synthetic grasps
and object-pose teleportation. No contact dynamics or IK success is claimed.
Its head camera has a recorded fixture-only downward mount; robot assets are not
changed. A witness certificate applies only to this assistance profile.

Implemented: visible-pixel-gated observations, historical object-cache revision,
three/four-room layouts and bounded local-model task runs. Remaining: multi-room
live ZMQ integration, learned room/object discovery and the actual DynaGraph memory lifecycle. Then curate certified ProcTHOR homes,
add suitable robots, freeze development/held-out splits, and eventually evaluate
a small native OmniGibson/BDDL placement pilot.

The hosted-model pilot is documented in
[agent_hosted_pilot.md](../experiments/agent_hosted_pilot.md) and is **disabled**.
No paid calls, model sweep, hardware motion or edits to the active agent's worktrees
are part of this preflight.
