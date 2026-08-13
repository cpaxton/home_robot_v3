# Agent-call-wrapping MCTS for mobile pick & place

**Branch:** `feat/sourccey-manip-planning` (adjacent)
**Date:** 2026-08-13
**Status:** Design sketch — no code yet.

## Goal

A *simple* MCTS planner that **wraps agent tool calls** as its expansion policy, so we can
do TAMP-style *mobile* pick-and-place without a hardcoded PDDL domain. The "world model"
for rollouts is our existing MuJoCo sim (via the same executor the agent already drives);
the "GPU" angle is batch-rolling candidate action sequences in parallel instead of
executing one greedy plan.

Not a substitute for the LLM agent — a *planner layer in front of it*: when the task is
"pick the apple from the counter and put it in the basket", the agent still proposes
`pick_place(...)`, but MCTS explores *which* preconditions/order to try, powered by sim.

## Why now

- `ArmManipProfile` discovery + E2E IK/gripper-contact tests (this branch) give us a
  cheap, robot-agnostic arm check: **can the EE/gripper reach that pose?** That is the
  manipulability predicate TAMP needs.
- `KinematicPickPlaceExecutor` + the agent tool pack already provide the *action set*
  (`pickup`, `place`, `navigate`, `look_at`, ...) and the *sim step* — no new actuators.
- Goal-conditioned re-planning exists ad hoc in the agent loop; MCTS formalizes it.

## Design

```
Task (goal spec, e.g. "apple -> basket") + scene state (voxel map, object poses)
                │
                ▼
      ┌──────────────────────┐   propose K candidate (tool,args) next actions
      │  Agent policy call   │◄────────── LLM/VLM, one batched prompt per node
      └──────────┬───────────┘
                 ▼
      ┌──────────────────────┐
      │   MCTS search node   │  state = {robot pose, object poses, graph belief}
      │ select→expand→rollout│  rollout = batch MuJoCo sim of K branches
      └──────────┬───────────┘
                 ▼
      ┌──────────────────────┐
      │   Batch sim engine   │  mjx / MjSimulate-batch, or threads on CPU fallback
      └──────────┬───────────┘
                 ▼
      best action sequence → agent executes top move for real
```

### Key pieces

1. **Search state**: the same info the agent already has — `AgentState` + object/pose
   belief + robot pose. Cheap snapshots (dicts), not a new world model.
2. **Expansion = one agent call.** At a node, prompt the LLM with current state + goal,
   *asking for K candidate next tool calls* (not one). The LLM is the policy prior;
   MCTS does the lookahead.
3. **Simulation = existing executor.** Run a candidate `(command,args)` through
   `DynamemTaskExecutor`/`KinematicPickPlaceExecutor` against the MuJoCo server for a few
   steps. Score terminal states with a small task-reward (object moved? gripper in contact?
   reachable? collision?).
4. **Selection/backprop**: UCT on sim scores. Because expansions are expensive (LLM calls),
   keep depth small (≤4–6) and breadth modest (K=4–8); amortize by batching the K branches
   per node in one sim batch.
5. **Move execution**: after search, execute the best child's action on the real robot /
   live sim via the normal agent tool path, then re-search (closed-loop).

### GPU / batch sim note

- Fastest path is `mujoco.mjx` (JAX, GPU batch) or `MjSimulate`-style batched stepping.
  Our vendored mujoco 3.5.0 wheel currently has **no `rollout` / `mj_simulate` / `mjx`**
  (checked 2026-08-13), so batching needs one of:
  - install a wheel exposing `mujoco.rollout` (CPU threads) — smallest change;
  - add `mujoco-mjx` + JAX for true GPU batch — bigger install, needs a no-conflict
    env (numpy pin already overridden for robocasa);
  - or for a *first cut*, run branches sequentially in a subprocess and parallelize
    across 2–3 processes (no new deps).
- The manip predicate (reachability/contact) is a *single* MuJoCo forward+IK — already
  cheap; the expensive part is only scene-wide object interactions.

## What "wrapping agent calls" means concretely

- New module `src/emet/motion/agent_mcts.py` with a tiny `MCTSNode` + `search()`.
- A policy shim `propose_candidates(state, goal, K) -> list[(tool, args), score_hint]`
  that calls the existing LLM client (`_call_llm` style) with a "propose K next actions"
  system prompt.
- A rollout hook `simulate(candidate, state) -> (next_state, reward)` that reuses the
  executor + MuJoCo server already used by `scripted_sim_pick_place.py`.
- Tests: reuse the `test_arm_manip_profile.py` reachability predicate to unit-test
  candidate scoring; keep full search out of CI (LLM + sim cost).

## Open questions

- Reward for "moved object toward goal" without a sim oracle — use gt placement readback
  (`read_sim_object_placements`) for sim rollouts; real-robot rollouts only measure
  reachability/contact, not object motion.
- How many LLM calls per search are acceptable (1/node = depth×breadth)? Budget a hard cap.
- Interaction with `attempt_ledger` — failed branches should seed the ledger so real
  execution doesn't repeat them.

## Status

- [x] `agent_mcts.py` skeleton + UCT search with a **distance-based heuristic policy + sampling** (`PickPlaceDistancePolicy`), unit tests in `test_agent_mcts.py` — no LLM yet
- [x] **MolmoSpaces scene-task extractor** (`src/emet/eval/scene_task_extractor.py`): loads `*_physics_metadata.json`, enumerates pickable objects (grasp assets), receptacle sites, emits `full_episodes.yaml`-schema tasks, and computes per-object gripper-contact reachability via `ArmManipProfile` + IK. Tests in `test_scene_task_extractor.py`.
- [x] **Exposed as a CHAT agent tool** `scene_tasks(object_filter, robot)` — the agent can now ask "what can I pick up and where can I put it" and get a digest with per-robot reachability (from live sim placements when connected).
- [x] **Verified against a live MolmoSpaces iTHOR sim** (sourccey, index 0, `emet jobs` job `20260813_191015_ff4452`, port-offset 60):
  - `scene_tasks` reads real `sim_object_placements` (124 bodies) from the connected server.
  - `scripted_sim_pick_place.py --manip-mode teleport --object bowl --receptacle microwave` succeeds live (1.99 m displacement, GT placement readback).
  - Reachability reports `none` from the scene origin — correct, since every pickable object is meters away; the TAMP loop must navigate before the arm predicate applies.
- [x] **Live MCTS TAMP on rby1 (kinematic, port-offset 70)**: `plan_pick_place_mcts` searched 12 scene candidates with `PickPlaceDistancePolicy`, chose bowl→drawer, IK-ranked grasps (rejected infeasible decoy grasp err=3.51, picked reachable grasp err=0.042), navigated to approach standoff, and streamed RRT arm paths. `scene_tasks` now reports real reachability (Apple/ButterKnife/Knife/Tomato) from live `sim_object_placements`. Execution reached `attach_verify_failed` on the synthetic COM grasp — a physics/execution detail, not a planning failure. Driver: `scripts/scripted_mcts_pick_place.py`.
- [ ] Policy shim `propose_candidates` via existing LLM client
- [ ] Batch sim hook (threads first; mjx later if env permits)
- [ ] Wire top-level move execution back into agent loop
- [ ] E2E smoke on sourccey/rby1 in MolmoSpaces (GPU eval, via `emet jobs`)
