# Navigation and exploration rework

## Contract and scope

One harness for exploration, find, EQA and manipulation: select a useful viewing
pose, validate and execute its route, verify measured arrival, then score the
fresh observation separately. Keep A* and the existing server ArrivalMonitor.
Acceptance requires Stretch and RBY1 simulation. No hardware or full sweeps;
heavy jobs run serially with CPU-safe/GPU-exclusive scheduling.

Baseline: `faebea0f`, including September 27 navigation-only pilot evidence in
[floor coverage recovery](../experiments/floor_coverage_recovery.md). Find verified
a fresh tomato view after an approximately 27 cm approach; exploration produced
a turn-only route and correctly failed for insufficient translation. Neither
establishes manipulation acceptance. Freeze model, perception, physics and task
budgets for paired comparisons. The sibling physical-TAMP work owns grasp/carry
repairs; selectively integrate reviewed navigation fixes, never edit its worktree.

## Reviewable implementation sequence

1. **Geometry.** Use planning-frame poses for ranking. Match physical-map cell
   indexing to nearest-center insertion/footprint geometry, preserve continuous
   endpoints including same-cell short moves, and refuse off-map snapping.
   Keep legacy map conventions unchanged pending separate migration. Record
   requested/resolved endpoints and snapping reasons. Tests must reproduce the
   recorded start (-1.0015765,-0.2790600) and goal (-1.1000000,-0.3000000).
2. **Contract.** Extend existing plan/attempt results with requested and effective
   XY/yaw, optional head pose, purpose, map revision, measured arrival errors,
   settling and terminal reason. Replace NaN/object-XYZ trajectory tails with
   explicit target metadata; temporary compatibility parsing belongs at one
   boundary. Preserve server-owned exploration (0.07 m/0.15 rad) and precision
   (0.02 m/0.03 rad) policies, freshness and bounded corrections. Distinguish
   unknown clearance from the 10 m fallback sentinel.
3. **Viewing-pose selection.** Retain the bounded eight-candidate frontier set.
   Resolve feasible poses before deduplication/ranking. Use shared search for
   all reachable costs, rather than first-goal selection. Estimate unknown area
   visible within camera FOV with map occlusion, then rank by gain divided by
   path length + footprint radius * absolute yaw change + one grid cell.
   Break ties by travel cost then stable order. Suppress previously unhelpful
   views until relevant evidence changes. Already-satisfied poses are explicit
   observation-only actions, never fictitious translation progress.
4. **Execution and task progress.** Validate yaw and translation, retaining the
   conservative sweep unless an adapter guarantees a specific motion profile.
   Revalidate after material map updates/deviations. Confirm stopping before
   correction; require fresh observation at measured arrival. Report motion,
   observation and task outcomes separately. Count sensor-supported newly
   observed area, not padded map growth. A useful stationary look is progress;
   motion without information is not automatically exploration success.

Keep behavior changes in the named physical navigation pilot until acceptance.
Do not add robot/task-specific recovery branches or lower safety thresholds.
An SE(2)/state-lattice planner is deferred: first collect independently feasible
routes that the repaired XY planner genuinely cannot represent.

## Acceptance

- Deterministic tests: world/local transforms, negative coordinates, boundaries,
  same-cell moves, snapping, resolved deduplication, zero-cost goals, heading wrap,
  blocked rotations, stale feedback, cancellations, duplicate commands and
  false-success prevention. Replay the recorded failure.
- Controller sim: each of Stretch and RBY1, three repetitions each of short
  translation, turn-only, translation plus final heading, feasible narrow
  passage, obstructed rotation, and unknown route followed by stationary sensing
  and guarded replanning. Positive cases require declared arrival/settling;
  negatives require correct safe stops. Unexpected contact, false arrival or
  motion after unconfirmed stop blocks promotion.
- Learned paired pilots: existing small Molmo exploration/find; Molmo/RoboCasa
  OVMM subsets; existing 12-question EQA dev set with two matched seeds per
  version. Exercise TAMP navigation/approach separately from carry/grasp scoring.
  Investigate each newly failing paired case; small totals are not statistical
  proof of non-regression. Simulator truth scores outcomes, not learned actions.
- Retain candidate scores/rejections, requested/planned/executed maps, pose/error
  timelines, command receipts, before/after camera views and independent safety
  measurements. Document results, figures and outstanding TODOs before promotion.

## Implementation checkpoints

- Frontier planning-frame fix: `ca09d8d8`. Geometry and continuous endpoint
  preservation: `6c97f8c3`. Requested/resolved endpoint provenance and honest
  clearance hints: `7c607d5e`. Combined regression suite: 496 passed, two existing
  SWIG warnings. Legacy padded-map indexing is unchanged.
- Geometry-only live Stretch recheck `20260928_215530_fb91dd`, frozen at
  `6c97f8c3`, is complete. It planned two waypoints instead of the old one-point
  turn-only route, but the full sweep rejected one unknown cell during rotation.
  Qwen chose stationary observe_floor (+20 observed cells; blocker unchanged),
  then diagnostics and a safe stop. No navigation/manipulation acceptance.
  Evidence: `~/runs/emet/navigation-geometry-pilot-20260928/explore/`.
- Next selection work must validate executable viewing poses/routes before
  choosing one. Current multi-goal search still selects the cheapest endpoint
  first and only then subjects that route to the swept-footprint check.
- Explicit route metadata, viewing-pose ranking, observation-based progress,
  two-robot controller battery and paired pilots remain pending.
