# Shared manipulation implementation status

Approved target: one registered manipulation interface for Codex, Qwen, CLI,
Discord and benchmark runners. Required arguments remain semantic object and
receptacle names. Optional hints express grasp/approach/placement guidance;
`prefer` reports adjustments and `strict` constrains supplied values. Protected
objects and stationary-base requirements are hard task constraints. Hints never
constitute observation evidence or authorization.

## Completed foundation

- A shared request boundary validates semantic selectors, hint mode, option
  groups, protected-object/base constraints and finite JSON values. It owns
  copies of caller inputs.
- Physical one-call `pick_place` delegates through the public plan and execute
  handlers, retaining plan revalidation and native input gates.
- Both planning and one-call handlers accept optional request arguments. They
  dispatch them through an adapter's `plan_request` method when available.
  Unsupported options explicitly refuse before motion; they never fall through
  to legacy execution. Empty optional dictionaries preserve names-only behavior.
- The shared observed-arm factory now applies capture-bound world-frame approach
  direction/distance guidance to transfer candidates before the existing route
  certification. Direction-only and distance-only hints retain the unspecified
  native value. These are preferred pregrasp targets; strict approach refuses
  until a path-direction guarantee is implemented. Discovery references are
  task-bound, revalidated against fresh captures, capped at 120 seconds and
  discarded after motion. Other hint groups and protected-object constraints still refuse
  explicitly. No live qualification or general hinted grasp support is claimed.
- Advanced metadata advertisement remains deferred until the shared adapter
  implements the options. No physical hints implementation is claimed here.

- Shared Cartesian motion and the diagnostic now use one pose-waypoint generator.
  The legacy executor's sampling is preserved and compared against its prior SO(3)
  interpolation in a regression test. Evaluation imports remain compatibility
  shims; no second interpolation implementation remains.
- Observed scene contracts now accept bounded surface geometry in addition to
  voxels, through the same collision checker and physical executor. Both require
  known-free workspace evidence. A source representation change invalidates a
  stored plan before motion; ground truth remains excluded from observed mode.
  Public-tool integration tests exercise both representations with synthetic
  evidence. This does not implement the live surface provider or remove remaining
  payload tracking requirements.

- Visual grounding now lives in shared perception, with compatibility imports
  for old evaluation callers. The shared live scene can consume model proposals,
  acquire fresh RGB-D after inference, revalidate view/object association and
  build its support geometry from the new capture. An explicit phase callback
  must permit model queries; known loaded/unknown phases can refuse them.
- Shared runtime grounding asks for object and destination regions only. It
  infers support locally and imposes no universal lower-destination assumption,
  supporting below-source, same-table and higher surfaces. Legacy sink profiles
  retain their explicit lower-destination requirement. Retry feedback identifies
  object/source/destination failures without changing the public reason code.
- Read-only native Panda/Qwen smoke admitted fresh geometry in 3.7 seconds,
  with no actor command calls. This tests live grounding, not full agent task
  success. Default application startup still needs the phase-owned grounder
  binding before the old diagnostic route can be removed.

## Remaining implementation, in order

1. Implement the request in the shared observed adapter. Define observation/frame
   binding, field validation, hint resolution/adjustment receipts, hard task
   constraints and strict-mode plan persistence. Preserve legacy plan arguments.
2. Integrate the tested diagnostic geometry and native motion behavior into the
   shared executor; accept approximate measured geometry without mandatory exact
   rigid-payload tracking or voxel scene representation. Do not wrap the Panda
   diagnostic as another execution backend.
3. Make the diagnostic runner a normal tool client. Keep robot calibration,
   kinematics, gripper configuration and controllers in robot bindings. Add the
   request options to canonical tool metadata only with tested adapter support.
4. Add native observed drawer motion through `set_receptacle_state`; current
   physical execution still refuses articulation. Use images/depth and declared
   robot feedback, never hidden articulation state or direct fixture commands.
5. Qualify the older observed Stretch table transfer: red cylinder to green
   region, protected blue cube and stationary base. Do not use the privileged
   legacy tabletop pilot as agent evidence.
6. Run matched Codex/Qwen tests through the real registry. Retain the familiar
   six-transfer comparison, then freeze 20 episodes per atomic family (RoboCasa
   transfer, drawer opening, Stretch table transfer), targeting 16/20 for each
   supervisor. Test categorization composition on ten episodes, targeting 8/10.

Keep all infrastructure failures, refusals, physical retries, observed
verification and independent official scores distinct. Existing diagnostic
successes do not qualify shared-agent execution. No live qualification runs
have been performed for this refactor yet.
