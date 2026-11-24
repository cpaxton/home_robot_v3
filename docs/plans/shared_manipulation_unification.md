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
- Advanced metadata advertisement remains deferred until the shared adapter
  implements the options. No physical hints implementation is claimed here.

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
