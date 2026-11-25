# Shared manipulation implementation status

Approved target: one registered manipulation interface for Codex, Qwen, CLI,
Discord and benchmark runners. Required arguments remain semantic object and
receptacle names. Optional hints express grasp/approach/placement guidance;
`prefer` reports adjustments and `strict` constrains supplied values. Protected
objects and stationary-base requirements are hard task constraints. Hints never
constitute observation evidence or authorization.

## Implemented foundation

- Canonical physical `pick_place` uses the public plan/execute handlers and
  shared observed adapter. Optional arguments dispatch through `plan_request`;
  unsupported options refuse before motion. Legacy arguments remain available.
- World-frame metric hand-pose hints and preferred approach direction/distance
  are supported by the common observed-arm factory. Discovery references are
  task-bound, revalidated against fresh measurements, capped at 120 seconds and
  discarded after motion. Strict grasp fixes the planned hand pose and refuses
  required clearance adjustments. Strict approach-path guarantees remain unsupported.
- Shared runtime visual grounding proposes object/destination regions and a
  body grasp site, closing angle and world-height fraction. Fresh RGB-D realizes
  the site; measured local material sets XY, and measured support/visible top
  set height. Robot hand geometry can raise a preferred grasp above the support,
  with requested/applied guidance and adjustments in planning receipts.
- Both supervisors use `GraspProposal`, the common support-candidate generator
  and the same physical executor. Robot bindings own hand/tool calibration,
  aperture limits, native joint mapping and controllers. No object CAD is used.
- Approximate task geometry uses a common translation/volume association policy
  for single/multiview sources. Newly visible faces need not rigidly match the
  initial partial surface. Pose observability remains explicitly unqualified;
  fresh timestamps, bounded association and observed lift/place checks remain.
- The observed-arm factory owns the phase/payload model-query gate. Inference
  requires idle, known-empty state. Dispatch failures preserve unknown load;
  verified placement restores empty. No model calls are added while carrying.
- Cartesian interpolation is shared with the diagnostic compatibility shim.
  Collision contracts accept observed voxels or bounded observed surfaces,
  require known-free workspace and invalidate plans on representation changes.
  The live map producer still uses the existing background/local-map machinery.
- The normal Panda benchmark CLI binds shared visual grounding and calls the
  canonical registry. The alternate diagnostic executor still exists and must
  not be mistaken for qualification of the shared tool.
- TAMP CI now includes grasp realization, phase, grounding and layering tests.
  The shared grasp boundary must not import evaluation/benchmark/Habitat/paper
  packages. Existing legacy controller-to-evaluation imports remain follow-up work.

## Native evidence and limits

Development controls use scripted outer tool selection, live Qwen grounding,
RGB-D/robot feedback, and explicitly reported simulation assists. They are not
frozen autonomous Codex/Qwen success-rate measurements or hardware qualification.
All failures and infrastructure exits are retained under repository-root
`logs/shared-harness-public-transfer-20261124/`.

- v23 passed observed lift verification and started carry, then lost grip.
- v27 used the VLM grasp proposal through the shared generator, passed observed
  lift verification, then lost grip during preplace. It also recorded counter contacts.
- v28's tighter joint tolerance caused settling failure; that setting was reverted.
- v29/v30 exited before the simulator bound its port; no task result was measured.
- v31 reached descent but exhausted tracking convergence with the torso near its lower stop.
- v32 used the existing fixed-torso binding, lifted approximately 84 mm and
  moved the pear approximately 39 cm into the sink. The independent benchmark
  goal checker returned true, but grip was lost before the planned release.
  The tool correctly remained partial; controlled transfer has not passed.

Detailed outcomes and earlier development history are in
[the control report](../experiments/shared_public_tool_transfer_20261124.md).
The current public metric-grasp contract is in [the API](../api/tamp.md).

## Remaining implementation, in order

1. Stabilize the native carry/placement transition and add observed task-goal
   verification/recovery after unexpected release. Preserve the distinction
   between reaching the benchmark goal and completing a controlled placement.
   Compare posture/hand-orientation alternatives using the same native binding.
2. Complete image-bound object/grasp and placement hints, protected-object task
   constraints and strict approach-path guarantees. Metric grasp support is
   implemented; all remaining unsupported options must continue to refuse.
3. Qualify shared runtime startup, live surface-map production and complete
   observed verification before removing the alternate diagnostic executor.
   Keep benchmark orchestration outside controller/perception/robot primitives.
4. Add observed native drawer motion through `set_receptacle_state`; physical
   execution currently refuses articulation. Use images/depth and robot feedback,
   never hidden articulation state or direct fixture commands.
5. Qualify the older observed Stretch task: red cylinder to green region,
   protected blue cube and stationary base. Privileged tabletop pilots are not
   shared-agent evidence.
6. Run matched Codex/Qwen trials through the real registry. Freeze 20 episodes
   per atomic family (RoboCasa transfer, drawer, Stretch transfer), targeting
   16/20 per supervisor. Then run ten categorization episodes, targeting 8/10.

Separate sustainability follow-ups: broad CPU/lint CI beyond the existing TAMP
workflow, the root MagicMock debug-path leak, legacy layering inversions and a
protocol/deprecation policy. The root checkout's Qwen/chat dependency-isolation
work is independent and has not been modified by this feature branch.
