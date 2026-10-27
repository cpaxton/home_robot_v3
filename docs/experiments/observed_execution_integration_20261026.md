# Observed execution integration, October 26

This change connects measured RGBD rigid tracking, tri-state visual effects,
robot-only native motion execution and the existing CHAT/task tools. It does not
claim live sensor-driven task success. Defaults remain read-only discovery until
a concrete qualified execution source is explicitly installed.

## Verified behavior

The real registry plans and executes a transfer on a MuJoCo robot-only rig through
`PhysicalPickPlaceExecutor`. A separate synthetic sensor/controller fixture
produces object observations and measured joints. Planning sends no commands;
execution records visual grasp-reference, co-motion/separation and post-retreat
support/settling evidence. No simulator object body or privileged RPC exists on
the runtime robot interface. The fixture uses an ideal hold/release model, not
MuJoCo contact dynamics; this is orchestration and verifier evidence only.

A second integration case calls the real `plan_task` and `execute_task_plan`
tools for a buffered swap. The shared task planner finds three transfers and the
same observed adapter executes and verifies each. This is one regression scenario,
not a new LLM or observed-contact benchmark score.

Failure regressions include missing initial evidence, held/unsupported initial
objects, empty grasp, missing visual evidence after commands, stale/replayed or
mismatched captures, unbounded camera/encoder alignment, full gripper volume still
near a released object, changed scene/model state, occupied/unknown destinations,
outside/unsupported/unsettled placement and external-motion invalidation. Known
pre-command failures return no-actuation receipts; possible physical progress is
partial and stops further dispatch. Placement requires a verified lift reference.

Tracking regressions include paired 3D rigid translation/rotation with outliers,
actual image-pixel translation through optical flow, collinear/tiny patches,
incorrect correspondences, textureless and repetitive surfaces, reference age and
failed-update cache clearing. The reference is fixed, not recursively renewed.
The tracker requires current instance masks and an independently supplied complete
geometry prior; these tests do not validate a live mask producer or calibrate the
error bounds used for hardware admission.

Wall-clock gripper settling no longer requires `MjData.time` to advance. Explicit
simulation-time execution retains its stalled-clock guard. No contact/effort
sensor is introduced.

## Validation

Committed implementation **`c532a758`**, managed CPU job
**`20261026_205621_349c68`**: **610 passed**, zero failures, errors or
skips; 126.160 s test time. Coverage includes all motion and TAMP tests, native
input isolation, real CHAT startup/tool/failure boundaries, observed discovery,
live capture binding and skill packs. `MUJOCO_GL=disable`; no local GPU model,
renderer or new VLM calls. Ruff on changed Python files and `git diff --check`
pass. Raw JUnit: `.tamp-runs/observed-execution-20261026/committed-regression.xml`; log:
`.tamp-runs/observed-execution-20261026/committed-job/job.log`.
The 52-page paper builds with no unresolved references/citations. Machine-readable
results and source hashes are in [the evidence record](observed_execution_integration_20261026.json).

Earlier overlapping development suites passed 606, 608 and 609 checks as scene,
receipt and reconnect guards were added; do not sum these or count them as agent
episodes. Final numbers above use the committed source and include all guards.

## Remaining work before observed-task numbers

The API now accepts an explicit observed execution source; the repository still
needs the actual live provider assembly and qualification: current instance masks,
complete payload bounds, camera/encoder alignment and inter-sample motion bounds,
robot/target occupancy provenance, multi-view geometry refresh, and performance
under occlusion and long robot motions. A declared prior or a calibration ID is
not evidence that those properties have been validated. Symmetric/untextured
objects remain unknown for the current feature-based tracker.

Keep privileged native controls separate. Next run held-out stationary Stretch
transfer, swap and failure cases through the same CHAT interface with Caliban,
preserving every attempt. Navigation/exploration and hardware sensor availability
remain separate gates. See the [provider contract](../api/observed_native_tamp.md).
