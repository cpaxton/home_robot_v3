# Measured release-pose and opening-path search

The checked-handoff comparison stops before the shifted cylinder's first
release. At the delivered arm pose, the nominal opening configuration puts a
fingertip 1.37–1.43 mm inside the table. Neutral passive-joint counterfactuals
also collide. Opening is correctly withheld. The wrapper now preserves the
`collision` code; release planning itself remains unresolved.

The next change should search release configurations from the measured state,
using the existing placement geometry and motion validators. Treat arm motion,
held-object volume, finger opening and retreat as one bounded search problem.
Do not assume a nominally valid planned release is valid after controller error.

1. Keep final XY/support goals and collision tolerances fixed. Declare the
   allowed release/drop-height range in the simulation experiment before
   running; do not create clearance by silently changing scored goals.
2. Generate a bounded set of nearby release poses with small height/orientation
   alternatives. Certify closed-gripper arm/payload motion, the full opening
   sweep and open-gripper retreat on private data. Reject any candidate outside
   its support/workspace or drop-height contract, even if IK succeeds.
3. Execute the selected candidate with fresh measured-state validation before
   opening. Preserve partial progress and inspect recovery on loss, stale
   geometry, tracking error or an invalid opening; never open on a failed probe.
4. Retain the independent released/support/settling/footprint checks. A successful
   command or small object height gain cannot replace them. Keep observed voxel
   unknown-space and gripper/passive-joint uncertainty contracts explicit; the
   first implementation may use privileged native geometry only.

Replay both recorded release failures offline first. Tests must demonstrate a
current-pose opening rejection with an alternative collision-free release, and
no opening command when all candidates violate collision/drop-height bounds.
Include obstacles on the arm, payload and opening/retreat volumes. Test stale
snapshots and joint-bound failures independently of collisions.

Freeze the native control/shift population prospectively, retain all prior
failed runs, and include conservative intended-failure cases. Once release is
reliable, repeat the loaded-binding versus equal-hold comparison, then expand
initial placements and run the production agent through Caliban. The robot,
model, public tool schemas and goals stay matched. Native tool references and
agent success remain separate; observed full-arm/contact-source qualification
and hardware Stretch trials are still outside the present acceptance.
