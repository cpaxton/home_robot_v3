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

## November 6 implementation and prospective gate

The opt-in `bounded` native diagnostic in `eval_tamp_contact_stability.py`
now uses `emet.motion.release`. Production and observed execution defaults stay
unchanged. It checks the original retained payload transform and the measured
held transform without renewing the retention reference. It tries the delivered
pose first, then nominal release heights +3/+6/+10 mm. A correction requires a
collision-checked direct joint edge; at most two corrections are permitted.

The declared bounds are 40 mm maximum raise from the measured start of release
refinement, 40 mm maximum object-bottom gap above the chosen support, and an
additional opening probe lowered 3 mm along the nearly vertical lift axis.
The final full-object XY region, support and settling score is unchanged. This
reserve samples one extra pose, not an uncertainty tube. The released payload
is watched during retreat, including a sampled vertical descent to the support;
this fixed-orientation check is not a dynamics certificate. The native profile
also waits for measured open joints and separation from both fingertips before
retreat; controller acknowledgment alone does not qualify release.

Private replays of both November 3 failed cylinder releases reject their old
opening with `collision` and find a higher alternative with predicted gaps
20.878/20.895 mm. Neither replay changes qpos/qvel/qacc/contact buffers or sends
commands. The nominal target in these replays is reconstructed from the saved
last placement joint command's FK; this is replay evidence, not task success.
Raw evidence: `.tamp-runs/release-search-20261106/replay.{py,json}`.

`configs/eval/tamp_release_pose_search.json` freezes four prospective episodes:
control and +15 mm initial-X shift, each with legacy and bounded release, all
using fine yaw and the same checked loaded-payload handoff. Use two CPU-only
simulators concurrently, queue the controls after the shifted pair, keep all
failures and source/input hashes. No Caliban inference or hardware trial is
needed for this native primitive gate. Any measured retention, opening,
tracking, retreat or final-support failure remains a failed task.

The first live shifted bounded run at `d5e1df78` exhausts both corrections
without opening: measured EE z is 0.52060 m at place completion, 0.52047 m after
the first command acknowledgment, and 0.51915 m after the second. Small nominal
height increments do not overcome the loaded command/FK offset. This is retained
as a failed run; it is not a reason to waive the opening or tracking checks.

The prospectively frozen follow-up in
`configs/eval/tamp_release_tracking_reserve.json` adds the positive measured
command-FK minus delivered EE-height error to the *hypothetical* lower opening
probe for new correction targets. It samples +3/+10/+20 mm nominal alternatives
within the same 40 mm raise/drop limits and two-command budget. Both nominal and
lowered openings are checked. Actual measured-state release still has to pass;
the measured error reserve is not a bound on future tracking error. A regression
overhang demonstrates why checking only the lowered opening would be insufficient
(the shared actual-opening guard already withheld unsafe opening in that case).
The follow-up is a separate pair of bounded development trials, not a matched
rescore of the first four episodes or an agent baseline.

The offset-aware shifted run releases successfully but stops on a conservative
rotated-cylinder AABB/support intersection during retreat. The exact cylinder
support extent and MuJoCo contact depth are within 1 mm while the rotated local
box exceeds it. `configs/eval/tamp_release_shape_extents.json` freezes a third
pair with exact primitive/mesh bounds and 0.1 simulated seconds of measured
supported stillness before retreat. There is no new object/table collision
exemption; deep contacts still reject. Keep all earlier failed task receipts.
