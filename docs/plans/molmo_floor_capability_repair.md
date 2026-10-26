# Molmo floor-recovery capability repair

The 2026-10-26 integration trace reduced unknown footprint cells from 18 to 7,
then could not observe the rest. No base motion executed. Floor recovery used
a hard-coded -1.4 rad lower tilt bound even though the robot's existing active
MuJoCo capability advertised -1.8326 rad. The rby1 adapter is a Galaxea proxy;
this is not native RB-Y1 or hardware qualification.

A CPU kinematic probe aligned the model camera at the recorded measured capture
and swept the two gaze joints. None of the seven cells projected into the image
within the old inspection envelope. A joint-range pose near pan=-0.51,
tilt=-1.8326 projected all seven into frame. This is a reachability hypothesis,
not occlusion, collision, stability, measured floor, or route clearance.

The generic repair uses the existing advertised head capability for the lower
inspection tilt bound, intersects pan with the existing ±1 rad inspection band,
and keeps the downward upper bound -0.7 rad. Unavailable capability retains the
legacy envelope. A robot unable to reach that downward band reports unavailable.
Requests outside the resulting bounds are rejected, not silently interpreted as
successful inspection. The tool schema defers the lower bound to the robot and
the result reports the actual inspection bounds.

Capture still requires measured head arrival, fresh calibrated RGB-D and an
actual map update. A measured base XY/yaw change beyond existing navigation
tolerances rejects the stationary observation before mapping. Base XY/yaw
stability does not establish full-body or hardware stability. No footprint
padding, unknown-floor exemption, visited-disk growth or obstacle removal is
introduced. Replanning and full swept-footprint checks remain mandatory.

387 targeted floor/gaze/adapter/integration-driver checks pass, including a
measured -1.8 rad capability case, legacy/narrow/no-downward capabilities and
base drift rejection. Live Molmo and broader integration revalidation are still
required before claiming this fixes the task or qualifies the expanded pose.
