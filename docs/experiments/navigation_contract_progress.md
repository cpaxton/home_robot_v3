# Navigation contract implementation progress

## Verified first slice (2026-09-06)

Navigation retries no longer bypass duplicate-step suppression, including
packets that also contain posture/head commands. Both clients enforce
acknowledgement deadlines and keep image-stream acknowledgement counters
monotonic. Timeout reports an unknown outcome, not a successful motion.

The repeat rby1/iTHOR probe `20260906_185618_d98e41` completed eight turns.
Measured headings in radians were:

```text
0, 0.785, 1.571, 2.356, 3.142, -2.356, -1.571, -0.785, 0
```

Before this fix, the equivalent row included 90- and 180-degree increments
for requested 45-degree turns. No motion tolerances changed. Camera height
remained 1.734–1.753 m in the rerun. This is a Molmo teleport-mode result,
not proof of accurate smooth or real-robot navigation.

Artifacts: `/tmp/emet-nav-retry-fixed-20260906/rby1-molmo`.
The run launched after `d43c6dd9`; subsequent Stretch-only and mixed-packet
hardening did not change its plain rby1 navigation packet semantics.

## Remaining implementation — not yet deployed

`CommandTracker` provides boot/session identity, bounded receipts, immutable
terminal outcomes, busy rejection, and deadline inspection. Its tests pass,
but it is not yet connected to client/server transport or adapter execution.
Do not advertise protocol v2 support from this foundation alone.

Required next work:

- Wire tracker receipts and version negotiation through both clients and all servers.
- Replace cached `at_goal` completion with command-specific adapter outcomes.
- Connect cancellation/deadlines to simulator and ROS controllers; distinguish
  cancellation request from confirmed stop, and handle reconnect/server restart.
- Add strict deploy compatibility checks and test bridge upgrades without motion.
- Rerun smooth-motion rows, then isolate residual Robocasa collapse and scene faults.

No hardware deployment or physical motion was performed. The existing
step-based retry fix is not the complete boot-scoped navigation contract.
