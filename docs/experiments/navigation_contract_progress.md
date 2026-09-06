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

## Review branch: `feat/navigation-contract`

Branched from `feat/query-driven-memory` at `9bc34ad9`. The shared protocol is now
wired through both ZMQ clients and all four server adapters. See the versioned
[bridge contract](../../src/emet_core/BRIDGE_CONTRACT.md) for exact guarantees,
bounded history, cancellation, ownership, and restart behavior.

Additional fixes found during integration:

- Generic's initial telemetry step is -1: command sequences now have an independent counter.
- Stretch simulation resolves relative goals into the controller's episode frame.
- Stretch's measured-rest check takes absolute velocities; negative motion is not rest.
- Mars preserves rejected/aborted/cancelled Nav2 results and ignores old goal callbacks.
- Mars odometry goals are stamped as odometry, never merely relabeled as map poses.
- Stretch ROS navigation service waits are bounded; cancellation requires fresh velocity feedback.
- Deploy checks protocol version, shared runtime inheritance, and implemented adapter hooks.
- The standalone robot package carries generated identical runtime sources, with a parity test.

Focused core/client/adapter-result/deploy tests: 60 passed; an additional four
client-frame/safety tests passed after updating completion assertions to receipts.
These are not real-hardware tests. ROS callback tests exercise the actual module
with future objects, without starting ROS or contacting robots.

### Live simulation checks

| Job | Mode | Result |
| --- | --- | --- |
| `20260906_192118_a629d0` | rby1/Molmo | Failed before navigation: negative initial command sequence; fixed and regression-tested |
| `20260906_192331_3ca605` | rby1/Molmo, teleport | Eight turns, nine captures, no failures; matching successful receipts |
| `20260906_192548_41bb8a` | Mars/Robocasa, yaw slew | Eight turns, nine captures, no failures; measured yaw error at numerical precision |
| `20260906_192750_730768` | Stretch/Molmo, teleport | Eight successful navigation receipts, nine captures; overall health status remains `incomplete_telemetry` (missing base-upright/target fields) |

Artifacts use `/tmp/emet-nav-contract-{smoke-v2,mars,stretch}-20260906` on the
development host. These runs used the review working tree; they are engineering
smokes, not frozen paper evaluations. Yaw slew is kinematically animated motion,
not a demonstration of accurate wheel-driven navigation. No motion tolerances
were relaxed. Heavy runs are serialized under the jobs GPU lock, CPU-safe with
one numerical-library thread. The failed launch is retained, not discarded.

### Remaining gates

1. Run the bounded S0 OVMM agentic-find regression (same 8 views, 12 rounds,
   8 navigation steps), then the existing integrated-agent/TAMP command gate.
2. Record any remaining perception/posture failures independently of transport.
   Robocasa torso collapse, camera alignment, scene-loader failures, and the
   Sourccey Hessian failure are **not** established as solved by this change.
3. Deploy and verify on reachable hardware without motion. Herman was unreachable;
   no hardware deployment or physical motion has been performed.

Do not start a full sweep or claim cross-robot navigation/learned OVMM success
from these bounded engineering checks.
