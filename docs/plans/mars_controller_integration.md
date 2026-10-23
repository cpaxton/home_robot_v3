# Mars controller integration and acceptance

Accepted order: placement recovery → public TAMP execution → agent acceptance →
Herman. Teacher first; learned checkpoints are a separate promotion gate. No
hardware motion is authorized by this document.

## Placement gate

Preserve the v5 20 g seed 1860001 retention failure and 50 g seeds
1860007/1860010 tracking timeouts. Diagnose with matched ablations; do not relax
geometry, goals, timeouts, 0.1 rad retention or existing pose/force/wheel contracts.
Pass the 300-case canonical carry suite, 30/30 continuous pick/place, zero unsafe
contact/wheel events and fresh source-bound regression tests before PPO.

The first ablation restored base compensation across rearm and recovered all
three cases; preserving tracking trim alone recovered only seed 1860007. The
candidate executor explicitly authorizes compensation after verified transport
and before descent, then disables it at support contact. Fault stops still clear
compensation. These three cases are development-selected diagnostics, not a new
held-out qualification result.

## Controller and public TAMP gate

Use masked world-frame base SE(2), one hand SE(3), coupled gripper targets and
relative-time schedules. Preserve command identity, ownership, deadlines,
update/cancel semantics, measured-stop confirmation and executor-only release.
FK/IK belongs in planning; runtime consumes poses and timestamps. Start in a
simulator loopback using the real command protocol before exposing hardware.

Retain the stationary suite. Freeze a separate two-support transfer suite with
two layouts, five seeds and three masses (30 cases), including coordinated base
motion. Count planning and admission failures separately, keep every identity,
and require all 30 oracle tasks to pass unchanged contracts before claiming
integrated public-tool Mars readiness. Component arm-path and one-pedestal tests
are not substitutes for this gate.

## Agent and learned-policy gate

Run GT oracle plans through public tools first; then three fixed-seed agent
trials per frozen task with normal observations. Record observed grasp,
retention, support and recovery, not just tool return status. Exercise retargets,
cancel, stale state, disconnect and partial execution. Keep oracle, teacher,
agent and learned-policy scores separate. Publish task success and failure
stages; interface acceptance requires no silent verification failures.

After simulator gates pass, run one controlled PPO experiment with scalar-only
W&B logging and the canonical readiness/regression supervisor. Compare against
the teacher on identical cases; no automatic promotion or retries.

## Herman gate

First audit feedback, units, actual control modes, speed/effort limits, timestamp
freshness, whole-body stopping and robot-side watchdogs. A local source audit of
`innate-os/ros2_ws/src/maurice_bot` finds arm joints configured with
`control_mode: 3` and position profiles; Nav2's velocity smoother declares a
0.2 s timeout. Neither establishes an independent arm watchdog, a whole-body
velocity executor, measured stop confirmation, calibrated external force
limiting, or the configuration actually running on Herman.

Before every command, enforce the existing latched boundary from one captured
origin: starting XY target, 5 cm measured translation hard boundary; yaw target
±45°, hard boundary ±47°; odometry age at most 200 ms. A violation requests a
bounded-time wheel-and-arm stop, reports confirmation and terminates nonzero.
Validate this path and watchdog behavior in loopback before separately
authorized supervised free-space hand tracking. Target loss stops tracking;
unreachable targets are rejected without chasing them. Gentle contact requires
a separate calibrated load-limiting procedure. Real-object pick/place follows
later. Tracking error is diagnostic, not a force certificate.

## Evidence

At each gate archive exact source/model fingerprints, source-bound regression
receipts, complete case denominators, failures and measured traces. Keep earlier
qualification artifacts immutable. Update the runbook, root RL TODO and paper
claims from measured outcomes. Mars first, with existing Stretch cases retained.

## Implementation checkpoint

- Candidate placement fix passed all 30 frozen continuous cases with zero unsafe
  contact/wheel events; source-bound full carry rerun remains pending.
- 367 fresh simulator/core/interface tests pass with no failures or skips.
- Simulator task-pose protocol and measured-Jacobian teacher pass component
  replay tests; the native public TAMP adapter remains Stretch-only.
- Guarded driver interface is loopback-tested and disabled by default. No
  physical driver is attached and no Herman motion has been sent.
- Two-support public-tool acceptance and subsequent agent trials remain open;
  do not mark them complete based on the component smoke or 30-case return-place.
