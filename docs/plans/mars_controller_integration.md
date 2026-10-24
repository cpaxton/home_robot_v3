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

An additional opt-in simulator research track is accepted in
[tracking and held-object training](retention_training.md): tracking-majority
sampling, randomized held payloads and an advisory occupancy estimator. Small
training diagnostics on that distribution do not claim public TAMP readiness or
replace the source-bound canonical supervisor below. Keep their reports separate.

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

- The immutable `b181db61` placement recovery completed 298/300 carry and 30/30
  continuous pick/place, with zero unsafe contact/wheel events. The two remaining
  100 g orientation failures remain in the report. Canonical readiness and the
  367-test source-bound receipt were verified for that snapshot only; PPO did
  not restart. See `paper/data/whole_body_control/qualification_v6.json`.
- Simulator task-pose protocol and measured-Jacobian teacher pass component
  replay tests. Public TAMP now has explicit simulator Mars routing, private
  planning state and measured physical execution; this does not attach the
  public adapter to a deployed ZMQ server or hardware driver.
- The frozen two-support v2 catalog has 30/30 fixture admissions. Its v1 catalog
  is retained: five right-layout 100 g cases lacked a valid arm-only witness;
  v2 supplies a whole-body FK witness for those same targets. Fixture admission
  does not establish executable transport or successful release.
- Public execution traces expose loaded-turn retention failures. Keep this gate
  open until all 30 full tasks pass. Source hashes bind the frozen catalog,
  shared safety/control modules, public adapter and agent execution dependencies.
- A revised executor completed the frozen left-layout 20 g diagnostic through
  grasp, 11.8 cm base transfer, support-confirmed release and stable placement;
  peak relative rotation was 0.013 rad and peak load 3.24 N, without wheel-lift
  termination. This selected case took about 100 simulated seconds and is not
  full task or speed qualification. The 100 g diagnostic still fails retention.
- Final wheel slew and measured path retiming are explicit public-facade modes,
  preserving the legacy default. A bounded 6 mm / 3 s contact approach confirms
  measured support before release. Fixed-time execution and new learned-policy
  qualification remain separate requirements.
- The normal prompt/parser/dispatcher agent harness is implemented, with
  oracle-first admission and three local seeds per task. Observed inputs are
  currently unavailable; this blocks normal-observation trials before model
  calls. No actual LLM task trial has been qualified.
- Guarded driver interface is loopback-tested and disabled by default. No
  physical driver is attached and no Herman motion has been sent.
- Passive SSH reached Herman on 2026-10-23; ROS Humble is installed. Only
  `/parameter_events` and `/rosout` appeared in the default ROS domain. This
  does not establish absence of drivers in other domains/environments, nor
  verify any real actuator mode, watchdog, load limit or stop behavior.
- Two-support public-tool acceptance and subsequent agent trials remain open;
  do not mark them complete based on the component smoke or 30-case return-place.
- The complete `9dee06fc` public transfer run improved from 4/30 to **16/30**:
  10/10 at 20 g, 6/10 at 50 g, 0/10 at 100 g. All thirty fixtures admitted;
  five wheel-unloading stops, six retention losses, one base-drift stop and two
  bounded-planning failures remain. The exact source passed **588 tests**.
  Successful tasks took a median 67.55 simulated seconds. See the
  [complete report](../experiments/mars_public_tamp_20261023.md).
- Nearby hand-only FK-witness tests now use the free base for both Mars and
  Stretch. These selected checks establish interface behavior, not full task
  or hardware acceptance.
- The [passive hardware audit](../experiments/herman_feedback_audit_20261023.md)
  found upstream velocity-unit and fabricated-zero feedback issues. A proposed
  patch passed offline C++ checks on temporary copies; it has not been applied
  upstream, built as a ROS package, deployed or exercised on hardware.
- The 2026-10-24 tracking-majority research track completed 10,240 additional
  PPO smoke steps and 603 source-bound regression tests. Its gripper-local
  occupancy head passed a selected twelve-attempt new-seed diagnostic, while
  full task success has not improved or qualified. Keep these experiments
  separate from canonical carry/public-tool promotion. See the
  [training report](../experiments/mars_retention_training_20261024.md).
