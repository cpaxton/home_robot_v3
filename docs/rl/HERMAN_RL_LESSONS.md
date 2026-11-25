# Herman RL lessons for Sourcey

This is the entry point for reusing Herman's experiments. See the
[hardware handoff](HERMAN_PICKUP_HANDOFF.md) for bring-up and
the [training ledger](TRAINING_MIXTURE.md) for dataset provenance.

## Controller behavior to carry over

- Use fresh measured hand-to-target error to update the reach. A small
  joint-angle miss is acceptable when the jaws reach the grasp region and
  the arm settles. Verify retention on a small lift before declaring success.
- Treat a stalled joint as reduced control authority. The controller can
  hold the joint at its measured angle and replan with the joints that
  track. An unreachable grasp should lead to a new approach or a confirmed
  hold. Record the proposed target, measured progress and stop receipt.
- Preserve a powered hold through controller handoffs. An elbow stall
  occurred with and without a gain transition. Recovery should verify
  healthy power, the measured hold and the active profile before moving.
- Include infeasible commands in evaluation. Exercise a self-collision
  constraint, an environmental obstruction and a tracking stall as
  separate cases. Score whether the controller makes safe progress or
  stops. For contact adaptation, measure peak force and response time
  with calibrated feedback. Servo load proxies need calibration before
  use as force targets.
- Transfer examples with their body frame, timing and task semantics.
  Recompute reach and dynamics for the receiving robot. Keep the source
  geometry, actuator assumptions, and outcome with each trace.

## Hardware and policy qualification

Herman has demonstrated a supported hold, wrist motion and one measured
reach step. The live safe-start adapter and feedback-driven pickup have
yet to be qualified. The repeated elbow stall remains a hardware
readiness issue.

For a reliable run, verify a sustained powered hold through a gain
handoff, repeated bounded pregrasp and stop trials, then a grasp with
retention. Hold the base still for the first pickup. Verify exclusive WBC
ownership and the wired travel limit before testing coupled motion.

Evaluate learned policies against the analytic controller on the same
held-out cases and seeds. Include restart and hold, tracking tolerance,
joint stall, infeasible goals, coupled motion, and grasp retention.
Require measured stop confirmation and no increase in hard contact or
wheel-lift failures. Record success, peak load and pose error separately.
The candidate reaching 16/24 fresh cases still had jerk and contact-load
regressions. It remains unpromoted. See [active results](ACTIVE_WORK.md).

## Evidence and reuse

The [incident manifest](../../configs/benchmarks/mars_arm_sock_replays_v1.json)
contains measured outcomes with source hashes. Its simulator audit cases
replay references from the measured start, while the training export
contains passing synthetic corrective examples. Simulator replay of a
reference alone does not reproduce an actuator stall. For that evaluation,
inject lost tracking and score hold or replanning explicitly.

The local Herman bank contains 511 simulated teacher frames in three
episodes. It was assembled and checked as an input to training. No
checkpoint has been shown to learn these lessons from that bank.

## Pickup policy qualification

Use the deterministic task-space controller as the baseline. Score hand
arrival and retained lift on held-out sock positions. Perturb the
stereo estimate and add joint tracking error in the plant. Include a
fixed-elbow case with a reachable fold and a separate unreachable case
where the expected outcome is a confirmed stop. Compare the same
initial states, references and disturbance timings for the teacher
and policy. Report pregrasp, retention, task error and confirmed stop
separately for each case.

The near-limit collector supports an explicit `--audit-cad-forearm`
option for incident audits. With the corrected hull, the four literal
hardware references still fail the spline's 0.01 rad joint margin at
admission. Preserve those records. The separately named `:inset`
cases move the synthetic initial state off the boundary for testing.
They do not prove recovery from a mechanical stall.

On the two inset cases the deterministic teacher passed both. The local
`mars-whole-body-advantage-recovery-v1/shared/last/actor` checkpoint
timed out on both, with confirmed stops and no flagged contacts or wheel
lifts. Peak hand position error was 13.6 mm on reach and 11.2 mm on
wrist tracking, versus 1.7 mm and 0.2 mm for the teacher. The full report is
`logs/herman-wave/sock-policy-inset-v5/report.json`. This is a local
checkpoint comparison. It does not score the newer managed candidate or
exercise physical pickup and retention.

### Arm-head tuning experiment

A 511-frame imitation run updated the arm gain head from the local
comparison checkpoint while leaving the base gain rows, gripper and shared
encoder frozen. At 50 updates the standalone actor still timed out on
both inset cases. At 200 updates it hit an unsafe contact on the reach
and timed out on the wrist. No standalone candidate was promoted.

An experimental controller that starts from the deterministic teacher
and adds a small clipped correction from the 50-update actor passed both
inset cases with confirmed stops. Peak hand errors were 1.4 mm and
0.39 mm, versus 1.7 mm and 0.18 mm for the teacher. This is comparable on
two simulated motions. Retention, an injected stall and live behavior
remain untested. The comparison and experiment scripts are in
`logs/herman-wave/sock-arm-head-tune-v1/`.

### Retained lift and injected stall

The captured-scene rehearsal supports `--correction-actor` for the bounded
simulator correction and `--stall-elbow` for a stuck position target. The
deterministic baseline retained a 49.8 mm lift. The corrected controller
retained a 49.5 mm lift. Both confirmed holds with both jaws on the proxy.
When the elbow target was held at its initial angle, a shared no-progress
check cancelled both at 0.60 s and confirmed a hold. Each measured elbow
excursion stayed below 0.08°, and neither logged a contact before stopping.
This exercises the shared stop logic, while recovery by replanning remains
untested. The stuck target models lost tracking and does not model the
physical cause of Herman's stall.

Results and traces are under
`logs/herman-wave/sock-retained-controller-comparison-v1/`. The comparison
used one captured scene with a rigid sock proxy and sent no robot commands.
The deterministic controller remains the baseline for first live pickup.

### Mars folded-pose trajectory cost

Rank feasible Mars trajectories by hand tracking, smoothness, and time
spent in the shoulder-and-elbow fold that repeatedly lost its hold.
`mars_folded_pose_cost()` gives a smooth joint-space exposure score. It
ramps up as the elbow moves from 70° to 90° and the shoulder moves from
−45° to −70°. The ramps multiply, so either joint can reduce exposure.
Integrate the cost over physical time. Holding at a risky endpoint costs
more than passing through it during an inward recovery.

Set `mars_folded_pose_weight` in the whole-body reward config to use the
term in a matched Mars training run. Zero preserves existing rewards.
Use the same cost to rank feasible planner candidates when introducing it
to live planning. The angles are provisional and based on this folded
region, not calibrated force thresholds. Check for equivalent task
success, less time near the fold, and recovery from parked starts.
The hardware fault and force limits remain independent of this cost.

The captured-scene rehearsal now reports `audit.safe_start.fold_region`.
This audit bounds the fold score over each straight joint segment with
±2° of shoulder and elbow uncertainty. It rejects entry or renewed folding
above the provisional risk threshold of 0.5. An already folded start can
leave monotonically and must end below that threshold.

The stock 20% lift from the nominal folded capture ended at a bounded
risk of 0.787. Extending it to 40% reduced that bound to 0.222, but three
of nine fixed shoulder/elbow bias sweeps from the nominal folded capture
still showed modeled contact. From the measured supported pose at 82°,
an exit to shoulder −64° and elbow 76° ended at a bounded risk of 0.328
and had zero modeled contacts across the same nine bias sweeps. Three
deeper candidates also passed the fold check. Static geometry and a pose
bound do not establish dynamic tracking, physical clearance, or actuator
health. The supported exit is the candidate for dynamic validation.
Local reports are `mars-fold-region-candidate-v1.json` and
`mars-fold-region-supported-v1.json` under `logs/herman-wave/`.

Mars's opt-in guarded velocity session now reduces only the elbow request
as its verified XL430 load estimate rises from 35% to 40%. It exposes the
remaining elbow speed scale to the caller before the next WBC update.
The session still monitors hardware health and stops on sustained or
ceiling load. Treat the scale as a bound on available joint motion when
replanning from measured hand pose. The live bridge must supply verified,
fresh load and health feedback before enabling this mode.
