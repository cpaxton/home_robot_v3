# Pose-conditioned manipulation policy and value gate

## Control contract

TAMP proposes a bounded set of world-frame end-effector transforms. The
controller converts each proposal to the current end-effector frame and uses
translation plus the continuous 6D rotation representation. The actor consumes
that target with measured joint/gripper history and emits bounded joint-velocity
residuals around a damped-Jacobian Cartesian servo. This keeps the execution
interface in task space without running a candidate-by-candidate IK planner.
The Cartesian servo is a feedback controller, not a feasibility certificate.

## Candidate set and masks

Use one shared pose-conditioned value network for every candidate in a padded
batch. Its recurrent state encoder processes proprioceptive history; a pose
encoder processes each relative SE(3) target; a robot embedding conditions the
value on the robot profile. A boolean candidate mask excludes padding from
inference, loss, ranking, and metrics. Do not mask hard-to-reach examples out
of training: retain them with outcome labels so the value model can learn their
low return.

An inexpensive robot-specific workspace prior may mark poses outside a broad
sampled envelope. Treat that as a feature or conservative prefilter, not a
ground-truth reachable mask: collision, joint coupling, payload and contact
still affect executability. A separate ensemble disagreement/OOD gate makes
the rejection decision abstain when evidence is weak.

## Value and rejection semantics

Train value from returns of full pose-conditioned closed-loop episodes. Reward
must include goal completion, retention and placement quality, while force,
jerk, wheel lift and destructive contact carry explicit penalties. Calibrate
the minimum accepted lower confidence bound on held-out candidate outcomes;
never tune it on frozen oracle fixtures. Low value rejects a pose. High
uncertainty is `unknown`, not a pass. Keep IK available as a bounded
diagnostic/fallback during development, but remove its repeated candidate
search from normal execution after the policy gate passes.

## Data and acceptance sequence

1. Sample robot joint configurations and convert them to end-effector poses to
   cover the local reachable workspace, then add near-boundary and out-of-range
   poses. Record the robot/model fingerprint and pose frame.
2. Train a pose-conditioned actor and critic on the same observations and
   targets. Preserve grasp, carry, release, disturbance, force and wheel-lift
   labels; split by episode and scene before calibration.
3. Compare value-ranked candidate rejection against bounded IK search and the
   existing feedback baseline. Include padding permutations, held-out workspace
   regions, the frozen left/right oracle cases, and generated multi-object TAMP
   cases. Require safe false-accept limits as well as task success.
4. Only then promote the pose contract to the live controller interface. Keep
   model-specific artifacts separate until each robot has adequate coverage.

## Initial implementation

`src/emet/rl/manipulation/pose_value.py` defines the relative-pose encoding,
masked candidate value architecture, and conservative ensemble selection
contract. This is a schema/architecture foundation; the current joint-target
training artifacts do not implement this contract and are not promoted.
