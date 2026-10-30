# Mars hold-repair training, 2026-10-25

The first bounded comparison implements fixed-current-base manipulation resets,
two-second reach holds (including physical pregrasp postures), fine tracking
rewards, categorical feature identity normalization and a bounded cost for all
six measured base velocities. Ordinary tracking retains 70% of reset selection;
hold trials occupy 20% of that reach share, rather than replacing manipulation.
Existing force, speed, retention, wheel and stop limits remain unchanged.

The earlier audit overstated the sampling gap: manipulation resets already
sometimes constrained the base, but sampled a displaced base target. The new
option holds the base target at its measured starting pose. Free-base and
coordinated cases remain available. Categorical normalization is stored as
ordinary identity statistics in each newly exported bundle; old bundles and
runtime loader behavior are unchanged.

## Initializations and explicit training transfer

Four initializations were compared on a stationary pose and a nearby FK joint
witness, with collision and static support screening. The better of the two
tracking sources was the precision-refinement MLP. Both useful tracking sources
use four history samples and predate the v7 wheel-feedback contract; the carry
source is a transformer. The experiment matches each source architecture rather
than copying incompatible weights.

Transfer is explicitly training-only and starts a fresh optimizer. It validates
the original fingerprint, compiled robot/actuator/mesh identity and action
units. Only contact cone/impedance changes and the declared v6-to-v7 bounded
wheel-feedback upgrade are allowed. Retention-input columns start with zero
influence; continuous-statistic prior count is capped and categories use native
units. The original bundle remains immutable and rejects normal runtime loading
under a mismatched contract. This migration does not qualify transferred control.

## Completed first pilot

One zero-output MLP and one precision-tracking MLP each received **4,096 PPO
steps**, with one CPU environment, constant learning rate 5e-5, exploration
standard deviation 0.1 and offline scalar-only W&B. No teacher residual or
demonstration loss was used. The runs preserve the complete ordinary/loaded
validation and matched teacher results, not just the hold cases.

Neither candidate passes the strict hold/nearby endpoint checks on either
evaluation fixture. The zero-output initialization learns about 13–14 mm hand
drift on a stationary target, while improving its nearby target residual to
about 8 mm. The transferred precision model also fails. Four thousand steps
are a short pilot, not convergence evidence. In particular, this does not show
that another initialization or longer well-specified PPO training cannot work.

The six-DOF quiet check also catches residual rocking in a zero-velocity
baseline whose hand position is otherwise stable. Position accuracy and
confirmed whole-body stopping are therefore reported separately. The new body
motion cost addresses components omitted from the older planar-motion reward;
the stop thresholds were not relaxed to count success.

Raw source/weight-bound manifests, validation, W&B scalar history and all
stationary/nearby traces are archived in
`paper/data/whole_body_control/hold_repair_pilot_v1.json.gz` with an adjacent
checksum summary. The 683-test source-bound receipt covers this implementation.
Two prototype setup rejections (legacy contract and architecture mismatch) are
retained locally; neither launched an invalid runtime policy.

## Follow-up: learned feedback head and corrected examples

The ordinary velocity head can learn a constant motion bias even from a zero
initialization. The completed follow-up adds an opt-in learned gain matrix
applied to current/future pose errors and measured velocities. Context/history
can change the gains, but deterministic mean motion vanishes when all these
signals vanish. A separate bounded gripper head retains executor-only release
authority. This is learned feedback without teacher actions or a residual; the
equilibrium property does not prove convergence or stability. Stochastic PPO
exploration also remains nonzero during training.

The follow-up excludes physical pregrasp holds from closed-empty reach examples.
Categorical features and feedback errors/velocities use fixed physical scales,
so a physical zero remains zero after normalization. Ordinary continuous context
statistics still adapt. Explicit tracking transfer retains the compatible
context/critic, reinitializes the new motion/gripper heads and starts a fresh
optimizer. Inference admission additionally pins the feedback, normalization,
layout, control, preview and pose-target implementations for these bundles.

The feedback comparison trained scratch and transferred MLPs for 4,096 PPO steps
each. On the same two development fixtures, transferred peak hand drift was
0.45/0.61 mm over two seconds, versus approximately 13/14 mm for the initial
velocity-head scratch pilot. It ended quiet on both fixtures, but endpoint quiet
is not a continuous hold. A subsequent exact continuation added 8,192 steps to
the selected feedback model, for 12,288 steps on that candidate and **24,576 new
PPO steps across all five runs**. W&B remains offline and scalar-only.

## Longer development checks and public TAMP execution

The continued checkpoint was tested on five frozen development fixtures across
two layouts and 20/50/100 g payload identities. These are pregrasp states, not
loaded carry trials. Each fixture has a ten-second stationary hold, a nearby
FK target at three and six seconds, and a reachable 0.12 rad wrist rotation at
six seconds. FK witnesses are screened for collision and static support margin;
no IK is used to generate these new nearby goals.

| Check | Result over five fixtures |
|---|---|
| Ten-second hold peak hand error | 0.34–1.17 mm; peak base displacement 1.70 mm |
| Nearby target, three seconds | 6.40–7.48 mm endpoint error; 0/5 endpoint passes |
| Nearby target, six seconds | 3.81–6.08 mm; 4/5 within position bound, 2/5 also quiet at endpoint |
| Wrist rotation, six seconds | 0.118–0.122 rad remaining; 0/5 passes |
| Continuous two-second quiet pose | 0/20; stationary longest streaks 0.05–0.35 s |

![Hold position, nearby tracking, rotation and continuous quiet checks](../../paper/figures/mars_hold_feedback_20261025.png)

The first panel compares the initial scratch velocity head's two-second
endpoint error with the continued feedback model's ten-second peak error.
Durations, fixtures and training budgets differ; it is a development comparison,
not a controlled estimate of the architecture's effect alone.

One rotation trial stopped on its inference deadline; it remains a failure.
There were no wheel-lift detections in these twenty cases. Peak reported contact
force was 1.98 N. These selected, predominantly free-space trials do not establish
compliant contact, disturbance recovery or loaded retention. Small body angular
oscillations still violate the unchanged six-DOF quiet check even where hand
position looks stable. Rotation and translation progress remain inadequate.

The same numbered checkpoint ran the three public delivery/return development
programs. **0/3 programs and 0/6 transfers complete.** Each makes 101 actor calls
and stops at the first pregrasp joint waypoint with confirmed stop. Final hand
error is 1.39–1.45 mm and orientation error 0.0047–0.0051 rad, while joint
certificate residual is 0.0153–0.0158 rad against the executor's 0.005 rad limit.
This exposes a pose-policy/joint-certificate mismatch; it is not successful
approach execution. The current backend still converts certified joint
waypoints to hand targets and checks the joint witness. It does not yet execute
the complete Cartesian schedule as one policy trajectory. Planning retains IK;
policy inference does not use it. No certificate or task threshold was relaxed.

The broader five-category validation and matched teacher results are archived,
including a disturbance retention loss and a failed paired no-regression gate.
One episode per category is a diagnostic, not a success-rate estimate. The
hold improvement does not establish better overall performance than the teacher.

Reproduce the bounded comparison with one CPU environment:

```bash
taskset -c 14 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  WANDB_CACHE_DIR=/tmp/emet-wandb-cache WANDB_CONFIG_DIR=/tmp/emet-wandb-config \
  .venv/bin/python -m emet.rl.manipulation.hold_experiment \
  --spec configs/benchmarks/mars_hold_training_feedback_v2.json \
  --output logs/manipulation/hold-feedback-reproduction --wandb-mode offline
```

Raw traces, embedded evaluation drivers, source/weight hashes, numbered
checkpoint manifests, training commands, scalar logs, validation and teacher
comparisons are archived under `paper/data/whole_body_control/`:
`hold_feedback_training_v2`, `hold_feedback_validation_v2` and
`hold_feedback_public_v2` (JSON summaries plus checksummed gzip reports).
Weights stay local in `logs/manipulation/hold-feedback-continued-v5/seed-64/step-12288`.
The feedback comparison precedes verifier hardening; the continuation and final
evaluations use the final verifier. Pinned feature/control sources match.
The fresh source-bound receipt covers **692 passing tests**. Earlier negative
pilot evidence remains unchanged and is bound to its own source snapshot.

Next gates are explicit orientation coverage/scaling ablations, measured
body/contact chatter diagnosis, and a Cartesian execution contract that preserves
independent collision/path checks without demanding an arbitrary IK joint
solution. Then repeat the unchanged public and loaded gates before extending
training. No new checkpoint is promoted, and no hardware was used.
