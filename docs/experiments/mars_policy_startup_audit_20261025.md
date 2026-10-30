# Mars policy startup and TAMP tracking audit, 2026-10-25

The checkpoint's first failure had two causes at different layers: the simulator
executor allowed inference before receiving a motion command, and the learned
actor itself could not hold an explicitly requested stationary pose. The idle
lifecycle bug is fixed. Learned task success is still unqualified.

## Fixed: motion before the first target

The earlier committed three-case report contains 14 actor calls per case but
zero arm commands before `approach_clearance` fails. A one-point clearance path
waits for physical settling; during that wait the actor was free-running on the
initial hold goal and moved the robot. It had not tracked a commanded TAMP arm
segment. The public execution calls were real, but their failure did not yet
measure nontrivial motion-target tracking.

The simulator now keeps bounded actuator hold until an explicit arm or timed
pose command arrives. It collects history while idle, invokes no teacher, and
does not clear faults. Explicit terminal holds also suppress inference until
another meaningful target resumes tracking. Traces distinguish `policy`,
`executor_idle_hold`, `executor_terminal_hold`, reference actions and fault stops.
Opt-in observation recording retains exact actor inputs before each step.

Replaying all three frozen cases now passes the initial settling wait and issues
one actual pregrasp arm target per learned episode. All three still stop on
pregrasp base drift: 13 calls for `left:20g:0`, 14 for `left:50g:0`, and 13 for
`right:20g:0`. All stops are measured and confirmed. The matched reference
completes delivery in each case and still fails return planning. Both controllers
therefore complete 0/3 full two-transfer programs. No retention, base, contact or
wheel thresholds were relaxed. Full traces are in
`paper/data/whole_body_control/public_tamp_idle_hold_v1.json.gz`.

## Isolating the controller from the simulated plant

One selected `left:20g:0` fixture is reset identically for four two-second hold
diagnostics. Each explicitly commands its measured starting arm pose with the
base constrained. The same 1 cm / 0.02 rad base envelope and force/wheel guards
remain active. A failed envelope triggers a measured stop; it is not counted as
a completed hold merely because the stopped robot becomes quiet.

| Controller | Completed two-second hold | Peak base drift | Peak hand position error |
| --- | --- | ---: | ---: |
| Reference | Yes | 0.139 mm | 0.410 mm |
| Zero velocity | Yes | 0.164 mm | 0.497 mm |
| Fixed checkpoint | No, confirmed stop | 10.661 mm | 15.284 mm |
| Same checkpoint after one second of zero-velocity history | No, confirmed stop | 10.656 mm | 14.973 mm |

The zero controller is a stationary diagnostic, not a trajectory-following
alternative. These results show that this selected plant can hold stably and
that one second of measured history does not repair the raw actor's drift.
They do not rule out dynamics or contact-model problems in other scenarios.
The measured base moves in the direction of the actor's positive velocity
commands, rather than reversing them.

![Stationary holds, inherited phase statistics and offline goal sensitivity](../../paper/data/whole_body_control/mars_policy_startup_audit_v1.png)

## Training and input issues

The selected retention checkpoint is descended from
`carry-ppo-v1/seed-31/step-200704`, followed by 10,240 mixed-task smoke steps.
It is not the earlier working tracking checkpoint. Those tracking families
remain preserved and use a different solver fingerprint; their weights cannot
be silently substituted into the newer contact fixture. This checkpoint was
never promoted as a useful controller.

Its inherited normalization phase means are 96.318% carry, 3.060% reach and
0.216% pregrasp, despite the newer tracking-majority reset mixture. Pregrasp
normalizes to about 21.5 standard deviations. After the idle fix, history also
contains the executor's gripper-open `release` phase; that category had zero
historical prevalence and normalizes to about 9,771 standard deviations before
clipping to 20. This is categorical feature support, not evidence of impossible
physics. Open-gripper reference is also about 6.1 standard deviations from its
carry-dominated mean.

An offline ablation that preserves raw phase bits or neutralizes normalized
phase features still predicts substantial unwanted motion. No such altered
input is executed. Normalization is a problem to address in a newly trained
candidate, not an established single-cause fix or a reason to rescale the
existing checkpoint at runtime.

Offline base-only targets at -10 cm, zero and +10 cm produce normalized forward
commands of approximately -0.249, +0.098 and +0.390. The target direction affects
the prediction correctly in this probe, but the zero-target output is biased.
This does not qualify closed-loop navigation or eliminate all frame bugs.

Other concrete training mismatches are:

- Correction from source inspection during hold repair: manipulation resets
  already sometimes constrain the base, but sample a displaced base target.
  The missing coverage is a stationary base during manipulation and explicit
  sustained holds, rather than base constraints being universally absent.
- The tested configuration permits 5 cm base goal error and uses coarse
  tracking precision; certified arm segments permit 1 cm drift. Hand accuracy
  still has its 5 mm contract and a separate fine stationary reward, so the
  coarse reward alone is not proof of the failure's cause.
- The stochastic training standard deviation is about 0.03. Seven of eight
  coordinates of the first actual commanded-target action put zero velocity
  outside two standard deviations; the largest offset is about 14.7 standard
  deviations. This is a poor state for a tiny-noise polishing run, although PPO
  can still move its mean gradually. No claim of mathematical unlearnability is
  made. The last run also decayed learning rate from 1e-5 toward 1e-6.

## Real-agent and IK accounting

The existing Qwen3-VL agent again selects discovery, planning and execution in
three turns. With the corrected idle gate it issues a real pregrasp arm target,
then reaches the same 13-call drift stop. Its initial physics, recorded actions
and base states match the scripted checkpoint episode exactly. Inputs remain
privileged and physics pauses during model inference.

There are nine initial planning IK calls and **six geometric replanning IK
calls during `execute_pick_place_plan`** in this corrected attempt. Those calls
revalidate measured-state TAMP paths; the actor performs no IK. The earlier
zero-execution-IK observation applied to an episode that failed during the
initial wait. Removing the remaining joint-witness executor requires a checked
Cartesian trajectory backend, not just renaming this policy hook. Its joint
convergence checks remain in place until an alternative path certificate exists.

## Next experiment and gates

First require the learned candidate to sustain stationary and nearby FK-reachable
goals for two seconds: hand error within 5 mm / 0.05 rad, base drift within the
existing arm envelope, measured quiet hold and no force/wheel failures. Include
both base-constrained and free-base hand targets, then retargets and disturbances.
Keep a substantial ordinary tracking share rather than making all resets grasps.

Audit the useful tracking checkpoint as a possible initialization source,
explicitly migrate its contact/observation contract through training, and compare
it with the current carry initialization. Train with representative categorical
features and base/hand mask combinations, plus enough hold and fine-accuracy
coverage. Compare exploration and learning-rate choices on these gates before
another long run. Do not add a hidden teacher residual or change runtime
normalization to make the old checkpoint appear better. After hold and pregrasp
pass, score the unchanged full TAMP trajectories and loaded manipulation cases.
The return/regrasp planning failure remains a separate regression.

## Evidence and reproduction

All 668 source-bound regression tests pass, including the idle activation,
explicit trajectory, terminal hold, named Mars/Stretch observation layout and
input-audit tests. The earlier focused run passed 41 tests; the final complete
run includes the additional trajectory regression. The verified receipt is
`paper/data/whole_body_control/mars-policy-audit-regressions-20261025.receipt.json`.
The full startup/input audit, raw observations, ablations and agent trace are
archived beside their summaries as `mars_policy_startup_audit_v1` and
`tamp_agent_idle_hold_v1`. Existing reports and failed cases remain intact.

```bash
taskset -c 14 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m emet.eval.mars_policy_audit \
  --case left:20g:0 --output logs/eval/new_mars_policy_audit
```

Use a fresh output directory. The tool records actor observations, normalized
support/clipping, offline predictions, stationary baselines and a public TAMP
replay. It starts no inference service and uses no hardware. No PPO training or
checkpoint promotion was performed for this diagnosis.
