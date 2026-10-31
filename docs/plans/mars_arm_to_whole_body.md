# Preserve the arm result while restoring whole-body learning

The five-output component passes all 16 unloaded arm admission cases. Transfer
it into the existing eight-output, history-and-preview whole-body actor. This
is a training warm-start, not a second runtime policy, teacher residual or new
task manager. Runtime still consumes masked base SE(2), hand SE(3) and gripper
targets and applies the existing executor and physical safety checks.

1. Verify the producer's completed run, exact environment, stable model/control/
   observation sources, all eight canonical component passes and checksums of
   its final actor, statistics and evaluation **before deserialization**. Copy
   the shared actor context, critic, five arm gain rows and normalization. New
   general base rows and grip offset start at zero. The existing restoring-yaw
   branch uses its normal constructor initialization; the earlier base policy
   is not preserved. Do not mutate the producer checkpoint.
2. Test actual PPO-produced weights, arm means, critic values, normalization and
   full runtime export/reload. Reject modified artifacts and incompatible
   configuration. Audit the real checkpoint's predictions, then repeat the
   unchanged physical arm cases before training. Run the complete source-bound
   regression gate and fresh matched base/arm teacher admission.
3. Run one bounded **32,768-step** local PPO pilot, seed90, one CPU14 environment,
   CPU inference and one Torch/BLAS thread. LR3e-5, gamma0.9975, std0.01 for each
   of eight outputs, rollout/batch256, five epochs, clip0.1 and target KL0.01.
   Freeze the copied normalization to preserve the arm feature basis. Rehearse
   75% arm /25% base episodes, with the existing smaller training amplitudes;
   validation resets remain excluded from sampling. Use full command costs
   from step0; no new action filter or teacher action loss. Record offline W&B
   scalars and evaluate all 18 canonical cases at step0, every8192 and final.
4. Keep each scheduled checkpoint, all failures, teacher-paired jerk/force and
   quiet-hold metrics. Evaluate the final checkpoint on all 66 expanded cases.
   Report arm and base separately; an aggregate score cannot hide forgetting.
   If arm regresses or base does not learn, inspect measured/commanded motion,
   clipping, reward terms and optimizer diagnostics before extending training.
   Frozen arm-only statistics may clip unfamiliar base inputs: measure it rather
   than silently adapting the normalization during this first experiment.
5. Only after reliable base/arm tracking, add simultaneous targets, loaded hold/
   lift transitions and the unchanged public TAMP pick/place programs. Those
   remain separate acceptance cases; empty-hand success does not qualify carry,
   agent execution or Herman. Do not weaken pose, path, jerk, force, wheel-load,
   stop-confirmation or frozen-fixture gates to start another run.

Use `configs/benchmarks/mars_tracking_mastery_arm_to_whole_body_v1.json` and
`mastery train --initialize-arm-from .../last --arm-evidence .../results.json
--freeze-normalization`. The component's producer files stay immutable and
unmodified components remain non-deployable. Generated weights and traces stay
in ignored `logs/`; only compact metrics, hashes and reports enter Git.
