# Isolate Mars arm learning before adding policy structure

Keep the production whole-body actor unchanged. A separate component experiment
uses the same pose/history/preview features and learned-feedback head with five
joint-rate outputs. The wrapper inserts zero wheel requests and a zero gripper
offset. The base remains free in physics: drift, six-DOF quiet, contact loads and
wheel unloading are still checked. Grip follows the existing executor reference.
There is no task manager, multi-head production network or teacher action loss.

`python -m emet.rl.manipulation.arm_component` provides the diagnostic. It reuses
the canonical arm cases, smaller training amplitudes, native reward and all
acceptance thresholds from the hand-accuracy preset. Case IDs are reset/logging
metadata only. Validation postures never enter training. A component checkpoint
has no production `manifest.json` and is not a deployable whole-body controller.

1. Run the zero-wheel/fixed-grip teacher on all 16 admission cases. All must pass
   the original pose/path/quiet/support guards. Compare the objective against
   zero arm requests, including stopped prefixes and discounted return.
2. Test five-to-eight action mapping, invalid command rejection, unchanged base/
   hand goal masks and grip reference, actual PPO updates, and checkpoint reload.
   Run fresh source-bound full controller regressions before training.
3. One 32,768-step scratch PPO diagnostic on CPU14 only: five action dimensions,
   std0.05, LR1e-4, gamma0.9975, rollout/batch256, five epochs, clip0.2 and
   target KL0.02. The scratch critic and less conservative optimizer are declared
   changes from the mixed warm-start; this is not a matched action-mask ablation.
   Keep the existing 16k raw-command penalty ramp and full physical safeguards.
   Evaluate all eight canonical arm cases at step0, every8k and after the final
   update, with full evaluation costs. Log offline scalars and preserve failures.
4. If this succeeds, compare against whole-body learning and test loading next.
   If it fails, inspect the same goal/command/reward traces before another run.
   Teacher/student initialization remains an option, not an imposed residual or
   a performance ceiling. No automatic extension or production promotion.

The teacher currently passes 16/16. Its discounted return beats zero requests on
16/16 cases, with minimum margin28.49 at gamma0.9975. This rules out that specific
bad-behavior preference; it does not prove the entire objective is well aligned.
All pose, jerk, force and retention thresholds remain unchanged.
