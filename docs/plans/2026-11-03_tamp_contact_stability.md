# Native contact stability experiment

The November 1 agent pilot passed both familiar controls and failed both +15 mm
initial-X shifts during cube lift. Its live replay rejected relative rotation
of 0.100849 rad while translation was 1.315 mm and both fingertips still touched
the cube. Keep that 2/4 agent result unchanged. First improve the evidence about
grasp geometry and loaded motion through the same public task tools.

The [frozen diagnostic population](../../configs/eval/tamp_contact_stability.json)
contains four live MuJoCo episodes: the two-goal control and shifted layout, each
under the existing grasp yaw candidates and a finer yaw candidate set. Add only
the +/-0.175 rad candidates. Preserve beam task search, 256 IK calls per
refinement, 2,048 provider calls, native controllers/contact parameters, final
regions, and 2 cm / 0.1 rad retention limits. The production adapter retains its
original policy; the variant belongs to the diagnostic adapter until qualified.
One development perturbation cannot justify promoting a new policy.

`scripts/eval_tamp_contact_stability.py` reconstructs initial placements from
versioned canonical assets, checks the compiled model signature, archives source
and scene hashes, and uses ordinary ZMQ arm/gripper commands. It runs
`scene_tasks`, `plan_task` and `execute_task_plan` through scripted CHAT dispatch.
There is no LLM or alternative task/execution loop. Setup uses the normal native
navigation posture and head look, followed by six simulated seconds of settling.
This is privileged, stationary simulation; hardware trials remain paused.

`ContactTrace` records selected planning targets separately from measured-joint
FK/contact reconstruction at closure, lift and placement transitions. Private
`MjData` prevents the diagnostic from modifying executing or planning state.
Full joint state, object/EE transform, current payload reference, fingertip
contacts and event timings remain in raw artifacts. Passive finger angles and
contact pairs are simulator debugging truth. No hardware sensor claim is added.
Recorder wall time is measured because instrumentation changes live timing.

Run at most two CPU simulations at once through `emet jobs`; no local GPU or
inference service is needed. Use `MUJOCO_GL=disable`, empty
`CUDA_VISIBLE_DEVICES`, and single OpenMP/BLAS threads. CLI help was checked:

```bash
python scripts/eval_tamp_contact_stability.py --case-id two-goal-shift \
  --policy baseline --port-offset 1940 --output NEW_RUN_DIR
```

`--prepare-only` freezes/validates inputs without starting a simulator or
controllers. Outputs must be new directories. Failures retain terminal summaries;
truncated audits cannot pass. Score the final released/settled goals and receipt
separately from contact residuals. Count all four fixed episodes, including
planning, tracking, retention and infrastructure failures. Initial measured
states are archived but independent live resets are not bit-identical replay
certificates. Compare both policies under the same recorder and startup policy.

If finer candidates produce a more stable grasp, check the control and a broader
placement grid before changing production defaults, then repeat the frozen
actual-agent cases through Caliban. If they do not, use the closure/lift traces
to distinguish targeting, actuator tracking and compliance-driven object motion.
A fix that admits payload rotation must include its volume in collision
validation; changing the retention threshold alone is insufficient. Assisted
RGBD and observed execution still require self-occlusion/background/contact source
qualification after this native gate.

The [completed comparison](../experiments/tamp_contact_stability_20261103.md)
retains 1/2 complete tasks per policy. Finer yaw completes the shifted cube lift
but fails later during placement. The next isolated candidate is a checked
loaded-grasp payload binding followed by complete placement replanning; neither
the candidate search policy nor a relaxed retention limit has been promoted.
