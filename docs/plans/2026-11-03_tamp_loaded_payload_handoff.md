# Checked loaded-payload handoff

The finer-yaw diagnostic completes the shifted cube lift but rejects later in
placement. Its placement targets use measured post-lift position while the
collision payload transform remains from before lift. Test a single checked
primitive handoff, with complete placement replanning, before changing production.

The [four-case follow-up](../../configs/eval/tamp_loaded_payload_handoff.json)
uses the same fine-yaw search for familiar and +15 mm layouts. Both policies
first pass the production lift/one-second bilateral hold verification, then add
the same 0.3 simulated-second stability interval. One keeps the old reference;
the other binds the measured loaded transform once. This hold-only control
isolates reference renewal from additional settling time. All episodes use
scripted public task tools, live MuJoCo physics and no LLM or hardware.

Throughout that interval the original 20 mm / 0.1 rad bound must still pass.
Both fingertips must contact the object with no other body contact. Relative
pose may vary by at most 2 mm / 0.01 rad from the first interval sample; the
anchor never resets. Loss, excessive creep, timeout or an already-violated
original bound cannot renew a reference. Each executed grasp permits one
handoff check. The default runner policy remains `preserve`.

After a successful renewed binding, the existing native execute path recomputes
placement targets and `place_only` replans every placement segment using the new
transform. The reference does not change at individual waypoints. This changes
the declared reference between primitives, rather than relaxing numerical pose
thresholds. Collision scope remains the existing sampled arm/payload checker;
no global dynamics/collision certificate or observed-input qualification follows.
Simulator contacts/passive joints remain explicit privileged diagnostic inputs.

The runner exposes `--payload-binding preserve`, `verified_lift_keep` (same
checked hold without renewal), and `verified_lift` (one measured renewal).
Archive source, nominal scene hashes, actual initial state, trace, binding
before/after, receipts, final goals and command audit. Retain all four outcomes
and keep the preceding failed four-case comparison and original 2/4 agent
population unchanged. At most two CPU simulations run together through
`emet jobs`; no local GPU or VLM is needed.

If this improves complete native tasks and preserves controls, next verify a
broader prospective placement population before promoting a default. Then repeat
the actual-agent cases through Caliban. A native profile success is not an agent
result or evidence that observed robot/object/contact sources are qualified.
