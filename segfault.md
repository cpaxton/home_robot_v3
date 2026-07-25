# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

# Cursor agent segfault log

Append incidents chronologically. Keep the newest entry self-contained and at the bottom so
`tail -n 80 segfault.md` shows the current diagnosis and exact recovery point.

Do not record an incident as a generic “GPU crash.” Distinguish:

- Agent/runtime crash: kernel names `node` or `emet`, often `invalid opcode` or null instruction
  pointer.
- Native workload crash: kernel names Python and usually identifies `libcuda`; the child exits 139.

## 2026-07-25 10:30 EDT — agent runtime invalid opcode

### Evidence

- Terminal symptom: `Segmentation fault (core dumped)` immediately after the Cursor agent exited.
- Kernel (`journalctl -k --since "2026-07-25 10:30:00"`):
  `traps: MainThread[1964009] trap invalid opcode ip:2e76edc ... in node[...]`
- Classification: agent/runtime crash (Mode B), not a Python `libcuda` episode crash (Mode A).
- `coredumpctl` is not installed, so no saved stack/core metadata was available.
- Post-restart GPU check: 23751 MiB free of 24564 MiB; no compute applications.
- No intentional long-running GPU job was found or restarted.

### Likely trigger context

The crashed session repeatedly started/stopped MuJoCo EGL servers and native probes inline, including
`pkill -f` cleanup, before returning to source edits. This does not prove causality, but it matches the
known agent-runtime failure pattern after native graphics teardown. Avoid more inline EGL probes in
the recovery session; use a dedicated terminal or managed background job when behavior verification
requires MuJoCo.

### Work completed immediately before the crash

The session diagnosed navigation timeout failures as wall-clock/simulation-time mismatch:

- Physics-only Stretch scene measured about 0.77x real time.
- Physics plus camera rendering measured about 0.27x real time.
- A 10-second wall-clock navigation timeout therefore permits only about 2.7 simulated seconds.
- The client wait loop itself exited correctly in an isolated 0.3 m test.
- Base yaw stopped about 0.099 rad short, matching the configured 0.1 rad controller tolerance.

An incomplete protocol implementation is present in the working tree:

1. `EMET_ZMQ_SIM_TIME_RATIO_KEY = "sim_to_real_ratio"` added to `zmq_protocol.py`.
2. Numeric `sim_to_real_ratio` added to `StatusStretchJoints`.
3. `MujocoServer` populates it from `physics_fps_counter.sim_to_real_ratio`.
4. Stretch ZMQ state messages publish it.
5. No client-side timeout scaling was implemented yet.
6. No tests or documentation for this protocol change were added yet.

These edits are mixed with substantial pre-existing manipulation work. Do not discard or bulk-stage
the tree.

### NEXT — recovery checklist

1. ~~Inspect `src/emet/controller/zmq_client.py` motion wait methods around `at_goal()` and
   `is_base_moving()`.~~ Done — scaled waits + base joint-speed idle.
2. ~~Add one helper that reads `EMET_ZMQ_SIM_TIME_RATIO_KEY`, uses scale
   `max(1.0, 1.0 / ratio)`, and clamps malformed/extreme values to a documented maximum.~~
   Done — `motion_wait_timeout_scale` / `read_sim_to_real_ratio` in `zmq_protocol.py`.
3. ~~Scale only simulator motion waits; absence of the key must preserve hardware behavior exactly.~~
4. ~~Tighten completion checks using published velocity/goal state where possible; timeout remains a
   safety bound, not the normal completion mechanism.~~ Base `x_vel`/`theta_vel` now fill joint
   velocity slots; nav wait exits on `at_goal` + idle.
5. ~~Add focused unit tests for absent, `None`, normal, slow, malformed, and clamped ratios.~~
6. Run focused non-GPU tests (`uv run emet test src/test/core/test_zmq_protocol.py`). MuJoCo behavior
   smoke outside the agent process if needed.
7. Review all staged/unstaged files before honoring the pending request to commit and push the current
   feature branch. Never push `main`.
