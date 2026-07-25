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
6. ~~Run focused non-GPU tests (`uv run emet test src/test/core/test_zmq_protocol.py`).~~
   Re-verified 2026-07-25 11:43 with the broader TAMP unit set (exit 0).
7. Review all staged/unstaged files before honoring the pending request to commit and push the current
   feature branch. Never push `main`.

## 2026-07-25 ~11:42 EDT — agent segfault mid TAMP docs/tests

### Evidence

- Terminal / Cursor UI: `Segmentation fault (core dumped)` while the agent was mid “Run Everything”
  on branch `feature/molmospaces-rby1-agent-manip` (PR #83 context in UI).
- Kernel (`journalctl` / `dmesg` for 11:20–11:50): **no** new `segfault` / `invalid opcode` /
  `libcuda` lines. Matches Mode B (agent/runtime), not Mode A episode crash — same family as the
  10:30 `node` invalid-opcode incident; this turn left no fresh kernel trap line.
- `/tmp/tamp_unit.log` from the crashed turn: pytest reached `[100%]` (dots only; summary truncated).
- Post-recovery: GPU **23731 MiB free / 24564**; compute apps none.
- `emet jobs`: `20260725_114158_32d3a0` **waiting** (`hmeqa-bal32-aff`) — unrelated; do not kill
  unless intentional.

### Work in flight at crash (tree intact — do not discard)

Uncommitted TAMP / manip stack (still present):

| Path | Role |
|------|------|
| `src/emet/controller/task/tamp/` | beam-style `approach → grasp → place` search |
| `scripts/scripted_tamp_pick_place.py` | table+rby1 kinematic smoke + figures |
| `src/emet/visualization/manip_figures.py` | paper PNG/PDF figures |
| `src/test/controller/task/`, `src/test/visualization/test_manip_figures.py` | unit tests |
| `docs/motion_planning.md` | “Task search (no-NN TAMP)” section |
| manip/sim/rerun/profile edits | kinematic pick-place, arm profile, sim manip, Rerun |

`docs/cli.md` habitat row is already on the branch (not dirty). Prior ZMQ `sim_to_real_ratio`
wait-scaling work is in-tree and covered by `src/test/core/test_zmq_protocol.py`.

### Verified after recovery

```bash
uv run emet test \
  src/test/controller/task/test_task_search.py \
  src/test/visualization/test_manip_figures.py \
  src/test/motion/test_arm_manip_profile.py \
  src/test/simulation/test_sim_manipulation.py \
  src/test/core/test_zmq_protocol.py -q
# exit 0 — 20 dots
```

### NEXT — recovery checklist

1. ~~Re-run focused unit tests (above).~~ Done, exit 0.
2. Do **not** re-probe MuJoCo EGL / Habitat inline in this agent turn.
3. Optional behavior smoke (dedicated terminal / `emet jobs`, not agent-inline):
   `EMET_SIM_NAV_TELEPORT=1 MUJOCO_GL=egl uv run python scripts/scripted_tamp_pick_place.py …`
   only when the user wants sim proof.
4. When the user asks: review full dirty tree, commit on **this feature branch only**, push
   `feature/molmospaces-rby1-agent-manip` — never `main`.
5. Leave waiting `hmeqa-bal32-aff` alone unless the user wants it cancelled.
