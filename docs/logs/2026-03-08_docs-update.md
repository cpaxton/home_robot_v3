# Refactor log: 2026-03-08 — Docs update and MuJoCo fixes

## Summary

Updated documentation to reflect the emet CLI, current commands, and correct paths. Fixed MuJoCo server shutdown and import compatibility.

## Changes

### simulation.md
- Added tip about [emet CLI](cli.md)
- Replaced `python -m emet.simulation.mujoco_server` with `emet serve mujoco` as primary
- Replaced `python -m emet.app.*` with `emet run *` where applicable
- Added CLI to demo scripts table

### dynamem.md
- Added `emet serve mujoco` and `emet run dynamem` as primary commands
- Added `emet sync -e dynamem` for SAM2 install

### debug.md
- Added "See also: emet CLI" at top
- Updated timing examples to use `emet run timing`

### apps.md
- Added emet CLI examples at top
- Updated simulation note to mention `emet serve` and `emet run`

### install_details.md
- Updated simulation quick start to use `emet sync -e sim`, `emet serve`, `emet run`
- Added `emet sync -e dynamem` for SAM2

### jetson.md
- Fixed invalid `emet.app.llm_agent` → `emet.app.chat` and `emet.app.ai_pickup`

### CONTRIBUTING.md
- Fixed stretch_ros2_bridge path (emet_ros2_bridge → stretch_ros2_bridge)
- Updated testing section to use `emet test`

## MuJoCo server fixes

### Shutdown (AttributeError: can't set attribute 'running')
- `MujocoZmqServer.stop()` was setting `self.running = False` but `running` is a read-only property
- Fixed: use `self._done = True` instead; set _done before stopping robot_sim so spin threads exit cleanly
- Join control thread and base send/recv threads before stopping robot_sim to avoid ConnectionError in threads

### Mujoco import compatibility
- Added fallback: `try: from mujoco import MjModel` / `except: from mujoco._structs import MjModel`
- Applied in stretch_mujoco_simulator.py, mujoco_server.py, mujoco_server_managed.py, utils.py
