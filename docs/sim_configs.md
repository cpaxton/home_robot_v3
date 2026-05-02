# Simulation launch configs

YAML files describe how to start `emet.simulation.mujoco_server` for **default MuJoCo** (packaged table + robot), **Robocasa**, or **MolmoSpaces** merges. They are used by:

- `emet serve mujoco` (internally builds the same argv shape)
- `emet run agent --start-sim` (reads config from the agent YAML or `--sim-config`)

## Reference files

| File | Purpose |
|------|---------|
| [`configs/sim/default_table_rby1.yaml`](../../configs/sim/default_table_rby1.yaml) | `scene_default.xml` + rby1 / Galaxea R1, headless |
| [`configs/sim/robocasa_pick_place.yaml`](../../configs/sim/robocasa_pick_place.yaml) | Robocasa kitchen task |
| [`configs/sim/molmospaces_ithor_train_0.yaml`](../../configs/sim/molmospaces_ithor_train_0.yaml) | MolmoSpaces iTHOR train index 0 + rby1 |

## Linking from an agent config

In any dynav-style agent YAML (the file passed to `--agent-config`):

```yaml
sim_config: configs/sim/default_table_rby1.yaml
```

Or inline:

```yaml
sim:
  kind: molmospaces
  scene: ithor
  split: train
  index: 0
  robot: rby1
  headless: true
```

Precedence: **`--sim-config PATH`** overrides both `sim_config:` and inline `sim:`.

## One-terminal agent + sim

`--command` / `-c` already runs non-interactive user turns then exits. With `--start-sim`, the sim is spawned in-process first (same ZMQ ports as `emet serve mujoco`; use `--port-offset` on the agent to match a non-default sim if needed).

```bash
uv run emet run agent --robot rby1 --agent-config configs/agent_rby1_discord.yaml \
  --start-sim --no-discord --command "What do you see?"
```

MolmoSpaces requires the emet-molmospaces wrapper (`.venv-molmospaces`) and scene assets; use `configs/sim/default_table_rby1.yaml` in `sim_config` for a smaller offline setup.

## `kind` field

- `default_mujoco` — optional `scene_path` for a merged MJCF; otherwise the default table scene + `--robot`.
- `robocasa` — `robocasa_task`, `robocasa_style`, `robocasa_layout`, plus shared flags.
- `molmospaces` — `scene`, `split`, `index`, optional `molmospaces_install`; merge runs before the server starts.

Shared flags on all kinds: `port_offset`, `headless`, `show_viewer_ui`, `no_cameras`, `use_glx`, `seed`, `steps`, `debug_molmospaces_spawn`, `verbose`.
