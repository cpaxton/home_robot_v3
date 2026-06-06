# Simulation launch configs

YAML files describe how to start `emet.simulation.mujoco_server` for **default MuJoCo** (packaged table + robot), **Robocasa**, or **MolmoSpaces** merges. They are used by:

- `emet serve mujoco` (internally builds the same argv shape)
- `emet run agent --start-sim` (reads config from the agent YAML, `--sim-config`, or a **built-in default** when neither is set — see below)

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

If you pass **`--start-sim`** with no `sim:` / `sim_config:` in the agent YAML and no **`--sim-config`**, emet uses the **packaged default-table MuJoCo** scene with the same **`--robot`** (or YAML `robot:`) and honors **`--headless`** for the sim as well.

Sim-only flags (same idea as `emet serve mujoco`) include **`--scene`**, **`--split`**, **`--index`**, **`--install-scene-if-missing`**, **`--robocasa-task`**, **`--sim-seed`**, **`--sim-steps`**, **`--sim-no-cameras`**, **`--sim-use-glx`**, **`--sim-show-viewer-ui`**, **`--sim-debug-molmospaces-spawn`**, **`--sim-show-subprocess-output`**. They require **`--start-sim`**.

## One-terminal agent + sim

`--command` / `-c` already runs non-interactive user turns then exits. With `--start-sim`, the sim is spawned in-process first (same ZMQ ports as `emet serve mujoco`; use `--port-offset` on the agent to match a non-default sim if needed). The subprocess uses a **new session** so Ctrl+C is handled cleanly: the sim is terminated before the agent exits. Non-interactive ``--command`` / ``-c`` runs **disable Discord** automatically; if Discord would have started (default on when ``DISCORD_TOKEN`` is set) and you did not pass ``--no-discord``, emet prints a one-time warning—use ``--no-discord`` in scripts to silence it. When using **Stretch** with ``--start-sim`` and ``--command`` / ``-c``, the MuJoCo server is started **headless** automatically (unless you pass ``--sim-show-viewer-ui``) so the passive viewer does not exit immediately and drop the ZMQ observation stream. The sim subprocess **does not print to your terminal** by default (MuJoCo / server logs are discarded); use ``--sim-show-subprocess-output`` with ``--start-sim`` to stream sim stdout/stderr here. The sim process is **stopped when the agent session ends** (right after the ZMQ client disconnects, and again in a ``finally`` guard).

```bash
# Default table + same robot as agent (no sim YAML required):
uv run emet run agent --robot stretch --start-sim --command "describe the scene"

uv run emet run agent --robot rby1 --agent-config configs/agent_rby1_discord.yaml \
  --start-sim --command "What do you see?"
```

MolmoSpaces one-liner (wrapper + assets required):

```bash
uv run emet run agent --robot rby1 --start-sim --scene ithor --headless \
  --command "describe the scene"
```

## `kind` field

- `default_mujoco` — optional `scene_path` in YAML for a merged MJCF; otherwise the default table scene + `--robot`.
- `robocasa` — `robocasa_task`, `robocasa_style`, `robocasa_layout`, plus shared flags.
- `molmospaces` — `scene` (`ithor`, `procthor-10k`, …), `split`, `index`, optional `molmospaces_install`; merge runs before the server starts.

CLI equivalent: **`--scene robocasa`**, **`--scene ithor`**, or omit **`--scene`** for the default table (see [cli.md](cli.md)).

Shared flags on all kinds: `port_offset`, `headless`, `show_viewer_ui`, `no_cameras`, `use_glx`, `seed`, `steps`, `debug_molmospaces_spawn`, `verbose`.
