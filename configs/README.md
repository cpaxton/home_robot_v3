# Example agent configs

Unified nested config: see [docs/emet_config.md](../docs/emet_config.md). Default: [`configs/emet/default.yaml`](emet/default.yaml).

## `agent_innate_mars.yaml`

Thin preset: `extends: configs/emet/default.yaml`, `robot: innate_mars`, embodied-agent overlays.

```bash
uv run emet serve mujoco --robot innate_mars --headless   # terminal 1
uv run emet run agent --config configs/agent_innate_mars.yaml --rerun   # terminal 2
```

## `agent_rby1_discord.yaml`

DynaMem / mapping hyperparameters for **rby1** + **`emet run agent`** + **Discord**.

1. One-time: `uv sync -e discord` and create a Discord bot token; invite the bot to your server with message read/send permissions.
2. Create a text channel (default name **`talk-to-stretch`**) or set `EMET_DISCORD_CHANNEL` to your channel name.
3. Start the sim, then the agent — see the comment block at the top of `agent_rby1_discord.yaml`.

From repo root:

```bash
export DISCORD_TOKEN="your-token"
uv run emet serve mujoco --robot rby1 --headless   # terminal 1
# --robot-ip defaults to 127.0.0.1; omit --vl-include-camera (on by default for *VL* LLMs)
uv run python -m emet.app.run_agent --robot rby1 \
  --device cuda --agent-config configs/agent_rby1_discord.yaml --discord   # terminal 2 (default --llm qwen3-vl-eqa; subprocess sets PYTORCH_ALLOC_CONF=expandable_segments:True unless preset)
```

Local LLM only (no sim): `uv run python -m emet.app.run_agent --offline`

Use `--input-path` to load a saved memory directory when you have one.
