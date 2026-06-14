# Running the embodied agent (`emet run agent`)

## Flow

1. Start simulation (or connect to a real robot), e.g. `emet serve mujoco --robot rby1 --headless`.
2. Run the agent: **`--robot-ip` defaults to `127.0.0.1`**, so `uv run python -m emet.app.run_agent --robot rby1` connects to localhost. Use **`--offline`** for local LLM chat only (no ZMQ, wrong prompt for “robot” questions—no tools).
3. Optional: load a saved DynaMem map with `--input-path <logs/memory_xxx>`.
4. Optional: Discord (`uv sync -e discord`, **`DISCORD_TOKEN` required**): `uv run python -m emet.app.run_agent --discord`. If `DISCORD_TOKEN` is missing, a warning is printed and the bot does not start. Terminal and Discord messages share one agent loop when the bot runs.

## Config and models

- **Scene / mapping YAML**: `--agent-config` (default `dynav_config.yaml`). Use a basename under `src/emet/config/` or a path to your own YAML. Key reference: [Dynav configuration](dynav_config.md) (`map_boundary`, depth source, motion planner, EQA).
- **Chat / tool backbone**: `--llm` (default `qwen3-vl-eqa`: one Qwen3-VL-8B int4 from `dynav_config.yaml` `eqa:` for chat + head camera). `emet run agent` sets `PYTORCH_ALLOC_CONF=expandable_segments:True` in the agent subprocess unless you already exported `PYTORCH_ALLOC_CONF` (reduces fragmentation vs SigLIP/detector). With **`--eqa`**, **`--share-memory-vllm`** (default) reuses that same VLM for DynaMem captions/answers. Text-only fallback: `qwen35-4B` / `qwen35-9B`. For VL models, **robot RGB is passed by default** on each new user turn; use **`--no-vl-camera`** to disable.
- **Device / length**: `--device`, `--max-tokens` apply to embodied and `--offline` modes.

## Testing

- Config resolution: `uv run emet test src/test/utils/test_resolve_config.py`
- VL registry: `uv run emet test src/test/llms/test_qwen_vl_registry.py`
- CLI defaults: `uv run emet test src/test/cli/test_run_agent_defaults.py`
- Manual: with sim up, `timeout 15 uv run python -m emet.app.run_agent --no-llm -c Q` (letter-mode smoke).
