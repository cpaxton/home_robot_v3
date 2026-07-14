# Your first test: interactive agent in simulation

Talk to the embodied agent with natural language (`emet run agent`) in MuJoCo.
This walkthrough uses **two terminals**: one for the sim, one for the agent.

**Prerequisites**

- Project installed with sim: `./install.sh -y --sim` (or `./install.sh -y --profile=full`)
- NVIDIA GPU with ~8+ GiB free for the default text chat model (`qwen35-4B`); ~12+ GiB if using `--llm qwen3-vl-eqa`
- From the **repo root**, prefer `uv run emet …`

Optional for home scenes: MolmoSpaces wrapper (`.venv-molmospaces`). Default install creates it when `packages/emet_molmospaces` exists; otherwise `./install.sh --molmospaces -y`.

---

## Path A — default table (fastest)

Good first smoke: a small table scene with a **red cylinder** and **blue cube**.

### Terminal 1 — sim

```bash
uv run emet serve mujoco --robot stretch --headless
```

Wait until you see the server running / ZMQ ports ready.

### Terminal 2 — agent

```bash
uv run emet run agent --robot stretch --no-discord --rerun
```

Open Rerun (optional): [http://localhost:9090?url=ws://localhost:9877](http://localhost:9090?url=ws://localhost:9877)

You should see a greeting and a `You:` prompt. Status lines like `*Thinking…*` mean the chat model is choosing tools (often ~10–30s text-only; much slower if you pass `--vl-include-camera`).

Vision questions (`what can you see?`) use `describe_scene`: **caption the head camera** with the **larger EQA VLM** (`--eqa` → Qwen3-VL-8B), optionally **ground with scene-graph/map**, attach a photo — no auto look-around. Chat stays on the fast `qwen35-4B` tool router; do not use the 8B as the default `--llm` just for routing. They should **not** dump low-confidence YoloE proposals into chat, and should **not** run a second full VL summarize pass after tools. Default memory stack is **Dynagraph** (`--memory-backend dynagraph`).

### Things to try (table)

| Say this | What should happen |
|----------|--------------------|
| `describe the scene` | VLM/memory description + often a camera image (`send_image`) |
| `what can you see?` | Same idea — vision tools, not raw YoloE class dumps |
| `find the red cylinder` | Navigate / localize toward the red cylinder |
| `find the blue cube` | Same for the blue cube |
| `is there a red cylinder on the table?` | Scene / memory style yes–no |
| `send me a picture` | Camera frame to the terminal (and Discord if enabled) |
| `show me the map` | Top-down map snapshot if mapping has run |
| `look around` / `scan` | Rotate in place (`scan_environment`), then often describe + photo |
| `explore` | Navigate to build map / memory |

Type `quit` (or `Q`) to exit.

**One-terminal scripted smoke** (no interactive prompt):

```bash
uv run emet run agent --robot stretch --start-sim --no-discord \
  -c "describe the scene" -c "find the red cylinder"
```

---

## Path B — MolmoSpaces iTHOR (home scenes)

Richer rooms for **find the Z** and **is an X on a Y** questions. Needs `.venv-molmospaces`.

### Terminal 1 — MolmoSpaces sim

```bash
# Stretch in iTHOR FloorPlan1 (index 0). First run may download scene assets.
uv run emet serve mujoco --scene ithor --split train --index 0 \
  --robot stretch --headless --install-scene-if-missing
```

Alternatives:

```bash
# Same scene via shortcut
uv run emet serve molmospaces --index 0 --robot stretch --headless

# Galaxea R1 instead of Stretch
uv run emet serve mujoco --scene ithor --split train --index 0 \
  --robot rby1 --headless --install-scene-if-missing
```

### Terminal 2 — agent

```bash
# Match --robot to the serve command
uv run emet run agent --robot stretch --no-discord --rerun
```

Or one terminal with `--start-sim`:

```bash
uv run emet run agent --robot stretch --start-sim \
  --scene ithor --split train --index 0 --install-scene-if-missing \
  --headless --no-discord --rerun
```

### Suggested flow

1. **Look around** — `describe the scene` / `what can you see?`
2. **Build a bit of map** — `explore` or `scan the environment`
3. **Find something** — `find the sofa`, `find the fridge`, `find the bed`
4. **Ask spatial / EQA-style questions** — see table below
5. **Map / photo** — `show me the map`, `send me a picture`

### Things to try (iTHOR)

Object names vary by floor plan. If a find fails, ask `what can you see?` or `list scene relations` and reuse labels from the reply.

**Find the Z**

- `find the sofa`
- `find the television`
- `find the fridge`
- `find the bed`
- `find the dining table`
- `go to the kitchen`

**Is an X on a Y?**

- `is there a remote on the coffee table?`
- `is a laptop on the desk?`
- `is there a plant on the floor?`
- `is a pillow on the bed?`
- `is there a mug on the counter?`

**Where / what**

- `where is the sofa?`
- `what is near the TV?`
- `what objects are on the table?`
- `list scene relations`

**Optional heavier EQA** (extra VRAM / slower startup):

```bash
uv run emet run agent --robot stretch --eqa --no-discord --rerun
```

Then try: `is there a lamp next to the bed?` / `where is the microwave?`

---

## Tips

| Tip | Detail |
|-----|--------|
| **Robot must match** | Same `--robot` on `serve` and `run agent` (`stretch`, `rby1`, …) |
| **Keep camera off the chat VL** | Default is fast. Only add `--vl-include-camera` if you want the LLM to see pixels directly (slow). |
| **VL image size** | `eqa.vl_image_max_side: 512` (default) downsamples RGB before VL/detector. Override: `--set eqa.vl_image_max_side=384`. |
| **Cap generation** | Default `--max-tokens 256`. Do not raise this for interactive tool use. |
| **First LLM call still costs** | ~2k-token tool system prompt; later turns reuse prefix KV cache. |
| **Disable Discord noise** | `--no-discord` (or unset `DISCORD_TOKEN`) |
| **GPU busy** | `./scripts/gpu_preflight.sh --kill-stale` then `NEED_MIB=12000 ./scripts/gpu_preflight.sh --wait` |
| **Wrong `emet` binary** | Always `uv run emet …` from this repo |
| **Restart after pulling speed fixes** | Kill the old `run_agent` process so new defaults apply |

More detail: [Agent run](AGENT_RUN.md), [MolmoSpaces](molmospaces.md), [Simulation configs](sim_configs.md), [CLI](cli.md).
