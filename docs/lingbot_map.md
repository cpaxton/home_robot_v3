# LingBot-Map (streaming depth + pose)

[LingBot-Map](https://github.com/Robbyant/lingbot-map) is a feed-forward streaming 3D reconstruction model (Geometric Context Attention). This repo uses an **isolated venv** (`.venv-lingbot-map`) because LingBot requires PyTorch 2.8 + FlashInfer, which conflicts with the main `emet` environment.

## Install

```bash
./scripts/install_lingbot_map.sh
# or: ./install.sh --lingbot-map -y
```

Download a checkpoint (recommended: **lingbot-map-long.pt**) from [Hugging Face](https://huggingface.co/robbyant/lingbot-map):

```bash
export LINGBOT_MAP_CHECKPOINT="$HOME/.cache/lingbot-map/lingbot-map-long.pt"
```

GPU: ~12–24 GB VRAM at 518×378 streaming resolution on an RTX-class GPU.

## Innate Mars sim demo (default table)

### 1. Record an episode (main venv)

Terminal 1:

```bash
uv run emet serve mujoco --robot innate_mars --headless
```

Terminal 2:

```bash
uv run python scripts/record_innate_mars_episode.py \
  --output logs/lingbot_episodes/mars_table_001
```

This runs `base_spin` (default): relative base rotation while saving RGB, sensor depth (GT), intrinsics, and `camera_pose` under `images/`, `depths/`, and `metadata.jsonl`. Use `--motion rotate_in_place` for the full Dynamem path.

### 2. Offline LingBot inference (lingbot venv)

```bash
.venv-lingbot-map/bin/python -m emet_lingbot_map infer \
  --episode logs/lingbot_episodes/mars_table_001 \
  --checkpoint "$LINGBOT_MAP_CHECKPOINT" \
  --output logs/lingbot_episodes/mars_table_001/lingbot \
  --keyframe-interval 2 \
  --use-sdpa
```

Use `--use-sdpa` unless FlashInfer + `ninja` are installed (see `./scripts/install_lingbot_map.sh`).

### 3. Evaluate vs sim GT + DA3 baseline (main venv)

```bash
uv run python scripts/lingbot_map_smoke.py \
  --episode logs/lingbot_episodes/mars_table_001 \
  --lingbot-output logs/lingbot_episodes/mars_table_001/lingbot \
  --rerun
```

Prints scale-aligned depth RMSE, trajectory ATE (after Sim(3) alignment), and optional DA3 depth baseline on the same RGB sequence.

### 4. Live ZMQ debug (optional)

```bash
uv run emet serve mujoco --robot innate_mars --headless
uv run emet debug-lingbot-depth --robot innate_mars --use-lingbot-pose
```

Uses a subprocess in `.venv-lingbot-map` for streaming inference.

### 5. DynaMem with LingBot depth (optional)

```bash
uv run emet run dynamem --robot innate_mars \
  --dynav-config dynav_innate_mars_lingbot.yaml -S --cpu-only
```

Requires `LINGBOT_MAP_CHECKPOINT` and `.venv-lingbot-map`.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `LINGBOT_MAP_CHECKPOINT` | Path to `.pt` weights |
| `LINGBOT_MAP_VENV` | Override path to `.venv-lingbot-map` (default: repo root) |
| `LINGBOT_MAP_USE_SDPA` | Set to `1` to skip FlashInfer (use PyTorch SDPA; default in smoke script and `dynav_innate_mars_lingbot.yaml`) |

## Decision notes (sim vs hardware)

- **Sim** provides sensor depth and sim `camera_pose` for quantitative A/B.
- **Hardware Mars** is RGB-only on ZMQ; LingBot is the intended depth+pose front-end once sim metrics look good.
- Head/arm motion during base rotation is acceptable for the initial `rotate_in_place` demo; general manipulation may need fixed-head navigation frames.
- **Sim nav:** default-table robosuite sim advertises `teleport_base` (set `EMET_SIM_NAV_TELEPORT=0` to force wheel drive).

See also: [Innate Mars robot doc](robots/innate_mars.md), [DA3 appendix](../paper/sections/appendix/03_depth_anything.tex).
