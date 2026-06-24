# MolmoBot-Data in emet

[MolmoBot-Data](https://huggingface.co/datasets/allenai/molmobot-data) stores expert manipulation
trajectories as batched HDF5 files with sibling MP4 camera feeds. Format details:
[allenai/molmospaces data_format.md](https://github.com/allenai/molmospaces/blob/main/docs/data_format.md).

## Inspect

```bash
uv sync
uv run emet dataset molmobot inspect /path/to/RBY1PickAndPlaceDataGenConfig/part0/train
uv run emet dataset molmobot list-trajs /path/to/trajectories_batch_0_of_1.h5
```

## Export (LeRobot-oriented JSONL)

```bash
uv run emet dataset molmobot export-lerobot \
  --src /path/to/RBY1PickAndPlaceDataGenConfig/part0/train \
  --out ./data/lerobot_rby1_pick \
  --task pick \
  --max-episodes 10
```

Output layout: ``episode_NNN/metadata.jsonl`` + ``episode.json``. Map fields to hello-robot
[LeRobot](https://github.com/hello-robot/lerobot) ``stretch-act`` preprocessing as needed.

## Sim replay

Terminal A:

```bash
uv run emet serve mujoco --scene ithor --robot rby1 --headless
```

Terminal B:

```bash
uv run emet dataset molmobot replay \
  --h5 /path/to/trajectories_batch_0_of_1.h5 \
  --traj-key traj_0 \
  --robot rby1 \
  --report /tmp/replay_metrics.json
```

Or: ``uv run python scripts/replay_molmobot_trajectory.py ...``

**Note:** H5 scenes may not match your merged MJCF 1:1; use replay for action-dimension /
ZMQ wiring checks until scene metadata alignment is implemented.

## Policy server (optional)

Install upstream MolmoBot, set ``MOLMOBOT_ROOT``, then use the wrapper package:

```bash
./install.sh --molmobot -y   # if install script extended; or manual venv
export MOLMOBOT_ROOT=/path/to/MolmoBot/MolmoBot
emet-molmobot serve-policy --hf-repo allenai/MolmoBot-DROID --action-type joint_pos
```

Package: [packages/emet_molmobot](../../packages/emet_molmobot/).

## Robots aligned with MolmoBot data

| MolmoBot platform | Emet robot id |
|-------------------|---------------|
| Rainbow RB-Y1 | `rby1` / `galaxea_r1` |
| Franka FR3 | `franka_fr3` |
| Mobile experiments (community) | `xlerobot`, `innate_mars`, `stretch` |

See [supported_robots.md](../robots/supported_robots.md).
