# Habitat EQA harness

Reproduce **GraphEQA-style** HM-EQA evaluation in Habitat-Sim while driving **emet** `GraphEQAMemory` / `DynagraphController`.

**Branch:** `feature/habitat-eqa-harness`  
**Plan:** [docs/plans/HABITAT_EQA_HARNESS.md](plans/HABITAT_EQA_HARNESS.md)

## Install

Habitat-Sim conflicts with the main MuJoCo stack (numpy / Python versions) and **has no Linux wheels on PyPI**. Use the isolated micromamba env:

```bash
./scripts/install_habitat.sh
```

This bootstraps **micromamba** (into `.micromamba/` if needed), creates `.venv-habitat` with **Python 3.10**, installs `habitat-sim` from **`aihabitat-nightly`** (headless + bullet), then editable `emet` + `emet-habitat`.

Requires: `curl`, network, ~2GB download for conda packages. No system `conda` required.

## Data

```bash
# Print HM3D + CSV download instructions
uv run python scripts/download_habitat_eqa_data.py --instructions

# Fetch HM-EQA questions + scene_init_poses only
uv run python scripts/download_habitat_eqa_data.py --fetch-csv
```

Defaults:

| Env var | Default |
|---------|---------|
| `HABITAT_EQA_DATA_DIR` | `~/.cache/habitat_eqa/data` |
| `HM3D_SCENE_DIR` | `~/.cache/habitat_eqa/hm3d/train` |

HM3D GLB layout: `<HM3D_SCENE_DIR>/<scene_id>/<scene_id>.basis.glb`

## Run

From main env (delegates to `.venv-habitat`):

```bash
uv run emet run graph-eqa-habitat --dataset hmeqa --question-id 0 --mock-llm
uv run emet run graph-eqa-habitat --method dynagraph --question-id 0 --mock-llm
```

Direct wrapper:

```bash
.venv-habitat/bin/emet-habitat info
.venv-habitat/bin/emet-habitat list-questions --limit 5
.venv-habitat/bin/emet-habitat run-episode --question-id 0 --method dynagraph --mock-llm
```

CLI group:

```bash
emet habitat info
emet habitat list-questions
```

## Methods

| `--method` | Config |
|------------|--------|
| `graph_eqa` | `dynagraph_merge_xy_m=0`, `dynagraph_staleness_horizon=0` |
| `dynagraph` | default merge + staleness |

## Architecture

1. **HabitatEQASimulator** — HM3D scene + RGB-D agent  
2. **HabitatRobotClient** — `AbstractRobotClient` shim (`get_observation`, `move_base_to`, …)  
3. **GraphEQAController / DynagraphController** — unchanged emet memory stack  
4. **EQAExecuter** — `run_eqa` loop + MCQ grading  

Observation poses use `convert_pose_habitat_to_opencv` in `src/emet/utils/pose.py`.

## Tests

```bash
uv run emet test src/test/habitat/ -q
RUN_HABITAT_TESTS=1 uv run emet test src/test/habitat/ -k smoke
```

Full GPU HM-EQA sweeps are not default CI.
