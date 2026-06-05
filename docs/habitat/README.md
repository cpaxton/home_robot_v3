# Habitat EQA harness

Reproduce **GraphEQA-style** HM-EQA evaluation in Habitat-Sim while driving emet `GraphEQAMemory` / `DynagraphController`.

| Doc | Contents |
|-----|----------|
| [install.md](install.md) | `.venv-habitat`, micromamba, `habitat-sim` from `aihabitat-nightly` |
| [data.md](data.md) | HM-EQA CSVs, HM3D downloads, Matterport API tokens, on-disk layout |
| [usage.md](usage.md) | CLI, `emet run graph-eqa-habitat`, methods, tests |
| [troubleshooting.md](troubleshooting.md) | Common errors (missing GLB, unauthorized download, wrong paths) |

**Engineering plan:** [docs/plans/HABITAT_EQA_HARNESS.md](../plans/HABITAT_EQA_HARNESS.md)  
**Branch:** `feature/habitat-eqa-harness`

## Quick start

From the repo root:

```bash
# 1. Isolated Habitat env (Python 3.10 + habitat-sim via conda)
./scripts/install_habitat.sh

# 2. HM-EQA question CSVs (~500 rows; small)
uv run python scripts/download_habitat_eqa_data.py --fetch-csv

# 3. HM3D scenes — see data.md for Matterport API tokens (not your web login)
export MATTERPORT_USERNAME='<token-id>'
export MATTERPORT_PASSWORD='<token-secret>'
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d train

# 4. Confirm question 0 scene is on disk
uv run python scripts/download_habitat_eqa_data.py --verify-question 0
uv run emet habitat info

# 5. Smoke episode (mocked LLM; no API key)
uv run emet run graph-eqa-habitat --mock-llm --question-id 0 --method dynagraph
```

**No Matterport access yet?** Download the free example pack to test the install only (~150MB; does **not** include HM-EQA scenes):

```bash
uv run python scripts/download_habitat_eqa_data.py --fetch-hm3d example
```

## Architecture (short)

1. **HabitatEQASimulator** — load HM3D `.basis.glb`, RGB-D agent  
2. **HabitatRobotClient** — `AbstractRobotClient` shim (`get_observation`, `move_base_to`, …)  
3. **GraphEQAController / DynagraphController** — emet memory stack (unchanged)  
4. **EQAExecuter** — `run_eqa` loop + multiple-choice grading  

Observation poses use `convert_pose_habitat_to_opencv` in `src/emet/utils/pose.py`.

## Code layout

| Path | Role |
|------|------|
| `src/emet/habitat/` | Config, datasets, metrics (main `.venv`) |
| `packages/emet_habitat/` | Habitat-Sim sim + runner (`.venv-habitat`) |
| `scripts/install_habitat.sh` | Bootstrap micromamba + `.venv-habitat` |
| `scripts/download_habitat_eqa_data.py` | CSV + HM3D fetch helpers |
