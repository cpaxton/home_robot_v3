# Habitat EQA harness

Reproduce **GraphEQA-style** HM-EQA evaluation in Habitat-Sim while driving emet `GraphEQAMemory` / `DynagraphController`.

| Doc | Contents |
|-----|----------|
| [install.md](install.md) | `.venv-habitat`, micromamba, `habitat-sim` from `aihabitat-nightly` |
| [data.md](data.md) | HM-EQA CSVs, HM3D downloads, Matterport API tokens, on-disk layout |
| [usage.md](usage.md) | CLI, `emet run graph-eqa-habitat`, methods, tests |
| [vlm_bakeoff.md](vlm_bakeoff.md) | VLM model bake-off reproduction (canonical-6, balanced-31) |
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

# 3. HM3D scenes — Profile → Settings → Developer Tools (not web login)
#    https://my.matterport.com/settings/account/devtools
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

### HabitatRobotClient notes

In-process shim at `packages/emet_habitat/emet_habitat/robot_client.py` (full docstrings on the class).

| API | Behavior |
|-----|----------|
| `move_base_to` / `execute_trajectory` / `navigate_to` | Goals are Habitat **world** `(x, z, yaw)`. `world_frame` is accepted for ZMQ API parity but **ignored**. |
| `get_emet_session` / `set_emet_session` | Optional local session dict (e.g. `sim_object_placements` for find-phase GT graph refresh). Not streamed from a server. |
| `hm3d_semantic_labeler` / `uses_hm3d_semantics` | Present when the simulator was built with HM3D semantic sensors. |
| Arm / gripper / head methods | Stretch-shaped **no-op stubs** so DynaMem manipulation wrappers import cleanly; EQA uses navigation only. |

Observation poses use `convert_pose_habitat_to_opencv` in `src/emet/utils/pose.py`.

## Code layout

| Path | Role |
|------|------|
| `src/emet/habitat/` | Config, datasets, metrics (main `.venv`) |
| `packages/emet_habitat/` | Habitat-Sim sim + runner (`.venv-habitat`) |
| `scripts/install_habitat.sh` | Bootstrap micromamba + `.venv-habitat` |
| `scripts/download_habitat_eqa_data.py` | CSV + HM3D fetch helpers |
