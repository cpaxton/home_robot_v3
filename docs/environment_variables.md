# Environment variables

Optional process-environment toggles for simulation, ZMQ clients, and MolmoSpaces. Most apps read these at startup; export in the shell before `emet serve` / `emet run`.

## MolmoSpaces

**[MolmoSpaces environment variables](molmospaces_environment_variables.md)** — spawn, autoplace, occupancy map, navigation teleport (`EMET_MOLMOSPACES_NAV_TELEPORT`), asset paths, and related test knobs.

**[MolmoSpaces spawn metadata](molmospaces_spawn_metadata.md)** — checked-in `molmospaces_spawn.json` per robot and `emet molmospaces write-spawn-metadata` (offline; not an env var).

See also [MolmoSpaces](molmospaces.md) for install and CLI usage.

## Configuration

| Variable | Used by | Purpose |
|----------|---------|---------|
| `EMET_CONFIG` | `emet run agent`, `emet run dynagraph`, `emet run dynamem`, `emet stream`, `emet capture` | Default path to unified nested config YAML when `--config` is omitted. Default: `configs/emet/default.yaml` in the repo. See [Unified EMET configuration](emet_config.md). |

## Benchmarks

Paper benchmark runbook: [paper_benchmarks.md](paper_benchmarks.md).

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_OVMM_OUTPUT_SIM` | `eval_ovmm_find_phases.py` | OVMM sim sweep output. Default `~/runs/emet/ovmm_find_phase` (`configs/ovmm/benchmark.yaml`). |
| `EMET_OVMM_OUTPUT_FULL` | `eval_ovmm_full.py` | Full OVMM (find + pick/place) output. Default `~/runs/emet/ovmm_full`. |
| `EMET_OVMM_OUTPUT_HABITAT` | `eval_habitat_ovmm_find_phases.py` | Habitat OVMM proxy output. Default `~/runs/emet/ovmm_habitat`. |
| `EMET_SQA3D_OUTPUT` | `emet sqa3d run-real-sweep`, `aggregate_sqa3d_sweep.py` | Default sweep output root. Default `~/runs/emet/sqa3d` (`configs/sqa3d/benchmark.yaml`). |
| `SQA3D_DATA_DIR` | `emet sqa3d`, `emet eval-sqa3d` | Root for SQA3D Zenodo JSON (`sqa_task/`, optional `localization_task/`). Default `~/.cache/sqa3d/data`. See [sqa3d.md](sqa3d.md). |
| `SCANNET_ROOT` | `emet sqa3d` embodied runs | ScanNet v2 download root (`scans/<scene_id>/…`). Default `~/.cache/scannet`. |
| `SCANNET_DOWNLOAD_SCRIPT` | `scripts/download_scannet_data.py` | Override path to `download-scannet.py`. |
| `SQA3D_MIN_FREE_MIB` | `scripts/run_sqa3d_gpu_sweep.sh` | Minimum free GPU MiB before starting a sweep (default `14000`). See [sqa3d_compute.md](sqa3d_compute.md). |
| `SQA3D_SCANNET_TEST_SCENE` | pytest `test_scannet_embodied_smoke` | Scene id for integration smoke (default `scene0380_00`). |
| `RUN_SQA3D_SCANNET_TESTS` | pytest | Set `1` to run embodied ScanNet integration smoke when mesh may be missing. |
| `EMET_EQA_VL_MODEL_SIZE` | EQA / GraphEQA VLM load | Override `eqa_vl.model_size` in `dynav_config.yaml` (legacy Qwen3.5 path; default EQA uses `eqa.vl_hf_model_id` **Qwen/Qwen3-VL-8B-Instruct** int4). |
| `EMET_SIGLIP_VERSION` | `get_shared_mask_siglip_encoder` (voxel map / dynagraph grounding) | Override the SigLIP checkpoint: `base`, `so400m` (default), `siglip2_base`, `siglip2_so400m`. A/B encoder upgrades without config edits. |
| `EMET_SIGLIP_DTYPE` | `SiglipEncoder` / `MaskSiglipEncoder` weight load | `float32` (default), `float16`, or `bfloat16`. Halves SigLIP VRAM (so400m: 3.5 GB → 1.75 GB); outputs are cast back to fp32 so stored features/thresholds are unchanged. int4/int8 unsupported (breaks the MaskSiglip head surgery). |
| `EMET_VLM_FRONTIER_SCORING` | Dynagraph EQA exploration (`controller_graph_eqa.py`) | Set `1` to let the EQA VLM pick the exploration frontier from candidate views (<=6 images/iteration) before the SigLIP-nearest heuristic. Only active in the dynagraph coverage-override path (`_eqa_explore_when_uncovered`); baseline `graph_eqa` is unaffected. Default off. |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch CUDA | Optional allocator hint (e.g. `expandable_segments:True`); set by `run_sqa3d_gpu_sweep.sh` if unset. |

## ZMQ and simulation (general)

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_ZMQ_STARTUP_TIMEOUT` | ZMQ clients, `emet run molmospaces-explore` | Seconds to wait for first observation (default 60). Documented in [molmospaces_environment_variables.md](molmospaces_environment_variables.md). |
| `EMET_NAVGRID_ASCII` | Dynamem / Dynagraph mapping | Terminal nav grid; see [dynagraph.md](dynagraph.md). |
| `EMET_NAVGRID_MAX_SIDE` | Nav grid ASCII | Default 320. |
| `EMET_NAVGRID_CONTEXTS` | Nav grid ASCII | Limit which hooks print. |
| `EMET_ROBOSUITE_POST_LOAD_DEBUG` | `robosuite_load_utils` | Post-load velocity / contact diagnostics after `RobosuiteZmqServer` startup. Set `1`/`true`/`yes`/`on`. |
| `EMET_MUJOCO_CTRL_DEBUG` | `robosuite_server` | Log stationary `ctrl` apply cycles (first N steps, then periodic). Set `1`/`true`/`yes`/`on`. |
| `EMET_MUJOCO_CTRL_DEBUG_VERBOSE` | `robosuite_server` | Full per-actuator `ctrl` lines (use with `EMET_MUJOCO_CTRL_DEBUG=1`). |
| `EMET_ROBOSUITE_AUTOPLACE` | `scene_base_spawn` | Planar base autoplace on Robocasa / default-table merges (default `1`). `0`/`false`/`no`/`off` disables. |
| `MUJOCO_GL` | MuJoCo rendering | e.g. `egl` for headless GPU cameras on Linux. RobosuiteZmqServer sets this automatically unless `--use-glx`. |
| `EMET_MARS_ONBOARD_DA3` | Innate Mars bridge | Set to `1` on the Jetson when `emet mars start --onboard-da3` runs DA3 stereo onboard (see [innate_mars_hardware.md](robots/innate_mars_hardware.md)). |
| `EMET_ROBOT_PASSWORD` | Deploy / Mars SSH | Optional password when not stored in the connection profile. |

See also [simulation_modules.md](simulation_modules.md) for maintainer-oriented module notes.

Add new cross-cutting `EMET_*` variables here or in a topic-specific doc and link from [simulation.md](simulation.md) / [README.md](../README.md) as appropriate.
