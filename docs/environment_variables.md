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

Paper benchmark runbook: [paper_benchmarks.md](paper_benchmarks.md). **Overnight smoke + diagnostics:** [evaluation.md](evaluation.md).

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_EVAL_EXPORT_MAP` | Habitat / OVMM / SQA3D episode bundles | Write `topdown_map.png` (default on). YAML: `eval.export_map`. Alias: `HABITAT_EQA_EXPORT_MAP`. |
| `EMET_EVAL_EXPORT_MAP_OVERLAY` | Habitat episode bundles | `topdown_map_overlay.png` (GT navmesh + agent map + trajectory; default on). YAML: `eval.export_map_overlay`. |
| `EMET_EVAL_EXPORT_MAP_VIDEO` | Same | `topdown_exploration.mp4` timelapse from stride map frames (default on). YAML: `eval.export_map_video`. |
| `EMET_EVAL_MAP_VIDEO_STRIDE` | Same | Steps between map frames when `EMET_EVAL_MAP_STRIDE=0` (default `5`). YAML: `eval.map_video_stride`. |
| `EMET_EVAL_EXPORT_VIDEO` | Same | Write `episode_rgb.mp4` (head camera; manifest `head_camera_mp4`). YAML: `eval.export_video`. Alias: `HABITAT_EQA_EXPORT_VIDEO`. |
| `EMET_EVAL_EXPORT_VIDEO_SUBSTEPS` | Habitat episode bundles | One RGB frame per Habitat `sim.step` during nav/rotate (default on). YAML: `eval.export_video_substeps`. |
| `EMET_EVAL_VIDEO_MOTION_PACED` | Same | Motion-paced MP4 encoding (default on). YAML: `eval.video_motion_paced`. |
| `EMET_EVAL_EXPORT_FRAMES` | Same | Save RGB frames under `frames/`. |
| `EMET_EVAL_EXPORT_OBSTACLE_GRIDS` | Same | `obstacles_2d.npy`, `explored_2d.npy`, `grid_meta.json` (default on). |
| `EMET_EVAL_EXPORT_TRAJECTORY` | Same | `trajectory.jsonl` (default on). |
| `EMET_EVAL_EXPORT_OBJECT_CROPS` | Same | Dynagraph object-crop mosaic when graph memory is present (default on). |
| `EMET_EVAL_MAP_STRIDE` | Same | Save intermediate maps every N steps (0 = final only). Alias: `HABITAT_EQA_MAP_STRIDE`. |
| `EMET_EVAL_EXPORT_GRAPH` | Same | Full graph checkpoint (heavy). Alias: `HABITAT_EQA_EXPORT_GRAPH`. |
| `EMET_EVAL_EXPORT_VOXEL_HISTORY` | Habitat episode bundles | Slim `observations_history.jsonl` from in-memory voxel frames (default on in Habitat runners). |
| `EMET_EVAL_EXPORT_VOXEL_PICKLE` | Same | Optional full `voxel_debug.pkl` via `write_to_pickle`. |
| `EMET_OVMM_OUTPUT_SIM` | `eval_ovmm_find_phases.py` | OVMM sim sweep output. Default `~/runs/emet/ovmm_find_phase` (`configs/ovmm/benchmark.yaml`). |
| `EMET_OVMM_OUTPUT_FULL` | `eval_ovmm_full.py` | Full OVMM (find + pick/place) output. Default `~/runs/emet/ovmm_full`. |
| `EMET_OVMM_OUTPUT_HABITAT` | `eval_habitat_ovmm_find_phases.py` | Habitat OVMM proxy output. Default `~/runs/emet/ovmm_habitat`. |
| `EMET_SQA3D_OUTPUT` | `emet sqa3d run-real-sweep`, `aggregate_sqa3d_sweep.py` | Default sweep output root. Default `~/runs/emet/sqa3d` (`configs/sqa3d/benchmark.yaml`). |
| `EMET_DYNAMIC_EXPLORE_OUTPUT` | `scripts/eval_dynamic_exploration.py` | Dynamic exploration sweep output. Default `~/runs/emet/dynamic_exploration` (`configs/benchmarks/dynamic_exploration.yaml`). |
| `EMET_DYNAMIC_EXPLORE_DYNAGRAPH_TIMEOUT_S` | `dynamic_exploration_runner.py` | Override per-run ``emet run dynagraph`` subprocess timeout (seconds). Default scales with explore budget (~105 min for Robocasa K=3 on GPU). |
| `EMET_DYNAMIC_EXPLORE_HEARTBEAT_S` | `dynamic_exploration_runner.py` | Heartbeat interval while a dynagraph subprocess is running (default `120`). Writes to stderr + `progress.jsonl`. |
| `EMET_DYNAMIC_EXPLORE_STALE_LOG_S` | `dynamic_exploration_runner.py` | Warn when `dynagraph.log` mtime is older than this many seconds (default `900`). Surfaces post-VLM / EQA hangs. |
| `EMET_DYNAMIC_EXPLORE_STALE_KILL_S` | `dynamic_exploration_runner.py` | Kill the dynagraph **process group** when the log is stale this long (default `2×` `STALE_LOG_S`). Prevents leaked GPU holders after wrapper-only `kill`. |
| `EMET_VL_GENERATE_HEARTBEAT_S` | `qwen3_vl_client.py` | Print `[vl] generate heartbeat` this often while `model.generate` runs (default `30`; `0` disables). Keeps eval log mtime fresh during long vision prefill. |
| `EMET_EQA_ANSWER_MAX_NEW_TOKENS` | `graph_memory.query_answer` | Cap `max_new_tokens` for the answer VLM call (default `256`; `0` uses client default). |
| `EMET_EQA_AGENTIC_VERIFY` | GraphEQA / dynagraph EQA | `1`/`0` — enable unified explore/navigate/verify/answer loop (`eqa.agentic_verify`). |
| `EMET_EQA_AGENTIC_ROUTER` | `AgenticEQAExecutor` | `1`/`0` — override `eqa.agentic_vlm_router`: let the shared Qwen3-VL pick tools via JSON tool calls (`0` = deterministic fallback only, for reproducible evals). |
| `EMET_EQA_TRACE` | `AgenticEQAExecutor` | `1` — append `agentic_trace.jsonl` (SigLIP embeds + GT) for offline tuning via `scripts/tune_agentic_verify.py`. |
| `EMET_EQA_QUESTION_TIMEOUT_S` | `controller_graph_eqa.run_eqa` | Per-question wall-clock cap for GraphEQA planning/nav loops (default `900`; `0` disables). |
| `EMET_SCENE_MAP_CACHE_DIR` | `scene_map_cache.py` | Root for prebuilt scene maps (graph + voxel). Default `~/.cache/emet/scene_maps`. |
| `EMET_USE_SCENE_MAP_CACHE` | OVMM find / dynamic explore | Load cached baseline and skip rotate/explore when present (default `1`). Set `0` or pass `--no-scene-cache`. |
| `EMET_SCENE_MAP_HF_REPO` | `scene_map_cache_hub.py` | HuggingFace dataset repo id for push/pull (e.g. `org/emet-scene-maps`). |
| `EMET_SCENE_MAP_CACHE_AUTO_PULL` | `ensure_cached_map` | On local miss, try HF pull when repo is set (default `1`). |
| `EMET_VLM_DEVICE_CHECK_MAX_PARAMS` | `vlm_device.assert_cuda_placement` | When `hf_device_map` is missing, cap parameter/buffer device sampling (default `512`). Avoids multi-hour stalls after int4 load. |
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
| `EMET_DYNAGRAPH_MCQ_DEBIAS` | Habitat `emet-habitat` dynagraph harness | `1` / `0` override `mcq_debias` after harness profile (CLI: `--mcq-debias` / `--no-mcq-debias`). |
| `EMET_DYNAGRAPH_MEMORY_SUMMARY` | Habitat `emet-habitat` dynagraph harness | `1` / `0` override CONFIRMED_MEMORY block (CLI: `--memory-summary` / `--no-memory-summary`). |
| `EMET_DYNAGRAPH_EXPLORE_UNCOVERED` | Habitat `emet-habitat` dynagraph harness | `off`, `on`, or `conservative` (CLI: `--explore-when-uncovered`). Default per harness: `habitat_eqa` uses `conservative` in [`configs/benchmarks/dynagraph.yaml`](../configs/benchmarks/dynagraph.yaml). |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch CUDA | Optional allocator hint (e.g. `expandable_segments:True`); set by `run_sqa3d_gpu_sweep.sh` and `run_sqa3d_sharded_sweep.sh` if unset. |
| `NEED_MIB` | `emet eval check/wait`, `scripts/gpu_preflight.sh` | Min free VRAM (MiB) (default `12000`). |
| `GPU_STABLE_CHECKS` | `emet eval wait` | Consecutive passing free-VRAM reads (default `3`). |
| `GPU_WAIT_INTERVAL` | `emet eval wait` | Seconds between reads (default `30`). |
| `GPU_SETTLE_SEC` | `emet eval kill-stale` | Sleep after pattern kills (default `15`). |
| `GPU_KILL_STALE` | `emet_gpu_between_steps` | Set `0` to skip process cleanup between overnight phases. |
| `EMET_GPU_PROTECT_PIDS` | `emet eval kill-stale` | Space-separated PIDs never killed (plus the caller process and its ancestors). |
| `EMET_JOBS_DIR` | `emet jobs` | Directory for job registry JSON (default `~/runs/emet/jobs`). |
| `EMET_JOB_ID` | smoke/queue scripts | If set by `emet jobs run`, scripts heartbeat via `emet jobs update` (and skip creating a new registry entry). Also write `OUT/progress.json` for ETA even without a job id. |

### Large paper eval orchestrator

Used by `scripts/run_large_paper_eval.sh` and `scripts/run_sqa3d_sharded_sweep.sh`. See [paper_benchmarks.md](paper_benchmarks.md) (Large paper eval queue).

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_LARGE_EVAL_LOG_DIR` | `run_large_paper_eval.sh` | Master + per-phase logs. Default `~/runs/emet/large_eval`. |
| `EMET_SQA3D_OUTPUT` | SQA3D sweeps / sharded sweep | Same as benchmarks table above. |
| `SQA3D_GPUS` | `run_large_paper_eval.sh` | Comma-separated CUDA ids (e.g. `0,1,2,3`) → `run_sqa3d_sharded_sweep.sh` (~linear speedup). |
| `SQA3D_NO_ISOLATE` | `run_large_paper_eval.sh` | Set `1` for in-process batch (`--no-isolate-episodes`; faster, OOM risk). |
| `SQA3D_METHODS` | `run_large_paper_eval.sh` | Space-separated methods (default `dynagraph dynamem`). e.g. `SQA3D_METHODS=dynagraph`. |
| `SKIP_SQA3D_TEST` | `run_large_paper_eval.sh` | Set `1` to skip test-split SQA3D sweeps (val only). |
| `SKIP_OVMM` | `run_large_paper_eval.sh` | Set `1` to skip OVMM find replicates. |
| `SKIP_DYNAMIC_EXPLORE` | `run_large_paper_eval.sh` | Set `1` to skip dynamic exploration matrix. |
| `OVMM_CPU_ONLY` | `run_large_paper_eval.sh` | Set `1` for `--cpu-only` OVMM (overlap with SQA3D on GPU in another terminal). |
| `DYNAMIC_EXPLORE_CPU_ONLY` | `run_large_paper_eval.sh` | Set `1` for `--cpu-only` dynamic exploration runs. |

## ZMQ and simulation (general)

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_ZMQ_STARTUP_TIMEOUT` | ZMQ clients, `emet run molmospaces-explore` | Seconds to wait for first observation (default 60). Documented in [molmospaces_environment_variables.md](molmospaces_environment_variables.md). |
| `EMET_ZMQ_TIMING` | `BaseZmqServer` | `1` — print periodic SEND/RECV timing lines. Default off (also enabled by server `--verbose`). |
| `EMET_NAVGRID_ASCII` | Dynamem / Dynagraph mapping | Terminal nav grid; see [dynagraph.md](dynagraph.md). |
| `EMET_NAVGRID_MAX_SIDE` | Nav grid ASCII | Default 320. |
| `EMET_NAVGRID_CONTEXTS` | Nav grid ASCII | Limit which hooks print. |
| `EMET_ROBOSUITE_POST_LOAD_DEBUG` | `robosuite_load_utils` | Post-load velocity / contact diagnostics after `RobosuiteZmqServer` startup. Set `1`/`true`/`yes`/`on`. |
| `EMET_MUJOCO_CTRL_DEBUG` | `robosuite_server` | Log stationary `ctrl` apply cycles (first N steps, then periodic). Set `1`/`true`/`yes`/`on`. |
| `EMET_MUJOCO_CTRL_DEBUG_VERBOSE` | `robosuite_server` | Full per-actuator `ctrl` lines (use with `EMET_MUJOCO_CTRL_DEBUG=1`). |
| `EMET_ROBOSUITE_AUTOPLACE` | `scene_base_spawn` | Planar base autoplace on Robocasa / default-table merges (default `1`). `0`/`false`/`no`/`off` disables. |
| `MUJOCO_GL` | MuJoCo rendering | e.g. `egl` for headless GPU cameras on Linux. RobosuiteZmqServer sets this automatically unless `--use-glx`. |
| `EMET_MARS_ONBOARD_DA3` | Innate Mars bridge | Set to `1` on the Jetson when `emet mars start --onboard-da3` runs DA3 stereo onboard (see [innate_mars_hardware.md](robots/innate_mars_hardware.md)). |
| `EMET_MARS_DA3_SPECKLE_OPEN_KERNEL` | Onboard Jetson DA3 | Morphological depth speckle filter kernel (pixels); `0` = off (default). Same helper as dynav `filters.depth_speckle_open_kernel`. |
| `EMET_MARS_DA3_SPECKLE_OPEN_ITERATIONS` | Onboard Jetson DA3 | Speckle opening repeat count (default `1`). Ignored when kernel is `0`. |
| `EMET_STREAM_VERBOSE` | `emet capture` / `emet stream` | `1`/`true` → per-step status + DA3 INFO timing (same as `--verbose`). Default off; dynav `stream.verbose` also applies. |
| `DA3_LOG_LEVEL` | Depth Anything 3 (`depth_anything_3`) | `WARN` (default during stream when not verbose), `INFO`, `ERROR`. Dynav `stream.da3_log_level` sets default via `configure_mapping_session_logging`. |
| `EMET_ROBOT_PASSWORD` | Deploy / Mars SSH | Optional password when not stored in the connection profile. |

See also [simulation_modules.md](simulation_modules.md) for maintainer-oriented module notes.

## Agent (`emet run agent`)

| Variable | Where used | Notes |
|----------|------------|-------|
| `EMET_VL_CACHE_SYSTEM_PREFIX` | `qwen3-vl-eqa` / `Qwen3VLClient` | `1`/`0` — cache system-prompt KV across agent turns (default on via `eqa.vl_cache_system_prefix`). CLI: `--cache-vl-prefix` / `--no-cache-vl-prefix`. |
| `EMET_ALLOW_CPU_VLM` | Qwen3-VL / Gemma VLM / Qwen2.5-VL load | `1` — allow silent CPU bf16 fallback when GPU int4 load fails. **Default off**: agent refuses CPU fallback (multi-minute “Thinking…” hangs). |
| `EMET_HF_LOCAL_ONLY` | VL / SigLIP `from_pretrained` | `1` — require local HF cache only (same idea as `HF_HUB_OFFLINE=1`). Warm cache is preferred automatically even when unset. |
| `EMET_VL_PREFIX_GENERATE_TIMEOUT_S` | `Qwen3VLClient` | Soft warn + disable prefix cache after a slow prefix-KV generate (default `45`). |
| `EMET_ATTN_EAGER` | `Qwen3VLClient` | `1` — force eager attention (debug). |
| `EMET_REQUIRE_FLASH_ATTN` | `attn_impl.resolve_attn_implementation` | `1`/`0` — require Flash-Attn 2 on CUDA VL loads. **Default on** when unset (fail loud instead of silent SDPA). |
| `EMET_ALLOW_SDPA_ATTN` | same | `1` — permit PyTorch SDPA when flash-attn is missing (slower; long multi-image EQA can crawl). Overrides the default require-flash policy. |
| `eqa.vl_image_max_side` | VL clients / `describe_scene` | Longest RGB edge before VL/detector (default `512`; `0` = no resize). Override: `--set eqa.vl_image_max_side=384`. |
| `eqa.vl_image_max_pixels` | Same | Optional `H*W` cap after side resize (`0` = off). |
| `EMET_AGENT_THINKING_STATUS` | Agent loop | `1`/`0` — emit `*Thinking…*` status lines (default on). CLI: `--thinking-status` / `--no-thinking-status`. Heartbeats every ~8s while the LLM is still running. |
| `EMET_AGENT_MODEL_DEBUG` | Agent / VL clients | `1` — model stack + prefix-cache hit logs + generate phase timings. CLI: `--debug-models`. |
| `EMET_AGENT_TOOL_DEBUG` | Agent loop | `1` — verbose tool I/O. CLI: `--debug-tools`. |
| `EMET_AGENT_CAMERA_DEBUG` | Agent tools | `1` — head-camera frame stats. CLI: `--debug-camera`. |
| `EMET_AGENT_MOTION_STATUS` | Controllers | `1`/`0` — fine-grained terminal progress for head sweeps / rotate-in-place / explore steps (default **on**). Discord still only gets coarse milestones (`*Look around: sweeping head*`, mid/end scan). |

See [AGENT_RUN.md](AGENT_RUN.md).

Add new cross-cutting `EMET_*` variables here or in a topic-specific doc and link from [simulation.md](simulation.md) / [README.md](../README.md) as appropriate.
