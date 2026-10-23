# Observed TAMP discovery diagnostics — October 14

The observed task provider is connected to `run agent`'s existing open-vocabulary
RGBD pipeline. Discovery uses single-frame segmented depth points and original
local receive times. Candidates remain discovery-only; observed planning and
execution are unavailable pending geometry and verification providers.

## Broader regression findings

The broader agent/config/camera-layout checks passed: 43 tests across
`test_agent_prompt_and_tools.py`, `test_agent_llm_multiturn.py`,
`test_agent_manip_tool_sequence.py`, `test_embodied_agent_config.py`, and
`test_observation_zmq_camera_layout.py`.

Inspection of the real detector path found and fixed:

- The CPU preset requested OWL-SAM as primary, but the processor skipped directly
  to its SAM2 fallback. Explicit primary selection is now honored.
- OWL-SAM returns instance class IDs and prompt texts rather than a `labels` list.
  The processor now maps class IDs back to prompt labels in detection order.
- SAM2 constructed two models; the first used the default device even when CPU
  was requested. It now constructs one model on the chosen device and shares it
  between box prediction and automatic masks. OWL-SAM forwards its device.
- Empty OWL detection/mask results used an all-zero instance map, which could
  become a whole-image object. Empty results now use background ID `-1`.

The detector/discovery/scene-graph regression batch passed 35 tests. Model
construction is controlled in the device test; it verifies no accidental CUDA
selection without loading a model. The real-model diagnostics below are separate.
Ruff and diff checks passed. XML outputs are `/tmp/tamp-broader-agent-tests.xml`
and `/tmp/tamp-detector-regression-tests.xml`.

## Actual CPU detector and camera diagnostics

Both diagnostics use actual OWL-SAM (`google/owlv2-base-patch16`, threshold 0.05),
the existing SAM2 small checkpoint, software-rendered live Stretch RGBD, and the
public `scene_tasks` tool. Vocabulary is `blue cube`, `red cylinder`, `table`.
Segmentation consumes image pixels; depth and camera calibration come through
ZMQ. Simulator body identities/poses are not inputs to proposals. Appearance
embeddings are disabled in this component diagnostic. There is no agent LLM,
manipulation, navigation task score, or exploration acceptance.

Local GPU use is disabled. CPU affinity is `0-7,12-31`; rendering uses OSMesa and
llvmpipe. Caliban's Qwen endpoint is not an OWL-SAM segmentation service, so this
specific detector diagnostic runs on CPU. The model cache is isolated under
`/tmp/emet-observed-model-cache`. All launches use `emet jobs`.

| Job | Artifact root under `/home/cpaxton/runs/emet` | Outcome |
| --- | --- | --- |
| `20261014_121222_85a0da` | `observed-tamp-camera-cpu-20261014` | Failed physics-progress watchdog after server connection/head command, before image segmentation. |
| `20261014_121625_066dbd` | `observed-tamp-camera-cpu-v2-20261014` | Completed capture/segmentation; public discovery rejected stale evidence (16.5 s old). Four internal proposals had incorrect blue-cube support labels. |

The first run raises `TimeoutError: live physics stopped advancing` from
`LiveTableRobot.step(1.)`, whose wall-time watchdog is 30 seconds. This establishes
failure of the progress gate; it does not establish whether physics completely
stopped or advanced too slowly under CPU rendering. Shutdown `BrokenPipeError`
messages occur afterward. No discovery score exists for that attempt.

The second run is explicitly narrower and cannot replace the failed progress
trial as a live-agent pass. Actual segmentation took 16.4898 seconds; receive age
at the tool call was 16.5123 seconds, exceeding the unchanged two-second gate.
`scene_tasks` returned `object_observation_stale`. It produced no public tasks.
The head command returned `None`, which is not measured arrival evidence. Its `protocol.json` pins the source SHA, dirty-diff
hash, detector configuration and scope. Inputs/output and timings are retained
under its artifact root when available. Native runtime/performance and physical
task completion remain separate from discovery-contract evidence.


## Frozen-frame detector replay

Job `20261014_122000_d42c6e` completed under
`observed-tamp-frozen-detector-20261014`. It replays the retained image with the
same actual detector and retains masks, labels, scores and a numbered preview.
It does not renew the old frame's receive time or claim live acceptance.

Inference took 11.7968 seconds. The detector produced six regions:

| Region | Label | Score | Image extent | Interpretation from retained image |
| --- | --- | ---: | --- | --- |
| 0 | blue cube | 0.1414 | x 0–239, y 249–395 | Broad table/front region, incorrect label |
| 1 | blue cube | 0.0870 | x 0–239, y 151–329 | Broad tabletop region, incorrect label |
| 2 | blue cube | 0.2391 | x 120–135, y 284–308 | Part of actual cube |
| 3 | blue cube | 0.3258 | x 120–134, y 309–325 | Another part of actual cube |
| 4 | red cylinder | 0.1079 | x 120–133, y 261–277 | Part of actual cylinder |
| 5 | red cylinder | 0.0611 | x 120–134, y 250–260 | Another part of actual cylinder |

No `table` detection survived this diagnostic configuration. The geometry
heuristic found broad horizontal patches in the two wrongly labeled regions and
proposed them as supports. The semantic error originates in detection; the
provider currently does not independently verify the label. Faster inference
alone would therefore not establish reliable discovery. Raising the score
threshold on this single frame is not claimed as a solution or acceptance.

Next: validate semantic object/support grounding from pixels, deduplicate parts
into object identities, and host perception where processing can meet the
freshness budget. Then rerun frozen frames and fresh live-camera trials with
independent geometric/task checks. Caliban can host model work, but its existing
Qwen server is not an OWL-SAM endpoint. Native camera/physics throughput remains
a separate runtime investigation. Observed planning/execution remain blocked.
