# Active sensor-only agent pilot

The navigation gates in [the repair report](sensor_navigation_repairs_20261106.md)
passed. The next gate is a small active whole-agent comparison. Navigation
acceptance is not evidence of answering or manipulation success.

## Frozen development protocol

- Cases: existing development questions 2, 6, 12, one worker seed (0).
- Arms: production `LazyGraphController` + `AgenticEQAExecutor`, and the adapted
  native GraphEQA planner/graph. Alternate first arm by question index; serial
  child processes, one CPU core and one GPU lease. RL has priority.
- The production agent uses `model_final`: the final model answer wins;
  earlier assessment letters cannot replace it. Instance streaming is off.
- Identical renderer scene/spawn, RGB-D calibration, circular 0.1 m body,
  RGB-D odometry, dense navigation geometry, primitive controller and physical
  footprint sweep checks. Production A* uses that qualified geometry; its
  semantic voxel/graph memory remains independent. No extra clearance padding
  is added on top of the mandatory full-footprint checks.
  No ground-truth pose, labels, object hints or navmesh routes enter either
  planner. The renderer retains its internal collision filter.
- Both start with a 360 degree panorama and stationary floor calibration.
  Dense geometry/recording, semantic ingestion every ninth acquisition callback
  and at task observations. Production tools retain their own ingestion calls.
- Four decision rounds, 200 total attempted primitives (including startup,
  gaze and failed commands), 900 seconds per episode including initialization.
- Same owned Qwen3-VL-8B int4 endpoint, 2048 generated tokens per request,
  maximum four selected images. Each agent keeps its native memory, image
  selection, auxiliary perception and conversation rules. Total calls/tokens
  are measured rather than declared equal. Worker seed does not reset the
  separately served model's random state.
- Preserve incomplete episodes, raw model requests/responses, camera PNGs,
  RGB-D calibration/pose NPZs, acquisition/action receipts and resolved emet
  parameters. Gold answers are used only after the episode model calls.
- Keep 20 GiB free. Stop on tracking loss, collision or terrain rejection;
  preserve the failure rather than switching to simulator pose.

This compares complete agents, **not** an isolated graph/memory ablation.
Different image selection and semantic memory are intentional system
differences. Three development cases cannot establish a benchmark improvement.

Run through the existing owned-server driver:

```bash
python scripts/check_grapheqa_model_api.py \
  --fork /path/to/grapheqa-model-api \
  --paired-active-question-ids 2 6 12 --max-images 4 --max-decisions 4 \
  --output /path/to/new-output
```

Use the managed GPU lease, offline weights, CPU 12 and single-thread math;
the Habitat child needs `.venv-habitat/lib` first in `LD_LIBRARY_PATH`.
The driver requires clean committed source in both repositories.

## Review gates and follow-up

First inspect q2 before interpreting the three-case score: verify the robot
actually moves, the agent sees current calibrated images, the final answer
comes from the model, and failed actions remain in context. An integration
failure is repaired and retained as a separate attempt, never relabeled as a
scored improvement.

Then compare target coverage, model context, answer, motion count, model calls
and runtime across the three pairs. Follow with the existing small Molmo and
RoboCasa OVMM acceptance cases using the production shared agent. This
sensor-only Habitat body has no manipulator: it cannot qualify OVMM pick/place.
Do not claim that an EQA/search run is an OVMM acceptance run.

## Disk preparation

Recovered 10.145 GiB by SHA-256 verifying and hardlinking 33 byte-identical
historical pairs of `model.mjb` / `physical_trace.model.mjb`. Both paths and all
contents remain available. Audit:
`~/runs/emet/physical-tamp-acceptance-20260926/dedup-identical-models-20261106.json`.
No model weights, environments or episode evidence were deleted.

## Results

The first completed six-arm attempt, source `72957d9b`, is retained under
`~/runs/emet/sensor-active-pairs-20261106-r4/`. It exposed a binding defect:
the production planner consulted the sparse semantic map, which rejected the
stationary starting footprint even though the qualified dense navigation map
retained occupied-footprint support. All three emet episodes completed with
`model_final`, but motion requests were rejected before translation.

Both agents scored 0/3 in this failed qualification attempt. GraphEQA completed
q2/q12 and retained a q6 output-length failure; emet completed 3/3. The original
scoring compared raw answers with letters. Rechecking using the existing exact
choice-text/letter normalizer changes no correctness flags in this attempt;
it does recognize emet's q6 `Next to TV` as D and q12 `None` as C. Raw model
answers remain unchanged.

Repairs bind production A* to the same dense navigation space, acquire a real
fresh frame for synchronous clients instead of waiting for a nonexistent recv
thread, and normalize answers only in scoring. The original artifacts remain
untouched. This attempt establishes neither useful target coverage nor answer
quality; rerun before interpreting any agent comparison.

Earlier startup attempts are also retained:

- `sensor-active-pairs-20261106-q2-r1`: cached environment lacks Flash Attention;
  subsequent runs explicitly use SDPA for both arms, without reinstalling it.
- `...q2-r2`: stopped to correct the final-answer policy before task scoring.
- `...q2-r3`: emet controller initialization failed on its unused manipulation
  socket's `None` address; reference q2 retained an output-length failure.

CPU boundary and production-controller checks pass. Live requalification is
pending; none of the above is an OVMM or manipulation acceptance result.

### Shared-geometry smoke, November 7

Job `20261107_092510_3ca19a`, source `e721687f`, is retained at
`~/runs/emet/sensor-emet-q2-20261107-r5/`. Sharing dense planning geometry
restored translation: 37 forward commands, with successive measured route
chunks progressing from the spawn into the entryway. The saved last image
(`active/q2/frame-00241.png`) shows an entry mat by a closed door and console,
**not** the requested bathroom rug. The view assessment correctly reported a
visible rug that did not yet answer the question.

The next model request was `explore_frontier(toward="bathroom")`. The loop
silently replaced it with `investigate(obs_id=-2000000)` through its legacy
`close_map_stay` rule. The same candidate consumed the remaining motion
allowance. At 200 primitives the sensor raised through the loop, preventing
the final model answer. This is retained as an incomplete smoke, not a wrong
answer or a passing EQA episode.

Repairs keep the legacy action rewrites out of `model_final`, return a measured
partial-motion failure when the primitive budget expires, and permit fresh
camera context after the motion cap (the wall deadline remains enforced).
Five real-dispatch rewrite reproductions cover legacy and model-final modes;
partial motion/head-budget tests preserve measured pose without extra motion.

### Grid-coordinate audit

Follow-up `sensor-active-pairs-20261107-r6`, source `23376423`, confirms that
q2 now reaches final model reasoning at the 200-primitive cap. The selected
`explore_frontier(toward="bathroom")` is dispatched unchanged, and the final
request retains 13 conversation messages including the actual tool results.
The model's `White` answer remains wrong; the harness does not replace it.

Auditing that context caught another binding error: a near frontier became
approximately `[-38, -36]` m because production semantic-grid conversion used
1024 cells at 0.1 m while navigation used 256 at 0.05 m. The active batch was
interrupted and its waiting S0/Molmo checks cancelled before admission. No
aggregate comparison is claimed for that incomplete cohort.

The controller now honors explicit `grid_size` and `grid_resolution` parameters
(defaults preserved). The sensor binding supplies the shared geometry's values
and rejects any resolution/size/origin mismatch before creating model transport.
A real-controller test checks a navigation-cell/semantic-world round trip;
another reproduces the old grid mismatch and requires startup failure.
