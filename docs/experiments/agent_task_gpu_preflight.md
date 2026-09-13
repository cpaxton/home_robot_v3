# Completed CUDA rollout preflight — 2026-09-13

The queued batch completed all three episodes with full visual evidence. **None
passed the complete task contract.** These are model-policy failures, not simulator
or queue failures. The benchmark's purpose here is to expose those failures rather
than equate a finished worker, confident narration, or partial placement with success.

| Task / layout | Placement goals | Decisions | Episode outcome | Runtime (s) |
| --- | --- | --- | --- | --- |
| Delivery / original three-room v2 | 0/1 | 2 | Claimed completion without manipulation | 16.0 |
| Collection / furnished three-room v2 | 1/2 | 16 | Wrong-object actions; decision budget exhausted | 88.8 |
| Revisit / four-room v2 | 2/2 | 16 | Placement and temporal checks passed; decision budget exhausted | 112.2 |

Runtime includes model loading, observations, actions and scoring; visual export
runs afterward. These are single development runs, not independent performance
estimates. Earlier CPU diagnostics used different code/visual profiles and must
not be interpreted as a controlled CPU-versus-CUDA comparison.

## What the evidence shows

**Delivery:** the model navigated to dining and searched for the cylinder, then
claimed it had placed the object. It issued no manipulation call. The before/after
images and position predicates confirm that the cylinder stayed in the kitchen.

**Collection:** the model delivered the red cylinder, then selected the green cube
for other placements instead of completing the requested blue-cube delivery. The
blue cube remained in the living room. The episode ended at its decision limit.

**Revisit:** an early red-cylinder delivery preceded the green-cube prerequisite.
The green delivery then triggered the private relocation. The agent revisited the
kitchen, observed the displaced cylinder, and delivered it again. Both final
placements and the temporal predicate passed. It nevertheless repeated navigation
and manipulation requests until the decision budget expired. Repeated requests for
already placed objects returned `no_reachable_task`; the policy failed to use its
recorded delivery receipts to terminate.

Thus the run demonstrates assisted observation refresh and redelivery after a
change, but does **not** establish successful task termination or learned DynaGraph
recovery. These action rollouts use visible-label text and the observed-object
cache, not the RGB QA runner or the full voxel/semantic-graph backend.

## Execution and provenance

- Model: cached Qwen2.5-3B-Instruct on CUDA, one episode at a time.
- Source: detached snapshot `48f9ec219953d1a2f4f4de5b89eb370d44c3fe19` at
  `/tmp/emet-gpu-rollout-source-20260913`.
- Limits: 16 decisions, 128 generated tokens per call, 300 seconds per episode;
  outer batch command limited to 1,200 seconds. No retries or paid model calls.
- Queue: `20260913_104943_dbbe9a`, terminal status `done`. It waited for the existing
  grasp and navigation supervisors **before** launching the GPU-exclusive child.
  This avoids holding the GPU lock while waiting on another queued lock user.
- GPU child: required the shared host lock and 14,000 MiB free memory. Model
  inference used CUDA; MuJoCo rendering used the previously verified Mesa EGL path.
- Batch root: `/tmp/emet-gpu-rollouts-20260913`. `summary.json` distinguishes
  `batch_status: completed` from `tasks_passed: 0`.

The event/image chains, recorded Python source snapshots and all visual artifact
hashes were verified. `paper/data/agent_task_gpu_v2/evidence_audit.json` records the
checks; `summary.json` retains model identity, source hashes, outcomes and trace
references. The three storyboards and full-size revisit frames were visually
inspected. At steps 480, 505 and 586 of the revisit trace, the images show the green
cube placed, the displaced cylinder observed, and the red cylinder placed again.

## Review and reproduce

Open `/tmp/emet-gpu-rollouts-20260913/index.html` for the batch overview. Each task
folder contains `report.html`, `episode.rrd`, `episode.mp4`, figures, immutable
model/tool events, metrics, and its source snapshot. Queue logs and the exact batch
launcher are retained beside them.

For a new rollout use the existing `emet jobs run --gpu-exclusive` lifecycle and
`emet eval agent-tasks agent --device cuda --model qwen25-3B-Instruct`, with the
suite/task pair and limits above. Use a fresh output directory. Do not bypass the
host queue or overwrite the original evidence.

Re-export the completed cohort without replacing earlier CPU results:

```bash
uv run emet eval agent-tasks export-agents \
  /tmp/emet-gpu-rollouts-20260913/cross_room_delivery \
  /tmp/emet-gpu-rollouts-20260913/two_object_collection \
  /tmp/emet-gpu-rollouts-20260913/moved_object_revisit \
  --paper-dir paper --bundle agent_task_gpu_v2
./paper/build.sh
```

`--bundle` selects a separate data directory and figure prefix. The current result
should guide the agent-owner work on unsupported completion claims, wrong-object
selection, and recognizing previously completed actions. The scorer remains
unchanged; no successful outcome has been manufactured by relaxing its conditions.
