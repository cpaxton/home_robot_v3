# Visual task environments and scored QA, v2

Follow-up: [completed CUDA action rollouts](agent_task_gpu_preflight.md).

The three bundled layouts now use `semantic_rooms_v2`. Kitchens have appliances,
cabinets and tiled backsplashes; living rooms have sofas; studies have bookshelves;
dining rooms have chairs. Colored wall treatments support these geometric cues.
These remain compact procedural MuJoCo fixtures with assisted movement/manipulation.
They are not native BEHAVIOR scenes or photorealistic homes.

The green object is now called a **green cube**, matching its shape. Trays have
visible bases and rims. Tables have legs and wider surfaces beneath the trays.
Robot RGB is 960×540 with a 65-degree vertical field of view. Assisted placement
returns the robot to an actual room-center pose, recording the travel and a view
that includes both trays. The three-pixel oracle visibility threshold remains for
legacy action observations; QA has a separate stricter evidence check.

## Ten scored question types

| ID | Test | Required input |
| --- | --- | --- |
| `color` | Cylinder color | One RGB view |
| `tray_count` | Number of visible trays | One RGB view |
| `relative_position` | Blue cube relative to green tray | One RGB view; camera-relative direction |
| `empty_trays` | Whether either tray is occupied | One RGB view |
| `room_identity` | Room type from kitchen furnishings | One RGB view; no room metadata |
| `observed_room` | Recall where the blue cube was observed | Three RGB views with named-room metadata |
| `inventory` | Unique small-object colors across rooms | Three RGB views |
| `moved` | Cylinder's image-relative movement | Matched before/after RGB views |
| `placement` | Both objects in their matching trays | Full-table RGB verification view |
| `unknown_location` | Abstain when the requested object is outside the only supplied view | One RGB view |

These QA episodes use **fixed observations**, not active navigation. Before/after
and placement views are staged fixture states, separate from learned action results.
The existing three action tasks still exercise search and multi-room tool execution.
Questions have explicit answer formats and must cite all supplied view IDs. Values
are scored deterministically; missing answers, wrong values and incomplete/foreign
evidence fail. Case and terminal punctuation are normalized; inventory order is
ignored. Citation correctness alone is not proof of model inspection.

The builder rejects required targets below 128 pixels, below eight pixels in either
bounding-box dimension, or touching the image boundary. It records camera pose,
segmentation counts and image hashes. The QA fixture contract checks shape/color
assumptions before generating keys, and image-relative directions use actual camera
extrinsics. This is a curated starter set; freeze new questions and independent
held-out layouts before using it to claim general QA performance.

## Run and score

Run from the implementation worktree, `/tmp/emet-agent-task-preflight` on this host.
For software rendering, first set:

```bash
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_ALWAYS_SOFTWARE=1
```

```bash
uv run emet eval agent-tasks qa-build \
  --suite src/emet/config/benchmarks/agent_tasks_four_room.yaml --out /tmp/my-qa-pack
uv run emet eval agent-tasks qa-agent /tmp/my-qa-pack/public \
  --model qwen35-0.8B --out /tmp/my-qa-agent
uv run emet eval agent-tasks qa-score /tmp/my-qa-pack \
  --answers /tmp/my-qa-agent/answers.json --out /tmp/my-qa-score.json
```

`qa-agent` uses the **shared task runtime with actual RGB**, unlike the earlier
text-only action-policy diagnostic. Its tools are `next_view` and `submit_answer`.
Each question resets conversation history; only that question's images are passed.
Unseen image citations and submission before all supplied views have been shown
are rejected. With `qa-score --out`, a companion HTML gallery shows model submissions
beside their evidence and reviewer keys. An accepted submission ends the task without requiring an extra
model completion reply. It records exact model inputs, image IDs, raw responses,
syntax repairs, tool outcomes and hash-linked events. Each worker has an owned
process group and hard timeout. The CPU VLM allowlist is cached Qwen3.5 0.8B/2B/4B;
loading is offline and paid calls are disabled.

For a smaller smoke test, add `--question color --max-rounds 3`. Scoring a subset
still reports missing answers against the full ten-question pack; it does not
silently count unattempted questions as passes.

A submitted answer file looks like:

```json
{"color": {"answer": "red", "evidence": ["view:0"]}}
```

Only give the policy the public question/image subset. `private_answers.json`,
`fixture.xml`, and `review.html` are researcher/evaluator files. The gallery contains
answer keys and must not be used as a policy input. All questions share one public
image store, but the built-in runner scopes input to each question's `view_ids`.

## Validation and visual artifacts

- All nine action task/layout witness controls passed with complete visual evidence:
  `/tmp/emet-v2-layout-controls`.
- All 30 QA cases across three layouts passed required-target readability checks:
  `/tmp/emet-qa-v2-release/{base,furnished,four_room}`. Each folder has `review.html`.
- Perfect-key scorer controls passed 10/10 per layout. Wrong-color and missing
  temporal-view controls each failed exactly their affected case. These are scorer
  controls, not model performance.
- A cached Qwen3.5-0.8B RGB smoke test answered the color question correctly with
  the right citation. The initial attempt failed because it returned a bare JSON
  tool array. The shared runtime now records a repair that adds only the missing
  envelope around complete known calls; the original response stays in the trace.
- 117 focused tests passed, including visibility thresholds, tampered images,
  question/key mismatch, incorrect answers, missing/foreign citations, RGB delivery,
  multi-view continuation, and submission completion.

The full local QA run is recorded at `/tmp/emet-qa-v2-release/local_vlm`; Qwen3.5-0.8B
passed **6/10**. It passed color, tray count, relative position, empty-tray occupancy,
kitchen identification, and insufficient-evidence abstention. Multi-view inventory
and room recall produced no accepted answers. Temporal direction was correct but
omitted the second evidence citation; placement verification was wrong. The score,
answers and trace hashes are retained in `paper/data/agent_task_visual_qa_v2/local_vlm.json`.
This is one development diagnostic, not an estimate of general QA performance.
Earlier action-policy results in the paper used the earlier visual profile and
must not be pooled with this release.

Regenerate the paper scene figure and validation summary after building all layout
packs and action controls:

```bash
uv run python scripts/benchmarks/export_visual_qa.py \
  --packs-root /tmp/emet-qa-v2-release --controls-root /tmp/emet-v2-layout-controls \
  --paper-dir paper
./paper/build.sh
```
