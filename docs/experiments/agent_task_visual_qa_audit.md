# Visual answerability audit — 2026-09-13

The current integrated benchmark contains **three action tasks**, repeated across
three layouts. It does not yet score question answers. The local model used in
the existing diagnostics receives camera-visible label text, not RGB. Reviewing
rendered images establishes candidate visual evidence, not VLM performance.

## Integrated tasks

| Task | Behavior under test | Independent check |
| --- | --- | --- |
| Cross-room delivery | Find the red cylinder; deliver it to the dining-room red receptacle | Cylinder center inside the goal volume; object not held; agent finishes |
| Two-object collection | Find objects in different rooms; deliver red and blue objects to their corresponding receptacles | Both placements, neither held, and agent finishes |
| Moved-object revisit | Observe the cylinder; deliver the green object; revisit the cylinder after relocation and deliver it | Placement predicates plus observation/delivery ordering |

The relocation is **0.45 m along the same kitchen table**, not a room change.
This tests stale position recovery; it does not establish difficult cross-room
object rediscovery. All nine task/layout combinations passed assisted solvability
controls. The real local policy passed delivery, placed both collection objects
but failed to stop, and failed the revisit task at 1/2 placements.

## Image spot check and question candidates

I inspected nine full-size recorded head-camera frames spanning the original,
furnished and four-room layouts, before/after placement, and matched viewpoints
before/after relocation. The self-contained gallery is
`/tmp/emet-agent-visual-qa-audit/index.html`. Its companion `questions.json` records
the source run, event hash, image hash, camera pose, required inputs and expected
answer. Images are copied unchanged; some are controlled runs, labeled as such.

| Question family | Example | Expected answer | Required evidence |
| --- | --- | --- | --- |
| Attribute | What color is the cylinder? | Red | Starting RGB image |
| Count | How many large colored pads are visible on the dining table? | Two | Dining image before delivery |
| Spatial | Is the blue block left or right of the green pad in this image? | Right | Living-room image; image-relative wording |
| Occupancy | Is either dining pad occupied by a small object? | No | Dining image before delivery |
| Action verification | What rests on the blue pad and on the red pad? | Blue block; red cylinder | After-collection image |
| Multi-view inventory | What small objects occur across these three room views? | Red cylinder, blue block, green block | Kitchen, living and dining views; exclude large pads |
| Observed location | In which visited room was the blue block observed? | Living | Observation plus recorded room/pose association |
| Temporal change | Did the cylinder move between these matched views, and which image direction? | Yes, left | Both kitchen views at the same camera pose |
| Insufficient evidence | From the starting image alone, which room contains the blue block? | Cannot determine | Starting RGB only; do not supply inventory or room labels |

These are manually audited candidates, **not integrated/scored QA episodes**.
The current text policy can support observed-location retrieval. Pixel attributes,
spatial relations and before/after visual comparison require image input or a
grounded perception tool; the current text observation omits some of that evidence.

## Findings and quality gates

- Colors and simple shapes are distinguishable at the inspected room-center
  viewpoints. Recorded segmentation areas there were 234–237 pixels for the red
  cylinder, 266 for small blocks, and about 2,500 for each receptacle.
  The existing three-pixel visibility threshold is an oracle-presence threshold,
  not a sufficient visual QA readability gate.
- The rooms are visually almost identical. A question asking a model to infer
  kitchen versus study from appearance is unsupported. Use supplied room IDs
  for now; add distinctive room furnishings before evaluating visual room identity.
- The `green_marker` asset is a green block, not a recognizable marker pen.
  Receptacles look like flat colored pads without tray rims. QA wording above
  matches the render. Reconcile task labels and assets before a VLM baseline.
- Close placement images crop the pads and table. Their object/pad relationships
  are still visible in the inspected frames, but they are poor inventory views.
  Capture a canonical room-center verification image after each placement.
- Extra furniture is made of plain boxes and often behind the default viewpoint.
  The furnished suite checks geometry; it is not yet a substantial visual-semantic
  difficulty increase or a polished household scene.
- The relocation is visibly different at matched viewpoints. Increasing world X
  appears as a move left in these camera images. Never label it "right" using world
  coordinates when asking an image-relative question.

Before adding QA scores, give each question an explicit start state, input modality,
allowed exploration tools, evidence requirement and normalized answer schema.
Separate single-view, active exploration, temporal memory and abstention cases.
Do not expose answer keys or evaluator overhead images to the policy. Freeze the
task/asset/observation profile before collecting a comparative QA baseline.
