# Decision-view grounding qualification

The coordinator's selected camera image now reaches grounding with its original
RGB-D, calibration, camera pose and navigation origin. Two live grounding calls
pass source RGB, displayed RGB and sensor-depth hash comparisons. A visible
foreground stool is admitted and its selected mask is manually checked.
**The microwave task is still unsuccessful:** the available view clips the
target, and Qwen abstains. This is evidence of a repaired handoff, not benchmark
superiority or successful pick/place.

## Source and changes

- `7f0ae342`: own one decision snapshot, attach a decision ID, pass it through
  existing `ground_object` and normal RGB-D/graph admission instead of recapture.
  Preserve history and original timestamps; exclude task labels and wrist views.
- `25534704`: repair the live-discovered `Frame.info` reassignment error.
  `Frame` is a namedtuple; annotate its metadata dictionary. The regression test
  now uses the production frame type instead of a mutable namespace.
- `803e397d`: retain map scope before inference and reject a snapshot after a
  map reset, before insertion/admission. This follow-up has CPU coverage;
  the live trial below used the preceding commit, `25534704`.

No tool name/argument is added by these repairs. CHAT still registers 34 tools;
all five live requests advertise the same 32, with the task-plan pair suppressed
by the existing provider gate. EQA's advertised surface is unchanged. See the
[tool inventory and consolidation priorities](agent_tool_surface_20261117.md).

The next tool round gets a new decision snapshot; older conversation images
remain historical. Missing camera data fails explicitly rather than reusing the
prior turn or recapturing silently. The recorded view can precede another action
in a batch. Geometry is evidence from that view, not continued visibility,
route permission or fresh manipulation authorization.

## Live checks and preserved failures

| Run | Frozen source | Result |
| --- | --- | --- |
| `external-coordinator-trial-20261117-r2` | `7f0ae342` | Independent operator identifies microwave visually; both groundings raise `can't set attribute` before admission. Six responses; no XYZ. |
| `external-coordinator-trial-20261117-r3` | `25534704` | Parent-operated diagnostic: microwave abstains, visible-stool control admits geometry; both groundings pass all image/depth association checks. Five responses. |

Managed jobs: r2 `20261117_082550_75b002`; r3
`20261117_083554_f0fd83`. Both ended normally. An exit-zero result is not scored
as task success. Raw artifacts are preserved under
`/home/cpaxton/runs/emet/external-coordinator-trial-20261117-r{2,3}/`;
`manual_evaluation.json` supplies separate task/contract judgments.

The r3 subagent hit its usage limit before writing any response. The parent then
operated the public mailbox from response 1, without simulator GT or privileged
object coordinates. `operator/protocol_annotation.json` records the change.
Thus r3 is an informed contract diagnostic, **not an independent usability
trial**. Its wall time includes prolonged operator waiting and cannot support
a runtime comparison.

Both runs retain the same tool schemas, admission configuration and budgets:
12 responses including final summary, at most eight base-tool macro requests,
physical execution guards, teleport disabled, stable CPU affinity and one
exclusive GPU job. Scene: Molmo iTHOR train index 0, `rby1` adapter with Galaxea
R1 MJCF proxy, not native RB-Y1 hardware. Caliban serves Qwen3-VL-8B; deployment
precision/revision remains unverified. Grounding uses `depth_candidates`,
`yoloe_sam2` proposals/recovery and VLM surface selection. The depth source is
renderer sensor depth with unchanged 0.25–2.5 m limits; no GT masks or object
poses enter grounding. Estimated-depth performance is not qualified here.

### r3 outcomes

Startup simulation image time advances 1.332→1.660 s; base-up dot world-Z is
0.999987. One stationary floor inspection adds 77 observed cells; one in-place
scan obtains a room view. There is no translation request or manipulation.
Twenty actuation-audit entries are accepted, with zero forbidden entries.

| Grounding | Decision view / request | Semantic / geometry result | Association checks |
| --- | --- | --- | --- |
| Partly visible microwave above stove | `2c6e800936984b57a24abaf0a21c9e2f` / 3 | Semantic abstention; no admitted position. Target clipped at upper edge. | Source RGB, displayed RGB and sensor depth all match. |
| Nearest black foreground stool, separate positive control | `e2cdda6e012645e2ac6c64c67f6e855b` / 4 | 18,082 masked depth points; world XYZ `[0.797331, 0.717954, 0.750399]` m; graph obs 14. | All three comparisons match. |

IDs above have the `decision-view:` prefix in actual receipts. Source images are
640×480; displayed images are 512×384. Their hashes are deliberately separate.
`handoff_verification.json` records the complete comparisons against saved PNG
pixels and NPZ depth, rather than assuming that a voxel index identifies the
camera attachment. Empty acquisition timing remains explicitly unverified.

The stool's candidate-0 mask follows the nearer stool's seat/legs and excludes
the farther stool and counter. This is qualitative visual inspection, not
ground-truth IoU or XYZ error scoring. The positive control does not count as
microwave-task completion.

Inspect the r3 evidence:

- [Coordinator input for stool control](/home/cpaxton/runs/emet/external-coordinator-trial-20261117-r3/public/request_0004/message_007_image_01.png).
- [Selected mask with context](/home/cpaxton/runs/emet/external-coordinator-trial-20261117-r3/private/perception/grounding/grounding-399770713494405f85cea3d294dbaeec-surface-0-context.png).
- [Original resolution and accepted box](/home/cpaxton/runs/emet/external-coordinator-trial-20261117-r3/private/perception/grounding/grounding-399770713494405f85cea3d294dbaeec-numbered.png).
- [Hash verification](/home/cpaxton/runs/emet/external-coordinator-trial-20261117-r3/handoff_verification.json) and [manual evaluation](/home/cpaxton/runs/emet/external-coordinator-trial-20261117-r3/manual_evaluation.json).

## CPU checks and remaining gates

The final focused pack has **187 passing tests**, covering camera ownership,
stereo inputs, calibration/timing, production CHAT/HTTP history, missing camera,
actual frame metadata, map-reset rejection, admission/cache, schemas, shared VLM
dialogue and existing acceptance helpers. Ruff and commit hooks pass.

The remaining view problem needs measured task gaze. CHAT has floor inspection
and base movement, but no general upward task-gaze request. Do not use repeated
floor inspection as a disguised object-gaze API or invent target coordinates.
Extend/consolidate the existing inspection capability with measured limits and
truthful rejection feedback, then qualify unclipped microwave reacquisition.

After that, run small paired EQA and simple OVMM regression gates before making
performance or merge claims. No broad sweep, new stopping/answer override,
manipulation, real-robot or TAMP result is established by this diagnostic.
