# Shared relative movement and planning integration

## Baseline

Inspected main `80818ac4` and PR #219 head `948993ba` on 2026-10-23.
Main is already an ancestor of the context stack. #219 is still open; do not
cherry-pick its provider runtime or duplicate its schemas before it lands.
After landing, fetch and merge main into the integration branch and rerun
shared schema/registration, tool outcome, dialogue and motion contracts.

## One capability, two agent adapters

Reuse the shared skill registry for `rotate_base(degrees)` and
`move_forward(meters)` in CHAT and EQA/OVMM. Positive yaw means left/CCW;
distance is body-frame forward in metres. Keep explicit finite arguments,
bounded requests and one measured controller implementation. No raw wheel
command tool, invented object instance, or simulator-only bypass.

The existing controller helpers return commanded angle/distance and ignore
the blocking base command's result. Existing prose can therefore imply actual
travel without measured arrival. Fix that boundary before exposing these
helpers to autonomous EQA/OVMM routing. Preserve legacy callers via thin
adapters rather than silently changing their float return type.

Use the schema-version-1 result envelope already used by TAMP: tool, status,
code, message, data and recovery. Put requested/clipped command, measured
displacement, arrival outcome and observation status in data. Distinguish
rejection before execution, partial progress, timeout, missing measurement,
and successful arrival. Unknown measurement is not zero and not success.
Extend shared result normalization explicitly; do not classify motion tools
as TAMP tools just to obtain envelope parsing.

Translation and rotation must use the existing validated navigation/arrival
path, including swept footprint and unknown-space checks where supported.
Do not reuse a straight-line clipping result as proof of full-body safety.
After execution acquire/update the camera and map, reporting capture failure
separately from motion failure. Make both facts available to the router.

## Planning tools

Keep #219's provider-qualified `plan_task` / `execute_task_plan` and existing
`plan_pick_place` / `execute_pick_place_plan` definitions and opaque handles.
Planning does not actuate; execution revalidates current state and reports
measured effects. Relative motion invalidates plans through those existing
state checks, not by pretending an earlier collision check remains current.
Unbound providers remain unavailable, especially in perception-only EQA.
Do not expose privileged native tools merely because a robot is connected.
Retain the same tool schema wherever a capability is enabled; availability
follows provider and safety restrictions, not benchmark-specific names.

## Perception feedback

Report visual presence, map retrieval, geometry admission and motion outcome
separately. A weak SigLIP miss must not label a VLM-confirmed target ABSENT.
Surface grounding failure is not evidence the target is absent. Retain actual
failure evidence without claiming a target range from unrelated masks.
Relative movement permits model-directed observation without requiring a
grounded instance; it does not grant permission to manipulate search evidence.

## Acceptance order

1. Unit contracts: finite arguments, sign/frame conventions, command rejection,
   measured partial/no movement, missing pose, capture failure, provider gating,
   result parsing and matching CHAT/EQA schemas.
2. Simulation: forward/turn/return with measured poses; obstacle and unknown
   floor rejection; fresh post-motion images. Unchanged arrival tolerances.
3. Repeat the Molmo microwave diagnostic and inspect exact model requests,
   tool receipts and camera trajectory. Preserve failed runs. A different
   semantic answer alone is not successful localization.
4. Small S0 and EQA regression gates, serial CPU-safe GPU-managed runs. Keep
   privileged TAMP qualification separate; no full sweep or hardware test.

This document is an implementation contract, not a claim these repairs have
already been implemented or validated.
