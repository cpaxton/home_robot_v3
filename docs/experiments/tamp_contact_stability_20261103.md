# Native TAMP contact stability — November 3

Adding finer grasp-yaw candidates **does not improve complete-task success** in
this fixed diagnostic: each policy completes 1/2 layouts. Both pass the familiar
control and stop on cube rotation in the +15 mm layout. The finer candidates do
complete the shifted cube's lift; their rejection occurs later during placement
descent. Keep the earlier **2/4 actual-agent result** unchanged. This experiment
uses scripted public task tools and zero LLM calls.

| Layout | Grasp search | Complete task | Completed transfers | Cube lift | Failure phase |
| --- | --- | --- | --- | --- | --- |
| Familiar control | Existing yaw candidates | pass | 2/2 | complete | none |
| Familiar control | Finer yaw candidates | pass | 2/2 | complete | none |
| Initial X +15 mm | Existing yaw candidates | fail | 1/2 | rejected | lift |
| Initial X +15 mm | Finer yaw candidates | fail | 1/2 | complete | place |

The four native diagnostic episodes complete **6/8 planned transfers**, with
**1,992 audited commands, zero forbidden commands and zero infrastructure
failures**. Both failed receipts preserve the completed transfer and stop
dispatch. Neither is a full-task pass. The candidate policy remains confined to
the diagnostic adapter; production grasp search and retention limits are unchanged.

![Measured native load rotation](media/tamp_contact_stability_20261103.png)

The figure shows sampled object-to-gripper rotation during **lift only**. Crosses
mark lift-time rejection. The finer-yaw shifted lift finishes below the limit;
its later placement rejection is deliberately excluded from that curve. Samples
are connected for readability. These four episodes are development diagnostics,
not an agent score, a certified reference population or a reliability estimate.

## What the comparison establishes

The shifted baseline rejects at 0.100764 rad relative rotation and 1.612 mm
translation, about 1.10 cm above the closure sample. Both fingertips still
contact the cube. With finer yaw, the requested grasp pose is 12 mm lower. Its
lift completes with a maximum recorded rotation of 0.077656 rad (4.45 degrees),
then descent rejects at 0.100184 rad and 2.867 mm translation. Both fingertips
remain in contact in that snapshot too. The 11.09 cm height at the latter
rejection is a **placement-descent height**, not the point where lift stopped.

The shifted cylinder's lift rotation decreases from 0.015097 to 0.003648 rad.
Those improvements in geometry and intermediate effects do not justify
promoting the candidate policy or removing either failed task. Recorded contact
penetration and passive-finger state are simulator debugging truth, not
calibrated hardware force or tactile evidence. This is a rigid payload-pose
rejection; these snapshots do not establish a dropped object.

## Fixed inputs and source

The [protocol](../../configs/eval/tamp_contact_stability.json) freezes all four
conditions before launch at `81c014cf`. The runner reconstructs fixtures from
versioned assets and checks the compiled model signature. Every condition uses
the same goal regions, native controllers/contact parameters, 256-call refinement
and 2,048-call provider IK limits, 20 mm / 0.1 rad retention limits, beam task
search, startup policy and recorder. Only +/-0.175 rad yaw candidates are added.

`scene_tasks`, `plan_task` and `execute_task_plan` run through scripted CHAT
dispatch over independent live ZMQ simulators. No cameras, local GPU, VLM or
hardware are used. Nominal scene hashes match within each pair; measured live
initial states are archived and are **not bit-identical**. Therefore this is not
an exact-state replay certificate or an isolated timing ablation. Recording took
at most 1.72 ms per sampled event in these runs; that overhead is measured rather
than assumed absent.

The [compact manifest](tamp_contact_stability_20261103.json) records every outcome,
native failure phase, selected target heights, loaded-lift residuals, source and
scene provenance, actual initial-state comparisons, job IDs and raw-file hashes.
Raw state/contact/response/command records and source snapshots remain under
`.tamp-runs/contact-stability-20261103/`. The full raw summary's pose extrema
span all recorded phases, including release; use phase-filtered records for
retention interpretation. Dense traces and compiled models are not embedded in
the repo. Plot reproduction uses the terminal population:

```bash
python scripts/plot_tamp_contact_stability.py \
  --runs .tamp-runs/contact-stability-20261103 --output FIGURE_BASENAME
```

## Next implementation gate

The current adapter computes placement targets from the measured post-lift
object position, but its collision/retention payload transform still comes from
the pre-lift grasp. Investigate a checked primitive handoff: after a verified,
stable loaded grasp, validate the existing bound, bind the measured payload pose
once, and fully replan placement against that new geometry. A failed bound or
lost contact must still stop. Rebinding on every waypoint would hide creep and
is not an acceptable substitute.

Compare that handoff against the preserved-reference policy with the same finer
candidate set and controls. If any larger rotation range is admitted instead,
the swept payload volume must explicitly cover it. Keep the existing rotation
limit until that collision argument is implemented and tested. Expand the
prospective placement population before promoting a policy, then repeat actual
`run agent` trials through Caliban. Observed RGBD/robot-self/background/contact
qualification remains a separate gate; hardware Stretch stays paused.

Follow-up instrumentation labels the active motion phase directly, and plotting
regressions ensure a later placement rejection cannot be called a failed lift.
The final selected diagnostic/native-adapter suite passes **40 tests**, with zero
failures/errors/skips, superseding the initial overlapping 36-check run.
Ruff and whitespace checks pass. The raw `81c014cf` captures remain unchanged;
the metadata/plotting follow-up does not supply a new physical result. The paper
appendix includes the figure and explicitly retains the failed complete-task
outcomes. Its build succeeds at **55 pages**, without undefined references or
citations.
