# Initial native base-approach pilot — November 16

Source `d6da3d6d` passes 2/3 original development tasks and 6/7 transfers.
The control and shift15 pass; swap-right25 completes two transfers, then fails
loaded placement IK for the third object. The same-source stationary ablation
fails cube retention after one transfer. Neither failed swap is promoted.

The cube now passes lift, original-reference hold, handoff and placement. Its
lift-complete relative rotation is 0.005687 rad against the unchanged 0.1 rad
bound. The checked 40 mm base approach finishes about 3.9 mm from its goal and
confirms its stop. A separate live base-only component finishes 4.4 mm from its
40 mm goal; that component is not manipulation or task acceptance.

Candidate audits contain 1,763 commands, zero forbidden commands and no
infrastructure failures; both handle-boundary checks pass. The 361-test affected
run has one failure: a unit stub lacks the real executor's logging callback
after diagnostics were added. It is retained and corrected for the next source;
there is no all-green software claim for this source.

A private IK/collision probe from the failed third-transfer final snapshot finds
five valid release centers inside the original destination region. It has no
live commands and cannot establish task success. It motivates refreshing
placement candidates after the measured loaded-grasp handoff. The follow-up also
composes the payload-center offset through the desired EE orientation, preserves
the unused primitive/episode planning budgets, and refreshes the base wait clock
after preflight. No collision, retention, release or final-goal limit changes.

All live episodes preserve the original case-file hash, layouts, goals and
120-second planning gate, serially on CPU 0–7. Resets are independent without an
exact paired-state certificate. This is privileged simulated Stretch development,
not observed/hardware qualification, training or a new whole-agent score.
Historical native 4/8 and Caliban 3/4 records are unchanged.

[Protocol](../../configs/eval/tamp_native_base_approach_pilot_20261116.json) and
[source/job/artifact manifest](tamp_native_base_approach_initial_20261116.json).
Raw trials and component/probe records remain in ignored
`.tamp-runs/base-approach-20261116/`.

The first refinement replay at `2477cd3f` passes all 365 software tests but loses
the first swap placement: its loaded sampler omitted the original proposal and
the initial planner's local reach patches. The control had completed successfully;
remaining trials were stopped and their interruptions retained. The next source
restores those candidates, retains whole-footprint containment and complete path
checks, and replays the original population. The failed run is not rescored.
