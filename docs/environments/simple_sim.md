# Simple simulation: isolate the local loop

[Environment index](README.md) · [Testing commands](../TESTING.md)

Use the packaged MuJoCo table scene to remove large-world exploration as a
confound. Start with a stationary robot facing visible objects, then add known
turns, short approaches, fresh reacquisition and manipulation. Include absent
queries: continued exploration until an external timeout is not a successful
"not found" response.

The [Stretch table config](../../configs/sim/default_table_stretch.yaml) is a
small control, not a claim of general scene coverage. A red-cylinder or blue-block
find can establish integration without establishing grasp success. Separate
velocity-controlled motion from teleport/kinematic oracle controls in reports.

Acceptance sequence: visible grounding → nearby approach/re-verification →
pick/place → multistep execution. Preserve position-hold and measured-motion
checks alongside learned tests so perception failures are not confused with
robot dynamics or camera calibration failures.

Setup: [simulation configs](../sim_configs.md), [simulation](../simulation.md).
Evidence: [segmented shared grounding](../experiments/segmented_shared_grounding.md)
contains manually reviewed good finds and invalidated false-positive finds;
[cross-task pilot](../experiments/shared_grounding_pilot.md) tracks follow-ups.

Figures should show the full camera view, proposed box, selected mask and depth
support, plus a measured base trajectory for approach tests. A crop alone can
hide an incorrect target association. Follow the [figure checklist](figures/README.md).
