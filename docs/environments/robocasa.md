# RoboCasa: bounded workspace search and manipulation

[Environment index](README.md) · [Testing commands](../TESTING.md)

Use selected kitchen/workspace tasks as the intermediate acceptance gate after
simple simulation. Clutter, occlusion, receptacle choice and manipulation make
these meaningfully harder than a visible tabletop object. For our test order,
choose bounded cases before requiring whole-home search. This is a protocol
choice, not a claim that every RoboCasa task is easier than every Habitat task.

Keep the shared agent and grounding policy unchanged. Record layout, style,
seed, robot, camera and initial target visibility. Explicitly distinguish finding
an object, finding a receptacle, picking and placing. Do not count a correct
localization as full OVMM success, or an oracle plan as learned TAMP success.

Start with one visible-target control and one cluttered/occluded search case,
then attempt learned pick/place before expanding the battery. Confirm the
selected robot/scene adapter works first; the existence of a YAML does not
establish support. The [Stretch kitchen config](../../configs/sim/robocasa_pick_place_stretch.yaml)
is a launch specification, not a passed acceptance result.

Setup: [simulation configs](../sim_configs.md).
Protocols: [OVMM find](../ovmm_find_phase_benchmark.md),
[full OVMM](../ovmm_full_benchmark.md),
[TAMP controls and tests](../experiments/tamp_clutter_testing.md).
Historical reports: [RoboCasa E2E](../dynagraph_robocasa_e2e.md) (check its branch
and robot assumptions), [shared grounding](../experiments/segmented_shared_grounding.md).

Figures: full workspace view, target/receptacle annotations, robot approach and
before/after manipulation views. Show occlusion and failures as well as clean
successes; see the [figure checklist](figures/README.md).
