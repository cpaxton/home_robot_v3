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

For the current open-sink pilot, use the separate
[open-receptacle config](../../configs/sim/robocasa_counter_to_sink_stretch.yaml).
Generation-only checks found that the same seed can produce different target
categories across processes and different placements within one process.
Archive and reuse the generated scene **and task metadata** for paired tests;
verify the instruction matches the actual fixture before launching the agent.
Saved generated XML now expands mutable robot includes, but still requires
the matching external mesh/texture assets. Also record the actual adapter
spawn: open-floor autoplace can move it metres from RoboCasa's suggested hint.
See the [retained failures and reproducibility probe](../experiments/shared_grounding_pilot.md).

Verify **compiled body dynamics**, not just renderings, after scene conversion.
A blanket mesh `inertia="shell"` compatibility rewrite inflated one generated
pear from 0.07874 kg to 4.570 kg and coincided with robot tipping during pickup.
The adapter now pins source RoboCasa masses, centers of mass and inertias before
mesh adaptation; it does not tune densities to make an episode pass. Generation
tests compare surviving source bodies against the adapted and saved model,
including zero-mass markers. Frozen scenes from before this fix retain their
old dynamics: keep corrected-dynamics fixtures separate and record the change.
Robot insertion, contact-solver settings and native robot mass calibration are
separate concerns; this fix does not validate all robot payload limits.

Also audit transformations **before** the native model is compiled. September
14 preflight found that installed fork `3d0bd42` unconditionally adds shell
inertia to unspecified meshes in `Kitchen.edit_model_xml`. Small objects then
have kilogram-scale masses before EMET sees them. Preserving those values is
source equality, not validation of authored dynamics. A separate dependency
review is needed; do not tune densities or use these fresh scenes for policy
acceptance meanwhile. See the source/export and matched-mode controls in the
[pilot report](../experiments/shared_grounding_pilot.md). Previously archived
corrected fixtures remain distinct interventions, not evidence that the current
installation's generator is fixed. Record dependency revisions and local
modifications along with the EMET source/configuration for future frozen rooms.

Setup: [simulation configs](../sim_configs.md).
Protocols: [OVMM find](../ovmm_find_phase_benchmark.md),
[full OVMM](../ovmm_full_benchmark.md),
[TAMP controls and tests](../experiments/tamp_clutter_testing.md).
Historical reports: [RoboCasa E2E](../dynagraph_robocasa_e2e.md) (check its branch
and robot assumptions), [shared grounding](../experiments/segmented_shared_grounding.md).

Figures: full workspace view, target/receptacle annotations, robot approach and
before/after manipulation views. Show occlusion and failures as well as clean
successes; see the [figure checklist](figures/README.md).
