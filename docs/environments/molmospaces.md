# MolmoSpaces: scene and embodiment diversity

[Environment index](README.md) · [Installation and integration](../molmospaces.md)

Use selected MolmoSpaces layouts and objects as transfer checks, not as a single
difficulty tier. Start with stationary visibility and known-motion tests before
cluttered search or manipulation. Distinguish asset/render problems from
navigation, perception and planning failures.

Record the actual MJCF/robot asset, not just a registry alias: prior `rby1`
tests in this project resolved to a Galaxea R1 asset and were not evidence of a
Rainbow RBY1 hardware model. Keep scene generation, spawn pose and robot adapter
separate from shared agent configuration. See [spawn metadata](../molmospaces_spawn_metadata.md)
and [simulation configs](../sim_configs.md).

Evidence: [grounding closeout](../experiments/ovmm_grounding_closeout.md) documents
the embodiment caveat; [segmented grounding](../experiments/segmented_shared_grounding.md)
includes offline multi-object/view tests. Offline masks do not prove closed-loop
find or manipulation. Publish scene overview, robot/camera pose and paired RGB/
mask views using the [figure checklist](figures/README.md).
