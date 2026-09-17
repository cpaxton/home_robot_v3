# Habitat / HM3D: large-environment search and EQA

[Environment index](README.md) · [Habitat setup and usage](../habitat/README.md)

Treat our Habitat OVMM find cases as long-range search stress tests, not the
first acceptance gate for the local manipulation loop. Targets can be far from
spawn in large environments. Short language descriptions may provide little
actionable location information. Measure starting distance/visibility per case;
do not assume all cases are distant or compare raw success rates across worlds
as if difficulty were matched.

Language and voxel retrieval propose places to inspect; they do not guarantee
a destination. Exploration must still cover unseen space and bring useful
evidence into view. Distinguish failure to see the target from failure to ground
a visible target, and both from selecting the wrong relational instance.

The [nearest-v2 registry](../../configs/ovmm/habitat_find_phase_nearest_v2.yaml)
asks for a lamp nearest a bed and a table. Its nearest-object scorer uses
horizontal proximity to start-category bodies, **not an "on" support relation**.
Do not pool this with legacy relation variants. A bed mask is not a lamp
localization, even if a full-frame language judgment says a lamp is nearby.

Habitat also supplies EQA tests for exploration, evidence retention and answers.
Keep EQA in the shared acceptance battery: improving find must not silently
change its memory policy or budgets. These find-phase tests do not establish
physical pick/place or real-robot dynamics.

Commands: [Habitat usage](../habitat/usage.md),
[evaluation runbook](../evaluation.md), [find benchmark](../ovmm_find_phase_benchmark.md).
Evidence: [shared pilot](../experiments/shared_grounding_pilot.md) records the
paired cases, infrastructure exclusions and a manually inspected false acceptance.

Figures should pair a top-down explored-space/path map with egocentric frames
at first visibility, investigation and final decision. Mark GT locations only
in evaluator figures, clearly separate from agent-visible inputs. Preserve map
coordinate conventions and units. See the [figure checklist](figures/README.md).
