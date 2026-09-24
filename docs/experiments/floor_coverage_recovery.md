# Floor-coverage recovery pilot

The September 23 small-room controls both stopped before grasping. Molmo had
28 reachable cells, all outside the 0.709–0.85 m grasp workspace. RoboCasa had
five candidates in range, all rejected by the explored-footprint check, not
by observed obstacle overlap. These are Stretch-in-MuJoCo controls, not RBY1.
Artifacts: `~/runs/emet/grasp-approach-audit-20260923/`.

## Contract

The tool result distinguishes `insufficient_floor_coverage`,
`workspace_obstructed`, and `no_reachable_workspace`. The last code does not
prove either missing coverage or a physical obstacle. Results include radial
bounds, reachable-cell counts, and rejection counts; no simulator truth enters
the agent. The planner retains its existing footprint and clearance checks.

For a confirmed pre-grasp rejection with no pickup executed and a successful
post-action observation, the model may request `observe_floor`. The failed
batch is discarded. Only that observation is executable in the next round;
another action requires a subsequent round after the observation succeeds.
The existing three-round budget is unchanged. Uncertain payloads, stale frames,
unconfirmed head motion, and failed mapping do not authorize a retry.

The observation moves only the head, requires measured head pose and a newer
received frame after motion, then updates the map from calibrated RGB-D.
Frame receipt sequence is not an acquisition timestamp guarantee; stronger
bridge timestamp validation remains useful. No clearance is inferred from a
successful observation. Adapters lacking the required capabilities refuse it.
Lidar may provide obstacle evidence, but free scan rays are not floor support
or proof against drop-offs.

## Acceptance

Run the frozen small-room Molmo and RoboCasa controls serially, with the same
Qwen preset, seeds, budgets and independent physical scorer. Inspect whether
the model requests the observation, whether the map gains usable coverage,
whether replanning succeeds, and whether physical pick/place passes. Do not
call the change an OVMM improvement solely because the head moved.

EQA uses its existing separate episode tool pack; this pilot changes CHAT
manipulation recovery, not EQA policy. The low-level diagnostics are shared.
No robot-specific policy branches or weakened safety thresholds are added.
