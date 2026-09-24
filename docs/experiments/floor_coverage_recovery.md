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

## September 24 results

Three sequential managed jobs, unchanged task prompts and physical scorers:

- `20260924_083453_7df611`, source `4cb00bf0`, Molmo + RoboCasa:
  both physical pick/place false. Molmo exposed a contradictory system prompt
  forbidding all recovery; RoboCasa happened to obtain a grasp workspace without
  the new observation, then failed fresh manipulation target verification.
- `20260924_084308_7c9cad`, source `9fc4c927`, Molmo: Qwen selected
  `observe_floor`, which succeeded, but the next turn returned object labels.
  Chat and perception share a VLM client; caption inference had reset the agent
  conversation. The system prompt repair alone was therefore insufficient.
- `20260924_084920_24fb06`, source `213b239f`, Molmo: conversation isolation
  across tool execution enabled the complete model-directed sequence: failed
  pickup, floor observation, and pickup retry. Reachable cells increased 28→37;
  nearest reachable target distance improved 1.027→0.927 m, still outside the
  unchanged 0.85 m maximum. The retry stopped safely; physical pick/place false.

Artifacts respectively live under `~/runs/emet/floor-recovery-20260924/`,
`~/runs/emet/floor-recovery-prompt-20260924/`, and
`~/runs/emet/floor-recovery-context-20260924/`. Inspect `process.log`,
`physical_result.json`, and `evidence/navigation/failed_approach_*.npz` inside
each room's `hybrid_learned_pick_place` directory. Chat transcripts are also
under the frozen worktree's `logs/chat/`. All agent processes exited zero;
managed jobs failed on the independent physical scorer, not task success.

203 focused tests pass on the final code, including navigation, view quality,
tool contracts, grasp handoff, payload guards, and conversation restoration on
both successful and exceptional perception calls. No full EQA run was made.

Next investigate why one floor view leaves the remaining approach unobserved:
inspect actual depth/map support and footprint coverage before adding retries
or enlarging budgets. Do not increase grasp reach or weaken unknown-space checks.
This establishes usable model-directed recovery, not completed OVMM acceptance.
