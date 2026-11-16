# Mars actual TAMP input diagnostic

The completed corrected-feedback baseline has retained-policy general success
151/164, but actual TAMP pregrasp success 0/6 versus teacher 5/6. Extra settling,
a frozen velocity-feature actor and the frozen GRU candidate did not fix that
gap. Diagnose the executed interface before another PPO run.

Run exactly four episodes with current-COM wheel feedback enabled from reset:

1. Teacher, `development:right:50g:1`, recording disabled.
2. Same teacher/case/reset, recording enabled; require identical physical
   outcomes, decisions, returns, object motion and commands to episode 1.
3. Retained legacy actor, same case, recording enabled.
4. Teacher, `development:left:50g:2`, recording enabled to locate the known
   initialization-sensitive planning/execution failure.

Keep actor weights, reference timing, geometry, rewards, tolerances and guards
unchanged. Zero additional settling. No PPO, policy promotion or hardware.
Use one numerical thread and one native worker on CPU14; focused tests use
CPU15. Freeze the source before starting. The previous full regression receipt
does not qualify modified diagnostic source; these are bounded diagnostic runs.

Optional `TampPrimitiveEnv(trace_dir=...)` records the sensor arrays actually
sent over IPC, received actions, existing per-tick robot traces and final stop
confirmation. The critic remains outside the recorded actor input. Error traces
retain the confirmed-stop result even when a planner exception precedes a
terminal transition. This does not relabel existing rollout errors as success.

Run `scripts/trace_mars_tamp_inputs.py --output logs/eval/mars-tamp-inputs-v1`
from a frozen checkout. Each recorded reset has a unique numbered `.json.gz`
archive; an existing output directory is rejected. Raw traces stay ignored.
Publish compact phase-local position/orientation errors, action magnitudes,
commanded/measured joint velocities, input shapes/twists and planner result.
Preserve any parity failure or earlier error instead of adding automatic retries.

Use the result to distinguish target/preview delivery, command tracking,
reference coverage and planner feasibility. A trace alone does not establish
causality or qualify grasp, carry, placement or hardware operation.
