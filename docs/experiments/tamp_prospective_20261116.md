# Prospective TAMP layouts and scripted recovery — November 16

The frozen prospective gate passes **4/4 native tasks and 4/4 actual-agent tasks,
10/10 transfers in each cohort**. Both separate native scripted recovery trials
also pass, including all five expected boundary checks. This extends PR242's
development evidence without changing its motion implementation.

| Population | Tasks | Completed / declared transfers | Audit commands |
| --- | --- | --- | --- |
| Four new layouts, native public tools | 4/4 | 10/10 | 2,573 |
| Same declared layouts, actual `run agent` | 4/4 | 10/10 | 2,545 |
| Separate native scripted recovery | 2/2 | 4/4 | 987 |

All trials use clean source `63b85dfa`. Native adapter, approach, placement,
policy, release and command-client hashes match PR242's frozen `855348a2`.
The [protocol](../../configs/eval/tamp_prospective_20261116.json) and
[plan](../plans/2026-11-16_tamp_prospective_evaluation.md) were committed before
any live execution. All six native fixtures compiled and retained one canonical
model signature. No case was replaced, retried or used to tune the policy.
Historical development results remain unchanged and are not pooled here.

## New layouts and actual agent

The four layouts combine new XY offsets and cube yaw in the existing two-goal
and buffered-swap families. Agent instructions and symbolic goals retain their
previous forms: this tests changed geometry through the integration, not new
language, observed spatial reasoning, environments or robots. Native and agent
startup states evolve independently; there is no exact paired-reset certificate.

The actual CLI uses remote Caliban Qwen/Qwen3-VL-8B-Instruct, with local GPU/VLM
use disabled. It completes 8/8 independently scored object goals with valid
result contracts in all four trials. There are zero forbidden commands,
infrastructure failures, shutdown warnings or stop-boundary violations. It makes
16 model calls, totaling 116.4 seconds of inference wall time; server-default
sampling remains unpinned. Two model responses bundle discovery and planning,
so this is not perfect one-call-format compliance.

One deliberately invalid-handle request is rejected before a separate user
turn authorizes the first task. That boundary is compliant. Successful final
receipts and independent footprint/support/stillness goals are both required.

![Prospective native and agent outcomes](media/tamp_prospective_20261116.png)

Each cell is one live trial, with completed/declared transfer counts. These four
untouched within-family layouts are prospective evidence, not a broad reliability
estimate. Cameras, observed grounding, household adapters and hardware remain
unqualified. No robot/contact physics or acceptance thresholds changed.

## Native scripted recovery

The stale-plan trial moves the head through its normal controller after planning.
`execute_task_plan` rejects the old handle with `stale_task_plan` and zero new
commands, then rejects reuse with `unknown_plan`. A separately preauthorized
scripted request rediscovers, obtains a fresh plan and completes both transfers.

The second trial completes one real transfer, then injects a single no-command
planning-admission refusal at the provider boundary. The public receipt reports:

```json
{"status":"partial","code":"task_motion_budget_exhausted","data":{"execution_started":true,"completed_steps":1,"planned_steps":2,"failed_step_1based":2,"failed_phase":"preflight"}}
```

The failed handle is consumed. After fresh measured-scene admission and planning,
a new request executes **one remaining transfer**, with a successful final goal
receipt. The original partial receipt and injected-fault record remain in the
manifest. All five declared recovery boundary checks pass, including the absence
of dispatch at the injected refusal. The first completed action is counted once.

This is an explicitly injected admission/budget refusal, not an actual exhausted
solver budget, grasp loss or sensor fault. These recoveries are scripted public-
tool requests, not evidence that the LLM can recover a partial physical task.
Held-object recovery remains untested.

## Verification and provenance

**416 affected tests pass**, with zero failures/errors/skips on the frozen source
(job `20261116_074332_cafbdd`). New tests verify one-shot refusal after a successful
transfer, fresh discovery and consumed handles, and no recovery after unexpected
failure or a failed boundary. Real native episodes validate the corresponding
physical and public-receipt behavior. Ruff and whitespace checks pass. The full
paper builds to 60 pages without unresolved references or citations.

[Compact results and source/job/summary hashes](tamp_prospective_20261116.json)
retain all ten episodes and both partial/stale receipt histories. Raw protocols,
initial/final states, actuation logs and model logs remain in ignored
`.tamp-runs/prospective-20261116/`; they are not a public raw-data archive.
Reproduce the figure from the checkout:

```bash
.venv/bin/python scripts/plot_tamp_prospective.py --record docs/experiments/tamp_prospective_20261116.json --output paper/figures/tamp_prospective_20261116
```

Next: exercise between-transfer partial failure through the LLM, then observed
scene admission and exploration. Merge/review the existing stack with integration
checks after retargeting; prospective evidence does not qualify those later gates.
