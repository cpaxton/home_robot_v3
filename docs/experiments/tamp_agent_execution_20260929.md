# Agent execution and broader TAMP coverage

Work follows benchmark PR #177 on branch `feat/tamp-agent-execution`.
Baseline remains 21/25 fresh GT/MCTS successes on 25/200 admitted controls.
Neither learned-agent nor physical acceptance is established.

## Initial experiment (cancelled; see live finding below)

Managed job `20260929_131223_7489e3` executes serially:

1. Three original admitted RBY1 tasks with diagnostic-only lift evidence, from
   source `f3114ddd`: scene00 cleanup slots 0 and 1, scene12 cleanup slot 1.
   The original r3 scene, certificates, geometry, scoring and tolerances remain
   fixed. Only executor diagnostic logging and tests change relative to r3.
   Logs record target/observed/pre-lift positions, vertical displacement,
   verification bounds, command step and session step. These distinguish stale
   observations from actual failed lifts without counting a timeout as success.
2. The expanded 200-candidate admission run from tested source `1b5b2da6`, then
   fresh MCTS replay of its admitted registry. Uses the bounded object-size
   fallback validated by the 3/4 admission, 3/3 replay pilot. Observation startup
   budget is explicitly 120 seconds (the Mars diagnostic reached observations
   with this budget); per-case wall limit remains 900 seconds. This is a separate
   benchmark version and cannot replace r3's original results.

Durable root: `~/runs/emet/tamp-validated-fixtures-20260928/`.
Outputs: `lift-evidence-r5/`, `full-r4/`, `replay-r4/`, and
`execution-expansion-20260929-job/`. This run was cancelled before expansion;
the replacement job and findings are recorded below.

The new diagnostics pass the controller and agent-tool regression slice
(24 tests). An existing test mock now explicitly disables query-driven memory,
so its truthy automatic mock attribute cannot route the test down the wrong
manipulation path. This does not change product behavior.

## Paper figures

`scripts/plot_tamp_benchmark.py` exports PDF/SVG/300-dpi PNG, source counts CSV,
input hashes, and captions. It refuses partial runs or replay/certificate mismatches.
The initial figures use terminal r3 results and are stored in
`~/runs/emet/tamp-validated-fixtures-20260928/paper-figures-r3/`:

- `tamp_coverage_and_success`: admission coverage and conditional MCTS outcomes.
- `tamp_v2_stretch_scene00_nav_goal_2`: initial/reference navigation layout.
- `tamp_v2_rby1_scene00_cleanup_0`: initial/reference cleanup layout. The reference
  succeeds while the original MCTS replay fails; do not caption it as MCTS success.

Layout panels show the declared disk approximation and measured reference object
positions. They are not collision-mesh renders or evidence of physical execution.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python scripts/plot_tamp_benchmark.py \
  --admission "$HOME/runs/emet/tamp-validated-fixtures-20260928/full-r3" \
  --replay "$HOME/runs/emet/tamp-validated-fixtures-20260928/replay-r3" \
  --output-dir /tmp/tamp-paper-figures \
  --example-id tamp_v2_stretch_scene00_nav_goal_2 \
  --example-id tamp_v2_rby1_scene00_cleanup_0
```

## Next performance gate

Use measured failure traces to repair execution, then run paired trials on all
25 original tasks, retaining failures and original denominators. Report action
budgets and repeats; do not evaluate only the cases that improved. The expanded
corpus measures coverage separately. Validate the shared agent tool sequence on
these fixtures before attributing changes to model reasoning. Actual learned
agent runs must record model, prompt, observations, action traces and budgets;
scripted tool calls and oracle identities must remain explicitly labeled.

## Live finding: invalid lift acceptance

Job `20260929_131223_7489e3` was cancelled after its first completed case exposed
an invalid success condition. The apple was 1.0166 m from its lift target but
accepted because Z increased by 0.1870 m. Server attachment offsets were about
1.39–1.65 m for the apple and 2.51 m for the potato. Command/session steps matched
at verification; this alone does not establish fresh or correct physical state.
The evidence does not yet isolate the frame, controller, or attachment race cause.

Original outcomes remain historical final-state control scores. In particular,
21/25 is not verified grasp or agent success. Paper captions have this qualification.
The completed first case and server log are retained in `lift-evidence-r5/`;
`termination.json` explains that remaining ledger running/pending entries are
interrupted/unexecuted. No r4 full sweep was started by that cancelled job.

Candidate guard `c428280d` requires measured-joint FK arrival within the existing
IK tolerance before motion is declared complete, and removes height-only lift
acceptance. This can lower the recorded score while exposing real failures.
The controller/agent-tool slice passes 25 tests. It is not yet a demonstrated
live reliability improvement.

Replacement managed job `20260929_132011_913a74` first evaluates the guard on
original RBY1 scene00 cleanup slot 0 using frozen variant `d245693e`, then runs
90 oracle-backed construction requests (Stretch, Mars, Nori) and fresh replay
from the tested r4 source. It retains the 120-second startup budget and 900-second
case wall limit. Outputs: `measured-arrival-r6/`, `oracle-expansion-r4/`,
`oracle-replay-r4/`, `measured-oracle-20260929-job/`. RBY1 expansion remains gated
on fixing measured execution. New outcomes are pending.

Final follow-up regression check: 95 controller, agent-tool, task-search, MCTS,
and physical-motion contract tests passed. Live measured-arrival acceptance and
expanded oracle results are still pending; no performance gain is claimed yet.
