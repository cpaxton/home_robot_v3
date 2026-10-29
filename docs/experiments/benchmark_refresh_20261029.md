# Updated benchmark refresh (2026-10-29)

The runtime is frozen main `cd169673ba69c5e8f63dbda4c0bcab25d373e339`,
in `.worktrees/benchmark-main-20261029`. The separate research branch supplies
the driver and new authored cases; it does not change this sweep's runtime.
The Habitat import plus actual RGB renderer preflight passed in managed job
`20261029_003735_d0e7db`. Root checkout edits and RL branches are untouched.

## Frozen population

`scripts/run_existing_benchmark_sweep.py` expands the existing registries into
237 tasks: the 39-question union of HM-EQA gate/dev IDs across lazy_graph,
dynamem, static_graph and dynagraph (156), two existing small-room find cases
across those backends (8), three Habitat nearest-v2 find proxies across those
backends (12), one native two-step privileged TAMP control, and the first-30
SQA3D recorded slice across its two supported backends (60).

EQA/OVMM use seed 0 initially; SQA3D retains its existing recorded-slice runner
and its default RNG behavior. This is a current-system refresh and comparison
of complete memory systems, not an isolated ingestion ablation, new holdout,
full upstream benchmark, or multi-seed significance test. The 39 IDs have been
used during development. Native TAMP is privileged/scripted physical feasibility,
not learned discovery or language planning. Find is not successful manipulation.
No ground-truth graph row is pooled with learned perception. Room graph rows
explicitly request sensor perception; their raw settings are retained.

The legacy dynamic-exploration relocation/assisted protocol requires a separate
physics/provenance audit. RoboCasa needs a qualified readable-object replacement;
the archived tiny-object case stays a declared stress case. Both gaps are listed
in the manifest rather than silently counted as evaluated.

## Admission and resuming

The coordinator holds no GPU. It waits for other registered GPU jobs or compute
processes, takes the existing exclusive lock per case, uses stable CPU 12 and
single-thread libraries, and yields 30 seconds between cases. A seven-hour
window limits admission; it does not shorten episode budgets or fabricate
completion. This matrix takes multiple windows. Create `PAUSE` in the output
directory to stop after the current episode. The default disk reserve is 20 GiB.
Three consecutive infrastructure failures stop admission for diagnosis.

```bash
# Run this coordinator itself through emet jobs --no-gpu-exclusive --cpu-safe.
PYTHONPATH=src .venv/bin/python scripts/run_existing_benchmark_sweep.py \
  --source /home/cpaxton/src/home_robot_v3/.worktrees/benchmark-main-20261029 \
  --output /home/cpaxton/runs/emet/benchmark-refresh-20261029 \
  --python /home/cpaxton/src/home_robot_v3/.venv/bin/python \
  --habitat /home/cpaxton/src/home_robot_v3/.venv-habitat/bin/emet-habitat \
  --max-hours 7 --execute
```

Omit `--execute` to materialize the matrix only. Repeating the same command
resumes unattempted entries; attempted failures remain in the ledger. Source,
driver, population and configuration changes are rejected on resume. Retries
need a new declared run, not replacement of a bad row. Offline model caches
and `--no-download` prevent accidental large downloads during the sweep.
Missing SQA3D SENS assets are infrastructure gaps, not permission to silently
switch to mesh rendering. No cloud judge is invoked.

## Evidence and figures

Keep the complete assignment manifest and ledger, process logs, per-case config,
metric artifacts, EQA debug-bundle references, exact saved model requests where
available, and native TAMP controls. New live authored cases additionally record
actual OpenAI-compatible request/response bodies, RGB-D, camera calibration,
base/head poses, and scorer-only physics snapshots. A protocol mismatch or
missing artifact is distinct from an incorrect answer or failed localization.

Prepare five figures after auditing the traces:

1. **Task outcomes and cost:** EQA accuracy, independent object/destination/both
   localization, TAMP completion; show assigned/completed/error/pending counts.
   Pair methods on matching cases. Report wall and available simulation time
   separately; missing simulation time stays missing.
2. **Evidence-to-action chain:** target visible in a captured image, image
   actually supplied to the model, semantic decision, accepted geometry,
   measured motion, final task result. Do not infer visibility from success.
3. **Top-down exploration:** start/route/headings, requested/reached viewpoints,
   observed/unknown/occupied floor and target positions, synchronized key RGBs.
4. **A recovery and a retained failure:** question/instruction, ordered image
   packets and their roles, tool feedback, measured before/after motion, final
   decision. Select by declared failure families, not flattering screenshots.
5. **Memory utility versus cost:** nodes/observations, retained target recall,
   task quality and runtime. A smaller graph is not itself better memory.

Publish a machine-readable figure input table with source/model/config hashes
and artifact paths. Existing matched pilot data can supply historical context,
but do not pool its different model endpoints or legacy decision policy with
this refresh. The final system assessment should distinguish tooling correctness,
capability evidence, task performance, and unresolved environment coverage.
