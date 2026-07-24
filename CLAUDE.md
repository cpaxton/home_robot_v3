# Claude / agent notes (emet)

Follow **`.cursorrules`** and **`.cursor/rules/`** for full project conventions. This file highlights GPU/eval launch rules that agents miss.

## Long GPU / paper evals → `emet jobs`

Do **not** start multi-hour Habitat, HM-EQA H2H, overnight batteries, or paper sweeps with unmanaged bare `nohup` when avoidable.

```bash
uv run emet eval status
uv run emet eval diagnose   # empty nvidia-smi ≠ Habitat EGL OK
uv run emet eval kill-stale # only if no intentional job is live
NEED_MIB=12000 uv run emet eval wait

uv run emet jobs run --name SHORT_NAME --need-mib 12000 -- \
  ./scripts/your_eval_script.sh …

uv run emet jobs                 # list + PROGRESS / ETA
uv run emet jobs status JOB_ID
uv run emet jobs logs JOB_ID --tail 80
uv run emet jobs cancel JOB_ID
```

- Registry: `~/runs/emet/jobs/` (`EMET_JOBS_DIR`).
- Wrapper sets `EMET_JOB_ID`. Orchestrators should call `emet jobs update … --units-done/--units-total/--phase/--current-id` and/or write `OUT/progress.json`.
- Never block a Cursor/Claude turn on Habitat/VLM or multi-hour GPU work; use `emet jobs run` and poll status/logs. Agent crashes here are usually `emet` segfaults after Habitat/EGL teardown — not a hidden CUDA process.
- Details: `docs/cli.md` (`emet jobs` / `emet eval`), `docs/evaluation.md`, `docs/known_issues.md`, `.cursor/rules/gpu-eval-workflow.mdc`.
