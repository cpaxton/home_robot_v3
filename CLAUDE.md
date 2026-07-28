# Claude / agent notes (emet)

Follow **`.cursorrules`** and **`.cursor/rules/`** for full project conventions. This file highlights GPU/eval launch rules that agents miss.

## Long GPU / paper evals → `emet jobs`

Do **not** start multi-hour Habitat, HM-EQA H2H, overnight batteries, or paper sweeps with unmanaged bare `nohup` when avoidable. Never as a blocking inline command in an agent turn — native GPU/EGL teardown crashes the agent process even when a detached child would have finished.

**Before Habitat:** `uv run emet habitat safe-start` (recover + detached jobs-wrapped EGL probe, no VLM). Wait until that job is `done` + logs OK, then `emet hmeqa h2h` / `overnight` via jobs.

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
- After an agent death: **`uv run emet status tail`** from the owning checkout (literal `next:` command; do not use a flat `~/runs/emet/STATUS.log` shared across v2/v3/v4), then `emet jobs` / `~/runs/emet/`.
- **Record before risky sim/GPU steps** (launch, resume, `kill-stale`, affinity experiments): leave job id, OUT, commit, and copy-paste resume via `emet status` / `scripts/status_log.sh` / `segfault.md`. See `.cursor/rules/gpu-eval-workflow.mdc`.

## Two segfault modes (do not conflate)

`docs/known_issues.md` has the full write-up.

- **Mode A — episode `libcuda` SIGSEGV (`exit=139`):** `emet-habitat run-episode` dies during Qwen3-VL vision generate while Habitat-Sim EGL shares the GPU. Orchestrator logs `FAIL … exit=139` / `dumped core`; kernel shows `python[…]: segfault … in libcuda.so`. The per-qid jsonl is **empty** — a crash, not a scored miss. Hot scenes: q104/q105 (`yogvKWUrdnw`), flaky q68.
- **Mode B — agent / `emet` null-IP SIGSEGV:** the Cursor/Claude agent process itself dies (`emet[…]: segfault at 0`, or `trap invalid opcode`) after a turn runs or probes Habitat / tears down a GPU context. The detached `emet jobs` child often keeps running — check it before re-launching.

## Rules

- Never block an agent turn on Habitat/VLM or multi-hour GPU work; use `emet jobs run` and poll `emet jobs` status/logs — do not long-`AwaitShell` on GPU work.
- Do not run a "quick" Habitat smoke inside a turn to verify EGL, then `kill` it — hard mid-episode kills precede the next `emet` segfault. **Empty `nvidia-smi` ≠ EGL healthy.**
- HM-EQA H2H scripts self-guard via `EGL_FAIL_ABORT` (default 2) and crash policy / streak abort (writes `native_crash_*.log`). Leave these on; treat empty per-qid jsonl as a crash to retry after the GPU recovers, not a miss.
- After a crash: `sudo dmesg -T | rg 'segfault|invalid opcode|libcuda'` · `uv run emet eval diagnose` · `uv run emet jobs` · check `~/runs/emet/`, `~/.cache/habitat_eqa/results/`.
- Details: `docs/cli.md` (`emet jobs` / `emet eval`), `docs/evaluation.md`, `docs/known_issues.md`, `.cursor/rules/gpu-eval-workflow.mdc`.
