# Claude / agent notes (emet)

Follow **`.cursorrules`** and **`.cursor/rules/`** for full project conventions. This file highlights the GPU/eval launch rules that agents miss and that cause session crashes.

## Long GPU / paper evals → `emet jobs` or nohup, never inline

Do **not** start multi-hour Habitat, HM-EQA H2H, overnight batteries, or paper sweeps as a blocking inline command in an agent turn. Native GPU/EGL teardown crashes the agent process even when a detached child would have finished.

```bash
uv run emet eval status        # free VRAM + compute apps (read-only)
uv run emet eval kill-stale    # only if no intentional GPU job is live
NEED_MIB=12000 uv run emet eval wait

uv run emet jobs run --name SHORT_NAME --need-mib 12000 -- \
  env EMET_ALLOW_SDPA_ATTN=1 ./scripts/run_hmeqa_agentic_h2h.sh OUT_DIR

uv run emet jobs                 # list registered jobs
uv run emet jobs status JOB_ID
uv run emet jobs logs JOB_ID
uv run emet jobs cancel JOB_ID   # not raw kill -9 on emet-habitat
```

- Registry: `~/runs/emet/jobs/` (`EMET_JOBS_DIR`); the wrapper sets `EMET_JOB_ID`.
- Verify CLI flags with `--help` before suggesting them. `emet eval diagnose` and `emet jobs` progress/ETA live on the eval branch (`exp/agentic-hmeqa-*`), not this branch.

## Two segfault modes (do not conflate)

`docs/known_issues.md` has the full write-up.

- **Mode A — episode `libcuda` SIGSEGV (`exit=139`):** `emet-habitat run-episode` dies during Qwen3-VL vision generate while Habitat-Sim EGL shares the GPU. Orchestrator logs `FAIL … exit=139` / `dumped core`; kernel shows `python[…]: segfault … in libcuda.so`. The per-qid jsonl is **empty** — a crash, not a scored miss. Hot scenes: q104/q105 (`yogvKWUrdnw`), flaky q68.
- **Mode B — agent / `emet` null-IP SIGSEGV:** the Cursor/Claude agent process itself dies (`emet[…]: segfault at 0`, or `trap invalid opcode`) after a turn runs or probes Habitat / tears down a GPU context. The detached `emet jobs` / `nohup` child often keeps running — check it before re-launching.

## Rules

- Never block an agent turn on Habitat/VLM or multi-hour GPU work; use `emet jobs run` / `nohup` and poll `emet jobs` status/logs — do not long-`AwaitShell` on GPU work.
- Do not run a "quick" Habitat smoke inside a turn to verify EGL, then `kill` it — hard mid-episode kills precede the next `emet` segfault. **Empty `nvidia-smi` ≠ EGL healthy.**
- HM-EQA H2H scripts self-guard via `EGL_FAIL_ABORT` (default 2) and `NATIVE_CRASH_ABORT` (default 1, writes `native_crash_*.log`). Leave these on; treat empty per-qid jsonl as a crash to retry after the GPU recovers, not a miss.
- After a crash: append evidence + recovery checklist to repo-root **`segfault.md`** (`tail -n 80`), then `journalctl -k` / `dmesg` · `uv run emet eval status` · `uv run emet jobs` · check `~/runs/emet/`. See `.cursor/rules/segfault-log.mdc`.
- Details: `docs/cli.md` (`emet jobs` / `emet eval`), `docs/evaluation.md`, `docs/known_issues.md`, `.cursor/rules/gpu-eval-workflow.mdc`.
