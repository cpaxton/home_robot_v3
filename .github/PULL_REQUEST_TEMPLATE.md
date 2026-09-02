## Summary

<!-- What changed and why. 1–3 bullets. -->

## Testing instructions

A reviewer should be able to follow this without asking you. Use **copy-pasteable commands** from repo root (`uv run emet …`). Do not write “tests passed” or only tick checkboxes.

Mark a subsection **n/a** if it does not apply. For GPU / Habitat / VLM, queue with `emet jobs` — do not run Habitat or a VLM as a blocking Cursor agent command.

### CPU / unit

```bash
uv run emet test path/to/tests
```

**Expect:** (pass, specific asserts, CLI help text, …)

### GPU / sim / eval

n/a

```bash
# Example: uv run emet jobs run --name … --need-mib 8000 -- CMD
```

**Expect:**

**Do not:** inline `emet-habitat run-episode`; `emet eval kill-stale` while a live job is running; `EMET_EVAL_RERUN=1` / `--rerun` on overnight `--via-jobs` batches unless you are watching the viewer.

### Robot / Jetson

n/a

```bash
```

**Expect:**

## Checklist

- [ ] Testing instructions above are copy-pasteable (commands + expected result)
- [ ] New/changed CLI flags and `EMET_*` env vars match `emet <cmd> --help` and docs
- [ ] New Python files use the Chris Paxton Apache stub (not Hello Robot)
- [ ] Docs updated for `src/emet/app/` / `src/emet/simulation/` / new subcommands
- [ ] Self-review of the diff vs `main`

## Screenshots (if applicable)

## Additional context
