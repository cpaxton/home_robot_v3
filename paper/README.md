# Dynagraph CoRL 2026 paper

LaTeX scaffold for the Dynagraph paper (CoRL 2026 submission format).

## Build

From the repo root:

```bash
./paper/build.sh
```

Or from this directory:

```bash
./build.sh
```

Remove auxiliary files:

```bash
./paper/build.sh --clean
```

Requires `latexmk` + `booktabs` (via `texlive-latex-extra`, `texlive-bibtex-extra` on Ubuntu). If TeX is not installed locally, `./build.sh` falls back to **Docker** (`texlive/texlive:latest`) when `docker` is on your PATH.

## Layout

- `main.tex` — document root (anonymous submission mode)
- `corl_2026.sty`, `corlabbrvnat.bst` — official CoRL 2026 template files
- `sections/` — one file per section (`00_abstract` … `06_conclusion`)
- `sections/appendix/` — robot platforms, Innate Mars, Depth Anything 3, ground-truth graph (04), Habitat HM-EQA parity (05), agentic EQA tools (07)
- `figs/` — paper figures (PNG; tracked via `.gitignore` exceptions)
- `data/` — **minimal** checked-in result summaries (not full episode traces); see [`data/README.md`](data/README.md)
- `references.bib` — bibliography stubs (expand before submission)

## Benchmarks and results

- **Paper experiments (start here):** [docs/experiments/README.md](../docs/experiments/README.md)
- **Habitat HM-EQA results vs prior art:** [docs/experiments/habitat_eqa_results.md](../docs/experiments/habitat_eqa_results.md)
- **Classic vs agentic-verify summaries:** [data/hmeqa_agentic_h2h/](data/hmeqa_agentic_h2h/) + [data/README.md](data/README.md)
- **Detailed operator runbook:** [docs/paper_benchmarks.md](../docs/paper_benchmarks.md)
- **Experiments plan (LaTeX):** `sections/04_experiments.tex`
- **Results tables:** `sections/05_results.tex`, `sections/appendix/05_habitat_eqa_parity.tex`

### Reproduce agentic H2H bars from checked-in JSON

```bash
uv run python scripts/summarize_hmeqa_agentic_h2h.py \
  --from-summary paper/data/hmeqa_agentic_h2h/holdout8_summary.json \
  --output paper/figs/hmeqa_agentic_h2h.png
uv run python scripts/summarize_hmeqa_agentic_h2h.py \
  --from-summary paper/data/hmeqa_agentic_h2h/balanced32_summary.json \
  --output paper/figs/hmeqa_agentic_h2h_bal32.png
```

Full Habitat re-run (maps + coverage figure): see `paper/data/README.md`.

## Related work

- Implementation: `emet run dynagraph` on branch `main`
- Habitat EQA harness: `feature/eval-diagnostics-smoke` (nav + grounding fixes); merge plan in [habitat_eqa_results.md](../docs/experiments/habitat_eqa_results.md)
- SQA3D harness: `docs/sqa3d.md` on branch `feature/dynamem-offline-real-benchmark`
