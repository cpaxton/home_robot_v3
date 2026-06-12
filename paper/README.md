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
- `sections/appendix/` — robot platforms, Innate Mars, Depth Anything 3, ground-truth graph (04), Habitat HM-EQA parity (05)
- `references.bib` — bibliography stubs (expand before submission)

## Benchmarks and results

- **Run / maintain paper sweeps:** [docs/paper_benchmarks.md](../docs/paper_benchmarks.md)
- **Experiments plan (LaTeX):** `sections/04_experiments.tex`
- **Results tables (fill manually from CSV):** `sections/05_results.tex`

## Related work

- Implementation: `emet run dynagraph` on branch `main`
- Habitat EQA harness plan: `docs/plans/HABITAT_EQA_HARNESS.md` on branch `feature/habitat-eqa-harness`
- SQA3D harness: `docs/sqa3d.md` on branch `feature/dynamem-offline-real-benchmark`
