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

Requires a TeX distribution with `latexmk`, `natbib`, and standard packages (on Ubuntu: `texlive-latex-extra`, `texlive-bibtex-extra`, `latexmk`).

## Layout

- `main.tex` — document root (anonymous submission mode)
- `corl_2026.sty`, `corlabbrvnat.bst` — official CoRL 2026 template files
- `sections/` — one file per section (`00_abstract` … `06_conclusion`)
- `references.bib` — bibliography stubs (expand before submission)

## Related work

- Implementation: `emet run dynagraph` on branch `main`
- Habitat EQA harness plan: `docs/plans/HABITAT_EQA_HARNESS.md` on branch `feature/habitat-eqa-harness`
