# emet-habitat

Habitat-Sim harness that drives **emet** `GraphEQAController` / `DynagraphController` on HM-EQA-style benchmarks.

Install from repo root (bootstraps **micromamba** if needed; **habitat-sim is not on PyPI for Linux**):

```bash
./scripts/install_habitat.sh
```

Uses Python 3.10 + `aihabitat-nightly` conda channel. Creates `.venv-habitat/` and `.micromamba/`.

Run a single episode (mock LLM, no GPU VLM):

```bash
.venv-habitat/bin/emet-habitat run-episode \
  --dataset hmeqa --question-id 0 --method dynagraph --mock-llm
```

Force navigation each planning step (movement / diagnostics smoke; requires `--mock-llm`):

```bash
.venv-habitat/bin/emet-habitat run-episode \
  --dataset hmeqa --question-id 3 --method dynagraph \
  --mock-llm --mock-llm-explore --max-planning-steps 5
```

Requires HM3D scenes and Explore-EQA CSVs — see `docs/habitat/README.md` (`docs/habitat/data.md` for downloads).
