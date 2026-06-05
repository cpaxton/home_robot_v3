# emet-habitat

Habitat-Sim harness that drives **emet** `GraphEQAController` / `DynagraphController` on HM-EQA-style benchmarks.

Install from repo root:

```bash
./scripts/install_habitat.sh
```

Run a single episode (mock LLM, no GPU VLM):

```bash
.venv-habitat/bin/emet-habitat run-episode \
  --dataset hmeqa --question-id 0 --method dynagraph --mock-llm
```

Requires HM3D scenes and Explore-EQA CSVs — see `docs/habitat_eqa.md`.
