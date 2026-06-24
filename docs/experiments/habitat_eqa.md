# Habitat EQA (HM-EQA / OpenEQA)

Primary paper goal: reproduce GraphEQA-style metrics on HM-EQA and OpenEQA subsets.

**Deep docs:**

- [habitat/README.md](../habitat/README.md)
- [habitat_eqa.md](../habitat_eqa.md)
- Parity appendix: `paper/sections/appendix/05_habitat_eqa_parity.tex`

## Branch note

Full Habitat harness lives on **`feature/habitat-eqa-harness`**. Merge before final paper numbers.

## Metrics

MC accuracy, mean planning steps; GraphEQA vs Dynagraph vs ablations.

## Entrypoint

```bash
./scripts/install_habitat.sh
.venv-habitat/bin/emet-habitat  # see habitat/usage.md for sweep scripts
```

## Related sim benchmarks (this repo, no Habitat install)

- [ovmm_find_phase.md](ovmm_find_phase.md) — HM3D proxy find-phase
- [dynamic_exploration.md](dynamic_exploration.md) — Emet sim exploration
