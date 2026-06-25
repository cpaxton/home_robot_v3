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

## Frame / map diagnostics

Voxel-map coordinates for Habitat are documented in [habitat/README.md](../habitat/README.md) (voxel-world section). Before using `topdown_map.png` in paper figures, audit bundles with [`scripts/audit_habitat_voxel_map.py`](../../scripts/audit_habitat_voxel_map.py) — see [evaluation.md](../evaluation.md#habitat-frame-sanity-before-trusting-map-colors).

## Related sim benchmarks (this repo, no Habitat install)

- [ovmm_find_phase.md](ovmm_find_phase.md) — HM3D proxy find-phase
- [dynamic_exploration.md](dynamic_exploration.md) — Emet sim exploration
