# Habitat install

Habitat-Sim conflicts with the main MuJoCo stack (numpy / Python versions) and **has no usable Linux wheels on PyPI**. Use an isolated micromamba environment.

## Requirements

- Linux x86_64 (tested on Ubuntu)
- `curl`, network access
- ~2GB download for conda packages (first install)
- Optional GPU for faster rendering (headless EGL works without a display)

## Install

From the repo root:

```bash
./scripts/install_habitat.sh
```

This script:

1. Bootstraps **micromamba** into `.micromamba/` if it is not already on `PATH`
2. Creates **`.venv-habitat`** with **Python 3.10**
3. Installs `habitat-sim` from **`aihabitat-nightly`** (headless + bullet by default)
4. Installs editable **`emet`** (no-deps) + **`emet-habitat`** + pip runtime deps from `packages/emet_habitat/requirements-pip.txt` (includes **`fla-core`** + **`flash-linear-attention`** for Qwen3.5 Gated DeltaNet kernels)
5. Verifies Habitat + FLA imports (`fla.ops.gated_delta_rule`); fails if kernels are missing so Qwen3.5 cannot silently fall back to multi-hour PyTorch ops

No system-wide `conda` is required.

### Headless vs GUI

Default is headless EGL (`HABITAT_HEADLESS=1`). For a GUI build:

```bash
HABITAT_HEADLESS=0 ./scripts/install_habitat.sh
```

## Verify

```bash
.venv-habitat/bin/python -c "import habitat_sim; print('habitat_sim OK', habitat_sim.__version__)"
.venv-habitat/bin/python -c "import fla.ops.gated_delta_rule, triton; print('FLA OK')"
.venv-habitat/bin/emet-habitat info
uv run emet habitat info
```

`emet-habitat` is a bash wrapper that prepends `.venv-habitat/lib` to `LD_LIBRARY_PATH`
(conda `libstdc++` vs system `CXXABI`). Use `.venv-habitat/bin/emet-habitat`, not bare
`python -m emet_habitat.cli`, if matplotlib import fails.

`emet habitat …` and `emet run graph-eqa-habitat` delegate to `.venv-habitat/bin/emet-habitat` from the main `.venv`.

## Why not the main `.venv`?

| Issue | Detail |
|-------|--------|
| PyPI | Only an old macOS wheel; Linux needs conda |
| Python | Nightly builds target **3.10**; main emet may use newer Python |
| numpy | Habitat and MuJoCo/Robocasa pin different numpy stacks |

The pattern matches **MolmoSpaces** (`.venv-molmospaces`): core CLI in main env, sim in a sub-env.

## Reinstall / update

```bash
rm -rf .venv-habitat
./scripts/install_habitat.sh
```

`.micromamba/` can be kept between reinstalls.

## Next step

Download HM-EQA CSVs and HM3D scenes: [data.md](data.md).
HM3D train/val/minival need API tokens: **Profile → Settings → Developer Tools** ([link](https://my.matterport.com/settings/account/devtools)).
