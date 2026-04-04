# emet-molmospaces

Thin wrapper for [MolmoSpaces](https://github.com/allenai/molmospaces) scenes and robots (e.g. rby1 / Galaxea R1). Requires a **local editable install of emet** (emet is not on PyPI), then **molmo-spaces** (mujoco 3.4, numpy>=2.2).

Install in a dedicated venv (to avoid numpy/mujoco version conflict with core emet). **MolmoSpaces** is installed **from GitHub** (not PyPI); upstream requires **Python ≥3.11**. From the **emet repo root**:

```bash
uv venv .venv-molmospaces --python 3.11
uv pip install --python .venv-molmospaces/bin/python --upgrade pip
uv pip install --python .venv-molmospaces/bin/python --no-deps -e .
uv pip install --python .venv-molmospaces/bin/python -e packages/emet_molmospaces
```

The last line pulls `molmo-spaces` from `git+https://github.com/allenai/molmospaces.git` plus `mujoco` / `numpy` per this package’s `pyproject.toml`.

Or: `./install.sh --molmospaces -y` from the repo root.

Then use from core emet: `emet molmospaces list-scenes` (core discovers `.venv-molmospaces/bin/emet-molmospaces` and runs it).

Or run directly: `.venv-molmospaces/bin/emet-molmospaces list-scenes`
