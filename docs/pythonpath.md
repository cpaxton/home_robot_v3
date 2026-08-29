# PYTHONPATH sanitizer

`emet.utils.pythonpath` rewrites `PYTHONPATH` / `sys.path` so **this repo’s `.venv` wins**
when `emet` launches a child (`emet serve mujoco`, `emet run …`) or when
`mujoco_server` starts.

**Why it exists, which function to call, and the call sites** are in the module
docstring of [`src/emet/utils/pythonpath.py`](../src/emet/utils/pythonpath.py).
This page is the operator checklist.

It is **not** an `EMET_*` toggle. The shell `PYTHONPATH` is the input; the sanitizer
filters and prepends.

## What it does

1. **Drops ROS entries** (`/opt/ros/`, typical Humble/Jazzy/Noetic paths) so a
   stub `cv2` cannot shadow `opencv-contrib-python` in the venv.
2. **Prepends** `src/` and the venv `site-packages` for the **active interpreter
   only** (tag from `.venv/pyvenv.cfg` `version_info`, e.g. `3.10` →
   `.venv/lib/python3.10/site-packages`).

Callers: `sanitize_emet_subprocess_env` (`emet` CLI bootstrap, `emet robots`,
sim-eval spawners) and `ensure_venv_site_packages_first` (in-process, including
`mujoco_server` before OpenCV/MuJoCo).

## Why the interpreter tag (not `python*/site-packages`)

A glob of every `python*` dir under `.venv/lib` will prepend **all** of them.
If a Python 3.10 venv still has a leftover `.venv/lib/python3.12/site-packages`
(partial `uv`/`venv` mix, an old extra, a copied tree), those 3.12 wheels load
first. Binary extensions (`scipy`, `numpy`) then fail with an **ABI mismatch**
(`undefined symbol`, `module compiled against API version …`) the moment the
sim server imports them — often *before* `ensure_venv_site_packages_first()`
runs, because `mujoco_server` imports numpy at module load.

The sanitizer therefore prepends only `python{major.minor}/site-packages` for
the venv’s own tag. If `version_info` is missing, it falls back to the old glob.

## Debug

```bash
# What the sanitizer would put on a child
uv run python -c "
from emet.utils.pythonpath import sanitize_emet_subprocess_env
print(sanitize_emet_subprocess_env().get('PYTHONPATH',''))
"

# Confirm scipy/numpy come from this venv
uv run python -c "import scipy, numpy; print(scipy.__file__); print(numpy.__file__)"
```

Those paths should sit under `.venv/lib/python3.10/` (or whatever `uv run python -V`
reports). If they do not: `unset PYTHONPATH`, avoid extra `python3.*` dirs in
`.venv/lib`, and launch with `uv run emet …` from the repo.

ROS/`cv2` shadowing: [dynagraph_dynamic_memory.md](experiments/dynagraph_dynamic_memory.md)
(unset `PYTHONPATH` for evals). OpenCV check: `emet.utils.opencv_import`.
