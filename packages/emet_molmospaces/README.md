# emet-molmospaces

Thin wrapper for [MolmoSpaces](https://github.com/allenai/molmospaces) scenes and robots (e.g. rby1 / Galaxea R1). Depends on **emet** and **molmo-spaces** (mujoco 3.4, numpy>=2.2).

Install in a dedicated venv (to avoid numpy/mujoco version conflict with core emet):

```bash
python -m venv .venv-molmospaces
.venv-molmospaces/bin/pip install -e /path/to/emet  # or pip install emet
.venv-molmospaces/bin/pip install -e .
```

Then use from core emet: `emet molmospaces list-scenes` (core discovers `.venv-molmospaces/bin/emet-molmospaces` and runs it).

Or run directly: `.venv-molmospaces/bin/emet-molmospaces list-scenes`
