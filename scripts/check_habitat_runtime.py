# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Preflight the full EQA import stack and one rendered frame, without a VLM.

Preserved from PR #169 / fdb441a9. A bare renderer probe misses native GL
side effects introduced by controller imports.
"""

import json
from dataclasses import asdict

import emet_habitat.runner  # noqa: F401
from emet_habitat.egl_probe import run_egl_probe

if __name__ == "__main__":
    result = run_egl_probe(question_id=15)
    print(json.dumps(asdict(result)), flush=True)
    raise SystemExit(0 if result.ok else 1)
