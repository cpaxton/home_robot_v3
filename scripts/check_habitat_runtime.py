# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Check EQA's actual import stack plus a rendered frame before loading models."""

import json
from dataclasses import asdict

# Import order matters: a bare Habitat probe misses native GL side effects
# introduced by shared controller imports. Do not instantiate a learned model.
import emet_habitat.runner  # noqa: F401
from emet_habitat.egl_probe import run_egl_probe

if __name__ == "__main__":
    result = run_egl_probe(question_id=15)
    print(json.dumps(asdict(result)), flush=True)
    raise SystemExit(0 if result.ok else 1)
