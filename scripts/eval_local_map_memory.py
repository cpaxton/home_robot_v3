# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Independent-process depth-map memory/time control with exact cell checksum."""

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from emet.motion.observed_workspace import ObservedWorkspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path)
    args = parser.parse_args()
    cls = ObservedWorkspace
    if args.reference_source:
        spec = importlib.util.spec_from_file_location('reference_workspace', args.reference_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls = module.ObservedWorkspace
    depth = np.full((121, 161), 1., np.float32)
    depth[55:60, 70:75] = 0
    start = time.perf_counter()
    stats = {}
    grid = cls.from_depth(depth, [[150, 0, 80], [0, 150, 60], [0, 0, 1]], np.eye(4),
        workspace=[[-.4, -.3, .2], [.4, .3, .8]], resolution=.005, max_cells=3_000_000,
        **({} if args.reference_source else {'build_stats': stats}))
    result = {'reference': str(args.reference_source) if args.reference_source else None,
        'cells': int(grid.cells.size), 'cell_sha256': hashlib.sha256(grid.cells.tobytes()).hexdigest(),
        'seconds': time.perf_counter()-start,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'build_stats': stats}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
