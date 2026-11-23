# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Identity of the current host's monotonic time domain."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path


def local_monotonic_clock_id():
    """Linux boot + time namespace identify this monotonic clock, without hostnames."""
    try:
        return _local_monotonic_clock_id()
    except OSError:
        return None


@lru_cache(maxsize=1)
def _local_monotonic_clock_id():
    # Exceptions are not cached: transient procfs failures can recover.
    identity = Path("/proc/sys/kernel/random/boot_id").read_text().strip() + os.readlink("/proc/self/ns/time")
    return hashlib.sha256(identity.encode()).hexdigest()
