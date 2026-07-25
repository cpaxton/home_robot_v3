# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Map MuJoCo body / placement labels to MolmoSpaces Thor-style grasp asset ids."""

from __future__ import annotations

import re
from pathlib import Path

# Mesh child suffix: Apple_4_1_1_0 → try Apple_4; hash bodies: bowl_<hash>_1_1_0.
_INSTANCE_SUFFIX_RE = re.compile(r"(_\d+){1,3}$")
_HASH_BODY_RE = re.compile(r"^([a-zA-Z]+)_[0-9a-f]{8,}(_\d+)*$", re.IGNORECASE)
_THOR_ASSET_RE = re.compile(r"^([A-Za-z]+(?:_[A-Za-z]+)*)_(\d+)(?:_\d+)*$")


def strip_molmo_instance_suffix(body_name: str) -> str:
    """Strip trailing ``_1_1_0``-style instance tokens from a Molmo body name."""
    name = str(body_name).strip()
    # Prefer dropping trailing _N_N_N groups (mesh children).
    while True:
        m = re.search(r"_\d+_\d+_\d+$", name)
        if not m:
            break
        name = name[: m.start()]
    # Also drop a single trailing _N if it looks like an instance index after Category_N.
    m2 = re.search(r"^(.+_\d+)_\d+$", name)
    if m2 and _THOR_ASSET_RE.match(m2.group(1)):
        name = m2.group(1)
    return name


def candidate_asset_ids(body_name: str, *, category: str | None = None) -> list[str]:
    """Ordered candidate asset folder names for grasp lookup."""
    out: list[str] = []
    raw = str(body_name).strip()
    if not raw:
        return out
    stripped = strip_molmo_instance_suffix(raw)
    for cand in (raw, stripped):
        if cand and cand not in out:
            out.append(cand)
        m = _THOR_ASSET_RE.match(cand)
        if m:
            tid = f"{m.group(1)}_{m.group(2)}"
            if tid not in out:
                out.append(tid)
    # Hash-style Objaverse bodies: keep full stem without mesh suffix for droid_objaverse.
    hm = _HASH_BODY_RE.match(stripped)
    if hm and stripped not in out:
        out.append(stripped)
    cat = (category or "").strip()
    if cat:
        # Title-case single token categories → Apple; multi-word → Apple (first token).
        token = cat.replace("-", " ").replace("_", " ").split()[0]
        titled = token[:1].upper() + token[1:].lower() if token else ""
        if titled and titled not in out:
            out.append(titled)
    return out


def resolve_asset_id_against_grasps_dir(
    body_name: str,
    grasps_dir: Path,
    *,
    category: str | None = None,
) -> str | None:
    """Return the first candidate that has a grasp folder under ``grasps_dir``/{droid,droid_objaverse,rum}."""
    roots = [
        Path(grasps_dir) / "droid",
        Path(grasps_dir) / "droid_objaverse",
        Path(grasps_dir) / "rum",
    ]
    for cand in candidate_asset_ids(body_name, category=category):
        for root in roots:
            if not root.is_dir():
                continue
            # Exact folder
            if (root / cand).is_dir():
                return cand
            # Case-insensitive / category prefix (Apple → first Apple_*)
            lower = cand.lower()
            matches = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.lower() == lower)
            if matches:
                return matches[0]
            if "_" not in cand:
                prefix_matches = sorted(
                    p.name for p in root.iterdir() if p.is_dir() and p.name.lower().startswith(lower + "_")
                )
                if prefix_matches:
                    return prefix_matches[0]
    return None
