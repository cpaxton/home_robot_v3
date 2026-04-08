# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Agent configuration loader. Loads YAML configs from this directory and merges overrides.

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_AGENTS_DIR = Path(__file__).resolve().parent

# Pre-defined agent config names
AGENT_CONFIGS = {
    "default_scene_graph": _AGENTS_DIR / "default_scene_graph.yaml",
    "cpu_scene_graph": _AGENTS_DIR / "cpu_scene_graph.yaml",
}


def load_agent_config(
    name: str = "default_scene_graph",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load an agent configuration by name or file path.

    Args:
        name: config name (e.g. "default_scene_graph") or path to a YAML file
        overrides: dict of dotted-key overrides, e.g. {"segmentation.primary": "sam2"}

    Returns:
        Merged configuration dict
    """
    path = AGENT_CONFIGS.get(name)
    if path is None:
        path = Path(name)
    if not path.exists():
        raise FileNotFoundError(f"Agent config not found: {name} (tried {path})")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    return cfg


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply dotted-key overrides to a nested dict."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value
    return cfg
