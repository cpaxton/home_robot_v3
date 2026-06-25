# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unified nested config loader (OmegaConf merge + legacy flat dynav compatibility)."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import yaml
from omegaconf import OmegaConf

import emet
from emet.config.embodied_agent_config import EmbodiedAgentConfig
from emet.config.rerun_config import RerunAgentConfig
from emet.config.sim_launch_config import SimLaunchConfig, decode_sim_launch_config, load_sim_launch_config_from_path
from emet.utils.config import resolve_config_yaml_path

_PACKAGE_PREFIX = "package://emet/config/"
_DEFAULT_CONFIG_ENV = "EMET_CONFIG"
_DEFAULT_CONFIG_REL = "configs/emet/default.yaml"
_LEGACY_DYNAV_CONFIG = "dynav_config.yaml"

# Top-level keys in nested emet configs (not folded into ``mapping`` during legacy normalization).
_RESERVED_TOP_LEVEL_KEYS = frozenset(
    {
        "defaults",
        "extends",
        "robot",
        "connection",
        "mapping",
        "agent",
        "sim",
        "sim_config",
        "embodied_agent",
        "rerun",
        "robots",
        "zmq",
    }
)

# Signature keys for flat dynav YAML (``dynav_config.yaml`` and copies).
_DYNAV_SIGNATURE_KEYS = frozenset(
    {
        "encoder",
        "voxel_size",
        "depth_source",
        "eqa",
        "detection",
        "motion_planner",
        "instance_memory",
        "use_instance_memory",
        "dynagraph_merge_xy_m",
    }
)

_CHAT_AGENT_SUBKEYS = frozenset({"llm", "eqa", "discord", "share_memory_vllm", "prompt", "device", "max_tokens"})


def default_config_path() -> str:
    """Return default config path (``EMET_CONFIG`` or packaged ``configs/emet/default.yaml``)."""
    env = os.environ.get(_DEFAULT_CONFIG_ENV, "").strip()
    if env:
        return env
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / _DEFAULT_CONFIG_REL
    if candidate.is_file():
        return str(candidate)
    return _LEGACY_DYNAV_CONFIG


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _emet_config_dir() -> Path:
    return Path(emet.__path__[0]).resolve() / "config"


def _resolve_config_reference(ref: str) -> str:
    """Resolve ``package://``, repo-relative, cwd-relative, or absolute paths."""
    ref = str(ref).strip()
    if ref.startswith(_PACKAGE_PREFIX):
        rel = ref[len(_PACKAGE_PREFIX) :]
        path = _emet_config_dir() / rel
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError(f"Packaged config not found: {ref!r} -> {path}")
    if ref.startswith("configs/"):
        repo_path = _repo_root() / ref
        if repo_path.is_file():
            return str(repo_path.resolve())
    try:
        return resolve_config_yaml_path(ref)
    except FileNotFoundError:
        repo_path = _repo_root() / ref
        if repo_path.is_file():
            return str(repo_path.resolve())
        raise


def _load_yaml_file(path: str) -> dict[str, Any]:
    full = _resolve_config_reference(path)
    with open(full, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path!r}")
    return raw


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _omega_to_plain(obj: Any) -> Any:
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _is_chat_agent_section(agent_val: Any) -> bool:
    if not isinstance(agent_val, dict):
        return False
    return bool(_CHAT_AGENT_SUBKEYS.intersection(agent_val.keys()))


def normalize_legacy_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    """Wrap flat dynav YAML under ``mapping:`` when needed."""
    if not raw:
        return {"mapping": {}}
    if isinstance(raw.get("mapping"), dict):
        return copy.deepcopy(raw)

    has_dynav_signature = bool(_DYNAV_SIGNATURE_KEYS.intersection(raw.keys()))
    if not has_dynav_signature:
        return copy.deepcopy(raw)

    mapping: dict[str, Any] = {}
    top: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _RESERVED_TOP_LEVEL_KEYS:
            if key == "agent" and _is_chat_agent_section(value):
                top[key] = copy.deepcopy(value)
            elif key == "agent":
                mapping[key] = copy.deepcopy(value)
            else:
                top[key] = copy.deepcopy(value)
        else:
            mapping[key] = copy.deepcopy(value)

    if mapping:
        top["mapping"] = mapping
    return top


def _apply_defaults_list(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = cfg.pop("defaults", None)
    if defaults is None:
        return cfg
    if not isinstance(defaults, list):
        raise ValueError("defaults: must be a list")

    merged: dict[str, Any] = {}
    for entry in defaults:
        if isinstance(entry, str):
            piece = normalize_legacy_yaml(_load_yaml_file(entry))
            merged = _deep_merge(merged, piece)
            continue
        if isinstance(entry, dict):
            for section, ref in entry.items():
                section_raw = _load_yaml_file(str(ref))
                if section == "mapping" and "mapping" not in section_raw:
                    section_raw = normalize_legacy_yaml(section_raw)
                    piece = {section: section_raw.get("mapping", section_raw)}
                elif section in section_raw and len(section_raw) == 1:
                    piece = {section: section_raw[section]}
                else:
                    piece = normalize_legacy_yaml(section_raw)
                    if section in piece:
                        piece = {section: piece[section]}
                    elif section == "mapping":
                        piece = {section: piece.get("mapping", section_raw)}
                    else:
                        piece = {section: section_raw}
                merged = _deep_merge(merged, piece)
            continue
        raise ValueError(f"Invalid defaults entry: {entry!r}")

    return _deep_merge(merged, cfg)


def _apply_extends(cfg: dict[str, Any]) -> dict[str, Any]:
    extends = cfg.pop("extends", None)
    if extends is None:
        return cfg
    if isinstance(extends, list):
        base: dict[str, Any] = {}
        for ref in extends:
            loaded = load_config(str(ref), _skip_extends_defaults=True)
            base = _deep_merge(base, loaded.raw)
        return _deep_merge(base, cfg)
    loaded = load_config(str(extends), _skip_extends_defaults=True)
    return _deep_merge(loaded.raw, cfg)


def parse_dot_override(spec: str) -> tuple[str, Any]:
    """Parse ``key.path=value``; coerces bool/int/float/null."""
    if "=" not in spec:
        raise ValueError(f"Override must be KEY=VALUE, got {spec!r}")
    key, _, raw_value = spec.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"Empty override key in {spec!r}")
    value = _coerce_scalar(raw_value.strip())
    return key, value


def _coerce_scalar(raw: str) -> Any:
    low = raw.lower()
    if low in ("null", "none", "~"):
        return None
    if low in ("true", "false"):
        return low == "true"
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def apply_dot_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return cfg
    result = copy.deepcopy(cfg)
    for spec in overrides:
        key, value = parse_dot_override(spec)
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def merge_robot_overlay(cfg: dict[str, Any], robot_id: str) -> dict[str, Any]:
    """Deep-merge ``robots.<robot_id>`` into live sections (mapping, zmq, agent, …)."""
    robots = cfg.get("robots")
    if not isinstance(robots, dict):
        return cfg
    key = robot_id.lower().replace("-", "_")
    overlay = robots.get(key)
    if not isinstance(overlay, dict):
        return cfg

    result = copy.deepcopy(cfg)
    for section, value in overlay.items():
        if section == "robots":
            continue
        if section in result and isinstance(result[section], dict) and isinstance(value, dict):
            result[section] = _deep_merge(result[section], value)
        else:
            result[section] = copy.deepcopy(value)
    return result


@dataclass
class AgentSectionConfig:
    """Chat / tool agent options (``emet run agent``)."""

    llm: str = "qwen3-vl-eqa"
    eqa: bool = False
    share_memory_vllm: bool = True
    discord: bool = True
    prompt: str = "simple"
    device: str = "cuda"
    max_tokens: int = 1024


@dataclass
class ZmqSectionConfig:
    allow_missing_depth: bool | None = None


@dataclass
class ResolvedEmetConfig:
    """Fully merged nested config."""

    raw: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @property
    def robot(self) -> str | None:
        r = self.raw.get("robot")
        if r is None:
            return None
        s = str(r).strip()
        return s or None

    @property
    def connection(self) -> str | None:
        c = self.raw.get("connection")
        if c is None:
            return None
        s = str(c).strip()
        return s or None

    @property
    def mapping_dict(self) -> dict[str, Any]:
        """Flat mapping parameters for legacy :class:`~emet.core.parameters.Parameters`."""
        mapping = self.raw.get("mapping")
        if isinstance(mapping, dict):
            return copy.deepcopy(mapping)
        return {}

    @property
    def zmq(self) -> ZmqSectionConfig:
        z = self.raw.get("zmq")
        if isinstance(z, dict):
            return ZmqSectionConfig(
                allow_missing_depth=z.get("allow_missing_depth"),
            )
        return ZmqSectionConfig()

    def agent_section(self) -> AgentSectionConfig:
        a = self.raw.get("agent")
        if isinstance(a, dict) and _is_chat_agent_section(a):
            return draccus.decode(AgentSectionConfig, a)
        return AgentSectionConfig()

    def embodied_agent(self) -> EmbodiedAgentConfig:
        e = self.raw.get("embodied_agent")
        if isinstance(e, dict):
            return draccus.decode(EmbodiedAgentConfig, e)
        return EmbodiedAgentConfig()

    def rerun(self) -> RerunAgentConfig:
        r = self.raw.get("rerun")
        if isinstance(r, dict):
            return draccus.decode(RerunAgentConfig, r)
        mapping_rerun = self.mapping_dict.get("rerun")
        if isinstance(mapping_rerun, dict):
            return draccus.decode(RerunAgentConfig, mapping_rerun)
        return RerunAgentConfig()

    def sim_launch(self) -> SimLaunchConfig | None:
        inline = self.raw.get("sim")
        if isinstance(inline, dict):
            return decode_sim_launch_config(inline)
        path_key = self.raw.get("sim_config")
        if path_key is None or path_key is False:
            return None
        if not isinstance(path_key, str) or not str(path_key).strip():
            return None
        return load_sim_launch_config_from_path(str(path_key).strip())

    def with_robot_overlay(self, robot_id: str) -> ResolvedEmetConfig:
        merged = merge_robot_overlay(self.raw, robot_id)
        return ResolvedEmetConfig(raw=merged, source_path=self.source_path)

    def with_mapping_updates(self, updates: dict[str, Any]) -> ResolvedEmetConfig:
        raw = copy.deepcopy(self.raw)
        mapping = raw.get("mapping")
        if not isinstance(mapping, dict):
            mapping = {}
        raw["mapping"] = _deep_merge(mapping, updates)
        return ResolvedEmetConfig(raw=raw, source_path=self.source_path)


def load_config(
    path: str | None = None,
    *,
    overrides: list[str] | None = None,
    robot: str | None = None,
    _skip_extends_defaults: bool = False,
) -> ResolvedEmetConfig:
    """Load nested emet config from *path* with optional dot-path overrides."""
    config_path = path or default_config_path()
    try:
        full_path = _resolve_config_reference(config_path)
    except FileNotFoundError:
        if config_path == _LEGACY_DYNAV_CONFIG or config_path.endswith("dynav_config.yaml"):
            full_path = str(_emet_config_dir() / _LEGACY_DYNAV_CONFIG)
        else:
            raise

    raw = _load_yaml_file(full_path)
    if not _skip_extends_defaults:
        raw = _apply_extends(raw)
        raw = _apply_defaults_list(raw)
    raw = normalize_legacy_yaml(raw)
    raw = apply_dot_overrides(raw, overrides)

    resolved = ResolvedEmetConfig(raw=raw, source_path=full_path)
    if robot:
        resolved = resolved.with_robot_overlay(robot)
    return resolved


def resolve_config_path_for_legacy_alias(path: str) -> str:
    """Map legacy basenames (``dynav_config.yaml``) to unified config when available."""
    if path in (_LEGACY_DYNAV_CONFIG, "dynav_innate_mars.yaml"):
        default = default_config_path()
        if default != path:
            try:
                _resolve_config_reference(default)
                return default
            except FileNotFoundError:
                pass
    return path
