# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""OVMM multi-env sweep prep, rates aggregation, and status helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from emet.eval.ovmm_find_phase import mapping_budget_from_row
from emet.utils.config import resolve_config_yaml_path

SWEEPS_DIR = "configs/ovmm/sweeps"
DEFAULT_PRESET = "molmo-robocasa"

FIND_RATE_KEYS = (
    "episode_id",
    "find_object_success",
    "find_recep_success",
    "find_partial_success",
    "localization_err_obj_m",
    "localization_err_recep_m",
    "error",
    "sim",
)
FULL_RATE_KEYS = (
    "episode_id",
    "ovmm_full_success",
    "ovmm_full_partial",
    "find_object_success",
    "pick_success",
    "find_recep_success",
    "place_success",
    "error",
    "sim",
)


@dataclass(frozen=True)
class PreparedSweep:
    """Paths written by :func:`prepare_multi_env_sweep`."""

    out_dir: Path
    find_episodes: Path
    full_episodes: Path
    sim_dir: Path
    preset_name: str


def resolve_ovmm_sweep_preset_path(name_or_path: str | Path) -> Path:
    """Resolve ``molmo-robocasa`` or a YAML path to an absolute preset file."""
    raw = str(name_or_path).strip()
    if not raw:
        raw = DEFAULT_PRESET
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    # Allow hyphenated CLI names: molmo-robocasa → molmo_robocasa.yaml
    stem = raw.removesuffix(".yaml").removesuffix(".yml").replace("-", "_")
    root = Path(__file__).resolve().parents[3]
    local = root / SWEEPS_DIR / f"{stem}.yaml"
    if local.is_file():
        return local.resolve()
    try:
        return Path(resolve_config_yaml_path(f"{SWEEPS_DIR}/{stem}.yaml")).resolve()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"OVMM sweep preset not found: {name_or_path} (tried {local})") from exc


def load_ovmm_sweep_preset(name_or_path: str | Path = DEFAULT_PRESET) -> dict[str, Any]:
    """Load a checked-in or custom OVMM sweep preset YAML."""
    path = resolve_ovmm_sweep_preset_path(name_or_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Sweep preset must be a mapping: {path}")
    data["_preset_path"] = str(path)
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False), encoding="utf-8")


def _robocasa_sim_yaml(
    *,
    layout: int,
    style: int,
    robot: str,
    task: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "kind": "robocasa",
        "robot": robot,
        "robocasa_task": task,
        "robocasa_style": int(style),
        "robocasa_layout": int(layout),
        "headless": True,
        "port_offset": 0,
        "seed": int(seed),
    }


def _molmo_sim_yaml(
    *,
    index: int,
    robot: str,
    scene: str = "ithor",
    split: str = "train",
    seed: int = 0,
) -> dict[str, Any]:
    return {
        "kind": "molmospaces",
        "robot": robot,
        "scene": scene,
        "split": split,
        "index": int(index),
        "molmospaces_install": False,
        "headless": True,
        "port_offset": 0,
        "seed": int(seed),
    }


def _episode_dict(
    *,
    ep_id: str,
    tier: str,
    sim_path: Path,
    object_name: str,
    start_recep: str,
    goal_recep: str,
    success_radius_m: float,
    mapping_max_nav_steps: int,
    object_gt_body: str | None = None,
) -> dict[str, Any]:
    ep: dict[str, Any] = {
        "id": ep_id,
        "tier": tier,
        "sim": str(sim_path.resolve()),
        "object": object_name,
        "start_recep": start_recep,
        "goal_recep": goal_recep,
        "success_radius_m": float(success_radius_m),
        "mapping_max_nav_steps": int(mapping_max_nav_steps),
    }
    if object_gt_body:
        ep["object_gt_body"] = object_gt_body
    if "default_table" in str(sim_path):
        raise ValueError(f"default_table sims are not allowed in multi-env sweeps: {sim_path}")
    return ep


def _expand_robocasa_episodes(
    block: dict[str, Any] | None,
    *,
    sim_dir: Path,
    suffix: str,
) -> list[dict[str, Any]]:
    if not block:
        return []
    layouts = [int(x) for x in (block.get("layouts") or [])]
    style_eq = bool(block.get("style_equals_layout", True))
    robot = str(block.get("robot", "stretch"))
    task = str(block.get("task", "PickPlaceCounterToCabinet"))
    seed = int(block.get("seed", 0))
    object_name = str(block.get("object", "jar"))
    object_gt_body = block.get("object_gt_body")
    start_recep = str(block.get("start_recep", "counter"))
    goal_recep = str(block.get("goal_recep", "cab"))
    radius = float(block.get("success_radius_m", 0.5))
    explore = mapping_budget_from_row(block, source=f"sweep robocasa {suffix}", default=8)
    tier = str(block.get("tier", "S1"))
    eps: list[dict[str, Any]] = []
    for layout in layouts:
        style = layout if style_eq else int(block.get("style", layout))
        sim_name = f"robocasa_l{layout}_s{style}_seed{seed}"
        sim_path = sim_dir / f"{sim_name}.yaml"
        _write_yaml(
            sim_path,
            _robocasa_sim_yaml(layout=layout, style=style, robot=robot, task=task, seed=seed),
        )
        eps.append(
            _episode_dict(
                ep_id=f"{sim_name}_{suffix}",
                tier=tier,
                sim_path=sim_path,
                object_name=object_name,
                start_recep=start_recep,
                goal_recep=goal_recep,
                success_radius_m=radius,
                mapping_max_nav_steps=explore,
                object_gt_body=str(object_gt_body) if object_gt_body else None,
            )
        )
    return eps


def _expand_molmo_episodes(
    rows: list[dict[str, Any]] | None,
    *,
    sim_dir: Path,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    eps: list[dict[str, Any]] = []
    for row in rows:
        ep_id = str(row["id"])
        index = int(row["index"])
        robot = str(row.get("robot", "stretch"))
        seed = int(row.get("seed", 0))
        scene = str(row.get("scene", "ithor"))
        split = str(row.get("split", "train"))
        # Optional ``sim`` stem; else episode id, or ``molmo_<scene>_idxN_rby1`` for rby1.
        if row.get("sim"):
            sim_name = str(row["sim"])
        elif robot == "rby1":
            sim_name = f"molmo_{scene}_idx{index}_rby1"
        else:
            sim_name = ep_id
        sim_path = sim_dir / f"{sim_name}.yaml"
        _write_yaml(
            sim_path,
            _molmo_sim_yaml(index=index, robot=robot, scene=scene, split=split, seed=seed),
        )
        eps.append(
            _episode_dict(
                ep_id=ep_id,
                tier=str(row.get("tier", "S2")),
                sim_path=sim_path,
                object_name=str(row.get("object", "bowl")),
                start_recep=str(row.get("start_recep", "cabinet")),
                goal_recep=str(row.get("goal_recep", "microwave")),
                success_radius_m=float(row.get("success_radius_m", 0.75)),
                mapping_max_nav_steps=mapping_budget_from_row(row, source=f"sweep molmo {ep_id}", default=15),
                object_gt_body=str(row["object_gt_body"]) if row.get("object_gt_body") else None,
            )
        )
    return eps


def prepare_multi_env_sweep(
    out_dir: str | Path,
    preset: str | Path | dict[str, Any] = DEFAULT_PRESET,
    *,
    sync_robocasa_registry: bool | None = None,
) -> PreparedSweep:
    """Write ``sim/``, ``find_episodes.yaml``, ``full_episodes.yaml`` under ``out_dir``."""
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    sim_dir = out / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)

    data = preset if isinstance(preset, dict) else load_ovmm_sweep_preset(preset)
    preset_name = str(data.get("name") or "custom")
    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}

    do_sync = sync_robocasa_registry
    if do_sync is None:
        do_sync = bool(defaults.get("sync_robocasa_registry", False))
    has_robocasa = bool((data.get("find") or {}).get("robocasa") or (data.get("full") or {}).get("robocasa"))
    if do_sync and has_robocasa:
        from emet.simulation.robocasa_registry_sync import sync_lightwheel_registry

        n = sync_lightwheel_registry()
        print(f"sync_lightwheel_registry: added/upgraded {n} entries")

    find_block = data.get("find") if isinstance(data.get("find"), dict) else {}
    full_block = data.get("full") if isinstance(data.get("full"), dict) else {}

    find_eps = _expand_robocasa_episodes(find_block.get("robocasa"), sim_dir=sim_dir, suffix="find")
    find_eps += _expand_molmo_episodes(find_block.get("molmo"), sim_dir=sim_dir)
    full_eps = _expand_robocasa_episodes(full_block.get("robocasa"), sim_dir=sim_dir, suffix="full")
    full_eps += _expand_molmo_episodes(full_block.get("molmo"), sim_dir=sim_dir)

    for ep in find_eps + full_eps:
        if "default_table" in str(ep.get("sim") or ""):
            raise ValueError(f"default_table not allowed: {ep}")

    find_path = out / "find_episodes.yaml"
    full_path = out / "full_episodes.yaml"
    _write_yaml(find_path, {"episodes": find_eps})
    _write_yaml(full_path, {"episodes": full_eps})
    meta = {
        "preset": preset_name,
        "preset_path": data.get("_preset_path"),
        "n_find": len(find_eps),
        "n_full": len(full_eps),
        "defaults": defaults,
    }
    (out / "sweep_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return PreparedSweep(
        out_dir=out,
        find_episodes=find_path,
        full_episodes=full_path,
        sim_dir=sim_dir,
        preset_name=preset_name,
    )


def _is_bind_or_init_fail(err: str) -> bool:
    e = err.lower()
    return (
        "did not bind port" in e
        or "could not initialize task" in e
        or "ran _load_model()" in e
        or "missing or incomplete" in e
    )


def load_result_rows(phase_dir: Path, *, backend: str = "dynagraph") -> list[dict[str, Any]]:
    """Load ``*_backend.json`` metrics from a find/ or full/ directory."""
    if not phase_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for p in sorted(phase_dir.glob(f"*_{backend}.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict):
            rows.append(d)
    return rows


def aggregate_ovmm_rates(
    out_dir: str | Path,
    *,
    backend: str = "dynagraph",
    exclude_bind_fails: bool = True,
) -> dict[str, Any]:
    """Aggregate find/full JSON into ``rates.json``; return the rates dict."""
    out = Path(out_dir).expanduser().resolve()
    find_rows_all = load_result_rows(out / "find", backend=backend)
    full_rows_all = load_result_rows(out / "full", backend=backend)

    bind_fail = 0
    find_rows: list[dict[str, Any]] = []
    for r in find_rows_all:
        err = str(r.get("error") or "")
        if exclude_bind_fails and _is_bind_or_init_fail(err):
            bind_fail += 1
            continue
        find_rows.append(r)

    full_bind = 0
    full_rows: list[dict[str, Any]] = []
    for r in full_rows_all:
        err = str(r.get("error") or "")
        if exclude_bind_fails and _is_bind_or_init_fail(err):
            full_bind += 1
            continue
        full_rows.append(r)

    n = len(find_rows)
    # Prefer scored-phase denominators when present (unscored ≠ localization miss).
    obj_scored = [r for r in find_rows if r.get("find_object_scored", True)]
    recep_scored = [r for r in find_rows if r.get("find_recep_scored", True)]
    n_find_obj = sum(1 for r in obj_scored if r.get("find_object_success"))
    n_find_recep = sum(1 for r in recep_scored if r.get("find_recep_success"))
    n_obj = len(obj_scored)
    n_recep = len(recep_scored)
    mp = sum(float(r.get("find_partial_success") or 0) for r in find_rows) / max(n, 1)

    nf = len(full_rows)
    fs = sum(1 for r in full_rows if r.get("ovmm_full_success"))
    fmp = sum(float(r.get("ovmm_full_partial") or 0) for r in full_rows) / max(nf, 1)

    rates = {
        "backend": backend,
        "find": {
            "n": n,
            "n_object_scored": n_obj,
            "n_recep_scored": n_recep,
            "find_object": n_find_obj,
            "find_recep": n_find_recep,
            "mean_partial": mp,
            "skipped_bind_or_init": bind_fail,
            "rows": [{k: r.get(k) for k in FIND_RATE_KEYS} for r in find_rows],
        },
        "full": {
            "n": nf,
            "full_success": fs,
            "mean_partial": fmp,
            "skipped_bind_or_init": full_bind,
            "rows": [{k: r.get(k) for k in FULL_RATE_KEYS} for r in full_rows],
        },
    }
    rates_path = out / "rates.json"
    rates_path.write_text(json.dumps(rates, indent=2) + "\n", encoding="utf-8")
    print(
        f"FIND (excl bind/init fails): n={n} "
        f"FindObj={n_find_obj}/{n_obj} ({100 * n_find_obj / max(n_obj, 1):.0f}%) "
        f"FindRec={n_find_recep}/{n_recep} ({100 * n_find_recep / max(n_recep, 1):.0f}%) "
        f"mean_partial={mp:.3f} skipped={bind_fail}"
    )
    print(
        f"FULL: n={nf} full_success={fs}/{nf} ({100 * fs / max(nf, 1):.0f}%) mean_partial={fmp:.3f} skipped={full_bind}"
    )
    print(f"Wrote {rates_path}")
    return rates


def ovmm_sweep_status(out_dir: str | Path, *, backend: str = "dynagraph") -> dict[str, Any]:
    """Summarize per-episode outcomes under OUT (for ``emet ovmm status``)."""
    out = Path(out_dir).expanduser().resolve()
    find_rows = load_result_rows(out / "find", backend=backend)
    full_rows = load_result_rows(out / "full", backend=backend)

    def _classify(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary = []
        for r in rows:
            err = str(r.get("error") or "")
            kind = "ok"
            if _is_bind_or_init_fail(err):
                kind = "bind_or_init_fail"
            elif err:
                kind = "error"
            summary.append(
                {
                    "episode_id": r.get("episode_id"),
                    "kind": kind,
                    "find_partial": r.get("find_partial_success"),
                    "ovmm_full_partial": r.get("ovmm_full_partial"),
                    "error": err[:160] if err else "",
                }
            )
        return summary

    return {
        "out_dir": str(out),
        "rates_exists": (out / "rates.json").is_file(),
        "find_episodes_exists": (out / "find_episodes.yaml").is_file(),
        "full_episodes_exists": (out / "full_episodes.yaml").is_file(),
        "find": _classify(find_rows),
        "full": _classify(full_rows),
        "n_find": len(find_rows),
        "n_full": len(full_rows),
        "n_find_bind_fail": sum(1 for r in find_rows if _is_bind_or_init_fail(str(r.get("error") or ""))),
        "n_full_bind_fail": sum(1 for r in full_rows if _is_bind_or_init_fail(str(r.get("error") or ""))),
    }


def collect_failed_episode_ids(out_dir: str | Path, *, backend: str = "dynagraph", phase: str = "find") -> list[str]:
    """Episode ids whose result JSON is a bind/task-init failure (for ``--rerun-failed``)."""
    out = Path(out_dir).expanduser().resolve()
    rows = load_result_rows(out / phase, backend=backend)
    ids: list[str] = []
    for r in rows:
        if _is_bind_or_init_fail(str(r.get("error") or "")):
            eid = r.get("episode_id")
            if eid:
                ids.append(str(eid))
    return ids


def write_rerun_episodes_yaml(
    out_dir: str | Path,
    *,
    phase: str = "find",
    backend: str = "dynagraph",
    dest_name: str | None = None,
) -> Path | None:
    """Filter find/full episode YAML to failed ids; write ``find_episodes_rerun.yaml``."""
    out = Path(out_dir).expanduser().resolve()
    src = out / ("find_episodes.yaml" if phase == "find" else "full_episodes.yaml")
    if not src.is_file():
        return None
    failed = set(collect_failed_episode_ids(out, backend=backend, phase=phase))
    if not failed:
        return None
    data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    eps = [e for e in (data.get("episodes") or []) if isinstance(e, dict) and e.get("id") in failed]
    dest = out / (dest_name or f"{phase}_episodes_rerun.yaml")
    _write_yaml(dest, {"episodes": eps})
    return dest
