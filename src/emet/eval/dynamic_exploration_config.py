# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Load ``configs/benchmarks/dynamic_exploration.yaml`` and build run matrices."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from emet.eval.benchmark_dynagraph import DYNAMIC_EXPLORE_BACKENDS
from emet.utils.config import resolve_config_yaml_path

DEFAULT_DYNAMIC_EXPLORE_YAML = "configs/benchmarks/dynamic_exploration.yaml"

MappingMode = Literal["explore", "rotate_only"]
ExplorePhase = Literal["explore", "world-change"]


@dataclass(frozen=True)
class DynamicExploreEpisode:
    id: str
    env: Literal["robocasa", "molmospaces"]
    sim: str
    question_env: str
    seed: int | None = None
    molmo_index: int | None = None


@dataclass(frozen=True)
class WorldChangeEpisode:
    id: str
    episode_id: str
    question_env: str
    relocate_body: str


@dataclass(frozen=True)
class LifelongEpisode:
    """K-cycle lifelong episode: explore/answer → checkpoint → fuzz → reload → repeat.

    ``questions`` holds one question list per cycle (short lists repeat the last entry).
    ``changes`` holds the fuzz spec applied *between* cycle t and t+1 (``moves``/``doors``/
    ``random`` blocks, see :mod:`emet.eval.world_fuzz`); missing entries mean no change.
    """

    id: str
    episode_id: str
    cycles: int
    questions: tuple[tuple[dict[str, Any], ...], ...]
    changes: tuple[dict[str, Any], ...]
    explore_iters_first: int = 12
    explore_iters_resume: int = 4

    def questions_for_cycle(self, t: int) -> list[dict[str, Any]]:
        if not self.questions:
            return []
        idx = min(int(t), len(self.questions) - 1)
        return [dict(q) for q in self.questions[idx]]

    def changes_after_cycle(self, t: int) -> dict[str, Any] | None:
        if t < 0 or t >= len(self.changes):
            return None
        spec = self.changes[t]
        return dict(spec) if spec else None


@dataclass(frozen=True)
class DynamicExplorePaths:
    output_dir: Path
    questions_yaml: Path


@dataclass(frozen=True)
class DynamicExploreConfig:
    paths: DynamicExplorePaths
    episodes: tuple[DynamicExploreEpisode, ...]
    world_change_episodes: tuple[WorldChangeEpisode, ...]
    lifelong_episodes: tuple[LifelongEpisode, ...]
    explore_budgets: tuple[int, ...]
    recovery_explore_iters: int
    profiles: dict[str, str]
    smoke: dict[str, Any]


@dataclass(frozen=True)
class ExploreRunSpec:
    episode: DynamicExploreEpisode
    backend: str
    mapping_mode: MappingMode
    explore_max_iters: int
    phase: ExplorePhase = "explore"

    @property
    def run_id(self) -> str:
        if self.phase == "world-change":
            return f"{self.episode.id}_world_change_{self.backend}"
        suffix = f"{self.mapping_mode}_{self.explore_max_iters}"
        return f"{self.episode.id}_{self.backend}_{suffix}"


def _normalize_lifelong_questions(raw: Any) -> tuple[tuple[dict[str, Any], ...], ...]:
    """Accept one question list (reused every cycle) or a list of per-cycle lists."""
    if not raw:
        return ()
    if isinstance(raw, list) and raw and all(isinstance(x, dict) for x in raw):
        return (tuple(dict(q) for q in raw),)
    out: list[tuple[dict[str, Any], ...]] = []
    for cycle_qs in raw:
        if isinstance(cycle_qs, list):
            out.append(tuple(dict(q) for q in cycle_qs if isinstance(q, dict)))
    return tuple(out)


def _path_from_env_or_yaml(env_key: str, yaml_value: str) -> Path:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(yaml_value).expanduser().resolve()


def load_dynamic_exploration_config(path: str | Path | None = None) -> DynamicExploreConfig:
    full = Path(resolve_config_yaml_path(str(path or DEFAULT_DYNAMIC_EXPLORE_YAML)))
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    paths_raw = raw.get("paths") if isinstance(raw, dict) else {}
    if not isinstance(paths_raw, dict):
        paths_raw = {}

    questions_raw = raw.get("questions") if isinstance(raw, dict) else {}
    if not isinstance(questions_raw, dict):
        questions_raw = {}

    wc_raw = raw.get("world_change") if isinstance(raw, dict) else {}
    if not isinstance(wc_raw, dict):
        wc_raw = {}

    profiles_raw = raw.get("profiles") if isinstance(raw, dict) else {}
    if not isinstance(profiles_raw, dict):
        profiles_raw = {"dynagraph": "interactive", "graph_eqa": "graph_eqa_baseline"}

    smoke_raw = raw.get("smoke") if isinstance(raw, dict) else {}
    if not isinstance(smoke_raw, dict):
        smoke_raw = {}

    episodes: list[DynamicExploreEpisode] = []
    for row in raw.get("episodes") or []:
        if not isinstance(row, dict):
            continue
        episodes.append(
            DynamicExploreEpisode(
                id=str(row["id"]),
                env=str(row["env"]),  # type: ignore[arg-type]
                sim=str(row["sim"]),
                question_env=str(row["question_env"]),
                seed=(int(row["seed"]) if row.get("seed") is not None else None),
                molmo_index=(int(row["molmo_index"]) if row.get("molmo_index") is not None else None),
            )
        )

    wc_eps: list[WorldChangeEpisode] = []
    for row in raw.get("world_change_episodes") or []:
        if not isinstance(row, dict):
            continue
        wc_eps.append(
            WorldChangeEpisode(
                id=str(row["id"]),
                episode_id=str(row["episode_id"]),
                question_env=str(row["question_env"]),
                relocate_body=str(row.get("relocate_body") or wc_raw.get("relocate_body") or "obj_main"),
            )
        )

    lifelong_raw = raw.get("lifelong") if isinstance(raw, dict) else {}
    if not isinstance(lifelong_raw, dict):
        lifelong_raw = {}
    ll_first_default = int(lifelong_raw.get("explore_iters_first", 12))
    ll_resume_default = int(lifelong_raw.get("explore_iters_resume", 4))
    ll_eps: list[LifelongEpisode] = []
    for row in lifelong_raw.get("episodes") or []:
        if not isinstance(row, dict):
            continue
        ll_eps.append(
            LifelongEpisode(
                id=str(row["id"]),
                episode_id=str(row["episode_id"]),
                cycles=int(row.get("cycles", 3)),
                questions=_normalize_lifelong_questions(row.get("questions")),
                changes=tuple(dict(c) for c in (row.get("changes") or []) if isinstance(c, dict)),
                explore_iters_first=int(row.get("explore_iters_first", ll_first_default)),
                explore_iters_resume=int(row.get("explore_iters_resume", ll_resume_default)),
            )
        )

    budgets_raw = raw.get("explore_budgets") or [8, 15, 30]
    budgets = tuple(int(x) for x in budgets_raw)

    paths = DynamicExplorePaths(
        output_dir=_path_from_env_or_yaml(
            "EMET_DYNAMIC_EXPLORE_OUTPUT",
            str(paths_raw.get("output_dir", "~/runs/emet/dynamic_exploration")),
        ),
        questions_yaml=Path(
            resolve_config_yaml_path(
                str(questions_raw.get("yaml", "src/emet/config/benchmarks/dynagraph_questions.yaml"))
            )
        ),
    )

    return DynamicExploreConfig(
        paths=paths,
        episodes=tuple(episodes),
        world_change_episodes=tuple(wc_eps),
        lifelong_episodes=tuple(ll_eps),
        explore_budgets=budgets,
        recovery_explore_iters=int(wc_raw.get("recovery_explore_iters", 4)),
        profiles={str(k): str(v) for k, v in profiles_raw.items()},
        smoke=dict(smoke_raw),
    )


@dataclass(frozen=True)
class SmokeRunPlan:
    """CLI overrides from ``smoke:`` in ``dynamic_exploration.yaml``."""

    episode_id: str
    backend: str
    explore_max_iters: int
    mapping_mode: MappingMode
    phase: ExplorePhase = "explore"


def resolve_smoke_run_plan(cfg: DynamicExploreConfig) -> SmokeRunPlan:
    """Return smoke-run settings from benchmark YAML (``smoke:`` block)."""
    raw = cfg.smoke or {}
    mapping = str(raw.get("mapping_mode", "explore"))
    if mapping not in ("explore", "rotate_only"):
        raise ValueError(f"Invalid smoke.mapping_mode {mapping!r}")
    phase = str(raw.get("phase", "explore"))
    if phase not in ("explore", "world-change", "lifelong"):
        raise ValueError(f"Invalid smoke.phase {phase!r}")
    return SmokeRunPlan(
        episode_id=str(raw.get("episode_id", "robocasa_seed0")),
        backend=str(raw.get("backend", "dynagraph")),
        explore_max_iters=int(raw.get("explore_max_iters", 3)),
        mapping_mode=mapping,  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
    )


def filter_episodes(
    episodes: tuple[DynamicExploreEpisode, ...] | list[DynamicExploreEpisode],
    *,
    env: str | None = None,
    episode_ids: list[str] | None = None,
    seed: int | None = None,
) -> list[DynamicExploreEpisode]:
    out = list(episodes)
    if env:
        out = [e for e in out if e.env == env]
    if episode_ids:
        id_set = {i.strip() for i in episode_ids}
        out = [e for e in out if e.id in id_set]
    if seed is not None:
        out = [e for e in out if e.seed == seed or e.molmo_index == seed]
    return out


def build_explore_run_matrix(
    cfg: DynamicExploreConfig,
    episodes: list[DynamicExploreEpisode],
    *,
    backends: list[str],
    explore_max_iters: list[int] | None = None,
    mapping_modes: list[MappingMode] | None = None,
    include_rotate_only: bool = True,
) -> list[ExploreRunSpec]:
    budgets = explore_max_iters or list(cfg.explore_budgets)
    modes: list[MappingMode] = list(mapping_modes or ["explore"])
    if include_rotate_only and "rotate_only" not in modes:
        modes.append("rotate_only")

    runs: list[ExploreRunSpec] = []
    for ep in episodes:
        for backend in backends:
            if backend not in DYNAMIC_EXPLORE_BACKENDS:
                raise ValueError(f"Unknown backend {backend!r}; expected one of {DYNAMIC_EXPLORE_BACKENDS}")
            for mode in modes:
                if mode == "rotate_only":
                    runs.append(
                        ExploreRunSpec(
                            episode=ep,
                            backend=backend,
                            mapping_mode="rotate_only",
                            explore_max_iters=0,
                        )
                    )
                else:
                    for k in budgets:
                        runs.append(
                            ExploreRunSpec(
                                episode=ep,
                                backend=backend,
                                mapping_mode="explore",
                                explore_max_iters=int(k),
                            )
                        )
    return runs


def _fusion_recall_fields(fusion: dict[str, Any]) -> tuple[Any, Any]:
    """Prefer fused-graph recall, then raw detections, then legacy top-level keys.

    ``compute_dynagraph_eval`` nests recalls under ``fusion.fused`` / ``fusion.raw``;
    older callers may still place them on ``fusion`` itself.
    """
    fused = fusion.get("fused") if isinstance(fusion.get("fused"), dict) else {}
    raw = fusion.get("raw") if isinstance(fusion.get("raw"), dict) else {}
    spatial = fused.get("spatial_recall")
    if spatial is None:
        spatial = raw.get("spatial_recall")
    if spatial is None:
        spatial = fusion.get("spatial_recall")
    label = fused.get("label_recall")
    if label is None:
        label = raw.get("label_recall")
    if label is None:
        label = fusion.get("label_recall")
    return spatial, label


def flatten_eval_metrics(
    metrics: dict[str, Any],
    *,
    run_spec: ExploreRunSpec | None = None,
    episode_wall_s: float | None = None,
) -> dict[str, Any]:
    explore = metrics.get("explore") or {}
    graph = metrics.get("graph") or {}
    fusion = metrics.get("fusion") or {}
    eqa = metrics.get("eqa") or {}
    spatial_recall, label_recall = _fusion_recall_fields(fusion if isinstance(fusion, dict) else {})
    health = metrics.get("graph_health") or {}
    row: dict[str, Any] = {
        "explored_fraction": explore.get("explored_fraction"),
        "explored_area_m2": explore.get("explored_area_m2"),
        "spatial_recall": spatial_recall,
        "label_recall": label_recall,
        "node_count": graph.get("node_count"),
        "edge_count": graph.get("edge_count"),
        "eqa_accuracy": eqa.get("accuracy"),
        "n_frames": metrics.get("n_frames"),
        "graph_health_n_object": health.get("n_object"),
        "graph_health_failure_class": health.get("failure_class"),
        "graph_health_singleton_frac": health.get("singleton_frac"),
    }
    if run_spec is not None:
        row.update(
            {
                "run_id": run_spec.run_id,
                "episode_id": run_spec.episode.id,
                "env": run_spec.episode.env,
                "backend": run_spec.backend,
                "mapping_mode": run_spec.mapping_mode,
                "explore_max_iters": run_spec.explore_max_iters,
                "phase": run_spec.phase,
                "question_env": run_spec.episode.question_env,
            }
        )
    if episode_wall_s is not None:
        row["episode_wall_s"] = float(episode_wall_s)
    if metrics.get("error"):
        row["error"] = metrics["error"]
    return row
