# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Factory for ``emet stream`` / ``emet capture`` mapping agents (DynaMem, GraphEQA, Dynagraph, SVM, …).

CLI profiles: :mod:`emet.app.zmq_obs`. User guide: ``docs/zmq_obs.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import click
import numpy as np

from emet.app.robot_cli import create_robot_client_from_cli
from emet.config.loader import finalize_resolved_config, load_config, resolve_config_path_for_legacy_alias
from emet.config.runtime import build_parameters_from_config
from emet.robots import DEFAULT_DYNAV_CONFIG_YAML

StreamBackend = Literal[
    "dynamem",
    "static_graph",
    "graph_eqa",
    "dynagraph",
    "ground_truth",
    "svm",
    "scene_graph",
    "voxel_only",
]

STREAM_BACKENDS: tuple[str, ...] = (
    "dynamem",
    "static_graph",
    "graph_eqa",
    "dynagraph",
    "ground_truth",
    "svm",
    "scene_graph",
    "voxel_only",
)

_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_localhost_host(host: str) -> bool:
    return host.strip().lower() in _LOCALHOST_HOSTS


@dataclass
class StreamAgentBundle:
    agent: Any
    dynav_resolved: str
    backend: str


def _robot_key(robot: str) -> str:
    return robot.lower().replace("-", "_")


def resolve_stream_config_path(dynav_config: str, *, dynav_from_default: bool) -> str:
    """Resolve config path for stream/capture (unified default when legacy dynav basename omitted)."""
    if not dynav_from_default:
        return dynav_config
    if dynav_config in (DEFAULT_DYNAV_CONFIG_YAML, "dynav_innate_mars.yaml"):
        return resolve_config_path_for_legacy_alias(dynav_config)
    return dynav_config


def resolve_stream_dynav_config(
    robot: str,
    host: str,
    dynav_config: str,
    *,
    dynav_from_default: bool,
) -> str:
    """Backward-compatible alias: returns resolved config *path* (robot overlays applied at load)."""
    del robot, host
    return resolve_stream_config_path(dynav_config, dynav_from_default=dynav_from_default)


def load_stream_parameters(
    robot: str,
    host: str,
    config_path: str,
    *,
    overrides: list[str] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Load mapping parameters with robot overlay and runtime depth rules."""
    cfg = load_config(config_path)
    cfg = finalize_resolved_config(cfg, robot_id=_robot_key(robot), overrides=overrides)
    parameters, allow_missing = build_parameters_from_config(cfg, _robot_key(robot), host=host)
    return parameters.data, allow_missing


def _resolve_allow_missing_depth(
    robot: str,
    parameters: dict[str, Any],
    *,
    allow_missing_depth: bool | None,
    graph_style: bool = False,
) -> bool:
    robot_key = _robot_key(robot)
    depth_mode = str(parameters.get("depth_source", "sensor")).lower()
    if allow_missing_depth is not None:
        return allow_missing_depth
    if graph_style:
        return depth_mode in ("da3", "auto") or robot_key in (
            "innate_mars",
            "galaxea_r1",
            "rby1",
            "stretch",
        )
    return depth_mode in ("da3", "auto", "lingbot") or robot_key == "innate_mars"


def _prepare_graph_style_parameters(
    robot: str,
    host: str,
    config_path: str,
    *,
    overrides: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    parameters, _allow = load_stream_parameters(robot, host, config_path, overrides=overrides)
    parameters.setdefault("dynagraph_merge_xy_m", 0.45)
    parameters.setdefault("dynagraph_staleness_horizon", 256)
    if parameters.get("graph_object_fusion") is None:
        from dataclasses import asdict

        from emet.memory.graph_eqa.graph_object_fusion.config import load_graph_object_fusion_config

        parameters["graph_object_fusion"] = asdict(load_graph_object_fusion_config())
    parameters["encoder"] = None
    return config_path, parameters


def _build_robot_client(
    *,
    robot: str,
    host: str,
    port_offset: int,
    parameters: dict[str, Any],
    enable_rerun: bool,
    headless: bool,
    rerun_native: bool,
    rerun_show_panels: bool,
    rerun_debug: bool,
    allow_missing_depth: bool,
):
    return create_robot_client_from_cli(
        robot,
        host,
        port_offset=port_offset,
        parameters=parameters,
        enable_rerun_server=enable_rerun,
        rerun_headless=headless,
        rerun_native_viewer=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        start_immediately=False,
        allow_missing_depth=allow_missing_depth,
    )


def create_dynamem_agent(
    *,
    robot: str,
    host: str,
    port_offset: int,
    dynav_config: str,
    enable_rerun: bool,
    headless: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    allow_missing_depth: bool | None = None,
    cpu_only: bool = False,
    voxel_only: bool = False,
    config_overrides: list[str] | None = None,
):
    """Build a DynamemController + ZMQ client (``DynamemController`` calls ``robot.start()``)."""
    from emet.controller.controller_dynamem import RobotAgent as DynamemController

    config_path = dynav_config
    parameters, allow_from_cfg = load_stream_parameters(robot, host, config_path, overrides=config_overrides)
    allow = allow_missing_depth if allow_missing_depth is not None else allow_from_cfg
    robot_client = _build_robot_client(
        robot=robot,
        host=host,
        port_offset=port_offset,
        parameters=parameters,
        enable_rerun=enable_rerun,
        headless=headless,
        rerun_native=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        allow_missing_depth=allow,
    )
    agent = DynamemController(
        robot_client,
        parameters,
        save_rerun=enable_rerun,
        cpu_only=cpu_only,
        manipulation_only=voxel_only,
        use_instance_memory=False,
        eqa=False,
        defer_eqa_vllm=True,
    )
    return agent, config_path


def create_dynagraph_agent(
    *,
    robot: str,
    host: str,
    port_offset: int,
    dynav_config: str,
    enable_rerun: bool,
    headless: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    allow_missing_depth: bool | None = None,
    cpu_only: bool = False,
    use_sensor_perception: bool = True,
    use_instance_graph: bool = True,
    ground_truth_mode: bool = False,
    visualize_ground_truth: bool = False,
    voxel_only: bool = False,
    config_overrides: list[str] | None = None,
):
    """Build a DynagraphController + ZMQ client."""
    from emet.controller.controller_dynagraph import DynagraphController

    config_path, parameters = _prepare_graph_style_parameters(robot, host, dynav_config, overrides=config_overrides)
    allow = _resolve_allow_missing_depth(robot, parameters, allow_missing_depth=allow_missing_depth, graph_style=True)
    if voxel_only:
        use_sensor_perception = False
        use_instance_graph = False
    robot_client = _build_robot_client(
        robot=robot,
        host=host,
        port_offset=port_offset,
        parameters=parameters,
        enable_rerun=enable_rerun,
        headless=headless,
        rerun_native=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        allow_missing_depth=allow,
    )
    agent = DynagraphController(
        robot_client,
        parameters,
        save_rerun=enable_rerun,
        use_sensor_perception=use_sensor_perception,
        cpu_only=cpu_only,
        use_instance_graph=use_instance_graph,
        ground_truth_mode=ground_truth_mode,
        visualize_ground_truth=visualize_ground_truth,
        manipulation_only=voxel_only,
    )
    return agent, config_path


def create_graph_eqa_agent(
    *,
    robot: str,
    host: str,
    port_offset: int,
    dynav_config: str,
    enable_rerun: bool,
    headless: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    allow_missing_depth: bool | None = None,
    cpu_only: bool = False,
    use_sensor_perception: bool = True,
    use_instance_graph: bool = True,
    voxel_only: bool = False,
    config_overrides: list[str] | None = None,
):
    """Build a GraphEQAController + ZMQ client."""
    from emet.controller.controller_graph_eqa import GraphEQAController

    config_path, parameters = _prepare_graph_style_parameters(robot, host, dynav_config, overrides=config_overrides)
    allow = _resolve_allow_missing_depth(robot, parameters, allow_missing_depth=allow_missing_depth, graph_style=True)
    if voxel_only:
        use_sensor_perception = False
        use_instance_graph = False
    robot_client = _build_robot_client(
        robot=robot,
        host=host,
        port_offset=port_offset,
        parameters=parameters,
        enable_rerun=enable_rerun,
        headless=headless,
        rerun_native=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        allow_missing_depth=allow,
    )
    agent = GraphEQAController(
        robot_client,
        parameters,
        save_rerun=enable_rerun,
        use_sensor_perception=use_sensor_perception,
        cpu_only=cpu_only,
        use_instance_graph=use_instance_graph,
        manipulation_only=voxel_only,
    )
    return agent, config_path


def create_svm_agent(
    *,
    robot: str,
    host: str,
    port_offset: int,
    dynav_config: str,
    enable_rerun: bool,
    headless: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    allow_missing_depth: bool | None = None,
    config_overrides: list[str] | None = None,
):
    """Build InstanceMemoryController (SVM) + ZMQ client."""
    from emet.controller.controller_instance_memory import RobotAgent as InstanceMemoryController

    config_path = (
        dynav_config if dynav_config != DEFAULT_DYNAV_CONFIG_YAML else InstanceMemoryController.default_config_path
    )
    config_path = resolve_stream_config_path(config_path, dynav_from_default=(config_path == DEFAULT_DYNAV_CONFIG_YAML))
    parameters, allow_from_cfg = load_stream_parameters(robot, host, config_path, overrides=config_overrides)
    allow = allow_missing_depth if allow_missing_depth is not None else allow_from_cfg
    robot_client = _build_robot_client(
        robot=robot,
        host=host,
        port_offset=port_offset,
        parameters=parameters,
        enable_rerun=enable_rerun,
        headless=headless,
        rerun_native=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        allow_missing_depth=allow,
    )
    if not robot_client.start():
        raise click.ClickException(f"Failed to connect to ZMQ on {host} (start sim or bridge first; match --robot).")
    agent = InstanceMemoryController(
        robot_client,
        parameters,
        use_instance_memory=True,
        create_semantic_sensor=True,
    )
    return agent, config_path


def create_scene_graph_agent(
    *,
    robot: str,
    host: str,
    port_offset: int,
    dynav_config: str,
    enable_rerun: bool,
    headless: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    allow_missing_depth: bool | None = None,
    cpu_only: bool = False,
    voxel_only: bool = False,
    config_overrides: list[str] | None = None,
):
    """DynaMem voxel map + open-vocabulary SceneGraphProcessor."""
    from emet.mapping.scene_graph.processor import SceneGraphProcessor

    agent, config_path = create_dynamem_agent(
        robot=robot,
        host=host,
        port_offset=port_offset,
        dynav_config=dynav_config,
        enable_rerun=enable_rerun,
        headless=headless,
        rerun_native=rerun_native,
        rerun_show_panels=rerun_show_panels,
        rerun_debug=rerun_debug,
        allow_missing_depth=allow_missing_depth,
        cpu_only=cpu_only,
        voxel_only=voxel_only,
        config_overrides=config_overrides,
    )
    if voxel_only:
        return agent, config_path
    sg_config_name = "cpu_scene_graph" if cpu_only else "default_scene_graph"
    sg_device = "cpu" if cpu_only else None
    processor = SceneGraphProcessor(config_name=sg_config_name, device=sg_device)
    agent.get_voxel_map().set_scene_graph_processor(processor)
    agent._stream_scene_graph_processor = processor  # noqa: SLF001 — stream stats only
    return agent, config_path


def create_stream_agent(
    backend: StreamBackend,
    *,
    robot: str,
    host: str,
    port_offset: int,
    dynav_config: str,
    enable_rerun: bool,
    headless: bool = False,
    rerun_native: bool = False,
    rerun_show_panels: bool = False,
    rerun_debug: bool = False,
    allow_missing_depth: bool | None = None,
    cpu_only: bool = False,
    use_sensor_perception: bool = True,
    use_instance_graph: bool = True,
    compare_to_gt: bool = False,
    dynav_from_default: bool = True,
    config_overrides: list[str] | None = None,
) -> StreamAgentBundle:
    """Instantiate a mapping agent for ``emet stream --backend``."""
    config_path = resolve_stream_config_path(dynav_config, dynav_from_default=dynav_from_default)
    if config_path != dynav_config and dynav_from_default:
        depth_mode = str(
            load_stream_parameters(robot, host, config_path, overrides=config_overrides)[0].get(
                "depth_source", "sensor"
            )
        ).lower()
        click.echo(
            f"Config: using {config_path!r} for {_robot_key(robot)} @ {host} "
            f"(depth_source={depth_mode!r}; robot overlay from unified config)."
        )

    common = {
        "robot": robot,
        "host": host,
        "port_offset": port_offset,
        "dynav_config": config_path,
        "enable_rerun": enable_rerun,
        "headless": headless,
        "rerun_native": rerun_native,
        "rerun_show_panels": rerun_show_panels,
        "rerun_debug": rerun_debug,
        "allow_missing_depth": allow_missing_depth,
        "cpu_only": cpu_only,
        "config_overrides": config_overrides,
    }
    if backend == "voxel_only":
        agent, config_path = create_dynamem_agent(**common, voxel_only=True)
    elif backend == "dynamem":
        agent, config_path = create_dynamem_agent(**common)
    elif backend in ("static_graph", "graph_eqa"):
        agent, config_path = create_graph_eqa_agent(
            **common,
            use_sensor_perception=use_sensor_perception,
            use_instance_graph=use_instance_graph,
        )
    elif backend == "dynagraph":
        agent, config_path = create_dynagraph_agent(
            **common,
            use_sensor_perception=use_sensor_perception,
            use_instance_graph=use_instance_graph,
            visualize_ground_truth=compare_to_gt,
        )
    elif backend == "ground_truth":
        agent, config_path = create_dynagraph_agent(
            **common,
            use_sensor_perception=False,
            use_instance_graph=True,
            ground_truth_mode=True,
        )
    elif backend == "svm":
        agent, config_path = create_svm_agent(
            robot=robot,
            host=host,
            port_offset=port_offset,
            dynav_config=config_path,
            enable_rerun=enable_rerun,
            headless=headless,
            rerun_native=rerun_native,
            rerun_show_panels=rerun_show_panels,
            rerun_debug=rerun_debug,
            allow_missing_depth=allow_missing_depth,
            config_overrides=config_overrides,
        )
    elif backend == "scene_graph":
        agent, config_path = create_scene_graph_agent(**common)
    else:
        raise click.ClickException(f"Unknown stream backend {backend!r}.")
    return StreamAgentBundle(agent=agent, dynav_resolved=config_path, backend=backend)


def _voxel_point_count(voxel_map: Any) -> int:
    if voxel_map is None:
        return 0
    obs = getattr(voxel_map, "observations", None)
    if obs is not None:
        return len(obs)
    get_pc = getattr(voxel_map, "get_pointcloud", None)
    if callable(get_pc):
        pc = get_pc()
        if pc is not None:
            return int(len(pc))
    return 0


def _map_stats(agent: Any, *, dynav_resolved: str) -> dict[str, Any]:
    vm = agent.get_voxel_map()
    stats: dict[str, Any] = {
        "dynav_config": dynav_resolved,
        "n_voxel_observations": _voxel_point_count(vm),
        "n_updates": int(getattr(agent, "obs_count", 0)),
    }
    if vm is not None and hasattr(vm, "get_2d_map"):
        _, explored = vm.get_2d_map()
        if explored is not None:
            if hasattr(explored, "cpu"):
                explored = explored.cpu().numpy()
            stats["n_voxel_explored_cells"] = int(np.asarray(explored).sum())
    return stats


def _graph_node_counts(graph_memory: Any) -> dict[str, int]:
    nodes = graph_memory.get_nodes()
    object_n = sum(
        1 for n in nodes if not n.is_viewpoint and not getattr(n, "is_frontier", False)
    )
    frontier_n = sum(1 for n in nodes if getattr(n, "is_frontier", False))
    viewpoint_n = sum(1 for n in nodes if n.is_viewpoint)
    total = len(nodes)
    out = {
        "n_graph_nodes": total,
        "n_graph_object_nodes": object_n,
        "n_graph_frontier_nodes": frontier_n,
        "n_graph_viewpoint_nodes": viewpoint_n,
    }
    try:
        from emet.memory.graph_eqa.graph_object_fusion.fusion import max_pairwise_object_bounds_iou

        max_iou = max_pairwise_object_bounds_iou(graph_memory)
        if max_iou is not None:
            out["max_object_bounds_iou"] = max_iou
    except Exception:
        pass
    return out


def stream_stats(agent: Any, backend: str, *, dynav_resolved: str) -> dict[str, Any]:
    """Status dict for periodic ``emet stream`` logging."""
    stats = _map_stats(agent, dynav_resolved=dynav_resolved)
    stats["backend"] = backend
    if backend in ("static_graph", "graph_eqa", "dynagraph", "ground_truth"):
        gm = getattr(agent, "graph_memory", None)
        if gm is not None and hasattr(gm, "get_nodes"):
            stats.update(_graph_node_counts(gm))
        else:
            stats.update(
                {
                    "n_graph_nodes": 0,
                    "n_graph_object_nodes": 0,
                    "n_graph_frontier_nodes": 0,
                    "n_graph_viewpoint_nodes": 0,
                }
            )
    elif backend == "scene_graph":
        processor = getattr(agent, "_stream_scene_graph_processor", None)
        sg = processor.scene_graph if processor is not None else agent.get_voxel_map().get_scene_graph()
        stats["n_scene_graph_objects"] = int(getattr(sg, "num_objects", 0) or 0) if sg is not None else 0
    elif backend == "svm":
        instances = agent.get_voxel_map().get_instances() if hasattr(agent, "get_voxel_map") else []
        stats["n_instances"] = len(instances) if instances is not None else 0
    return stats


def format_stream_stats(stats: dict[str, Any]) -> str:
    parts = [
        f"{stats.get('n_voxel_observations', 0)} voxel obs",
        f"{stats.get('n_voxel_explored_cells', 0)} explored cells",
    ]
    if "n_graph_nodes" in stats:
        obj = stats.get("n_graph_object_nodes", stats.get("n_graph_nodes", 0))
        vp = stats.get("n_graph_viewpoint_nodes", 0)
        fr = stats.get("n_graph_frontier_nodes", 0)
        total = stats.get("n_graph_nodes", 0)
        if vp or fr:
            graph_part = f"graph {obj} obj / {vp} vp / {fr} fr ({total} total)"
            max_iou = stats.get("max_object_bounds_iou")
            if max_iou is not None:
                graph_part += f", max iou {float(max_iou):.2f}"
            parts.append(graph_part)
        else:
            parts.append(f"{total} graph nodes")
    if "n_scene_graph_objects" in stats:
        parts.append(f"{stats.get('n_scene_graph_objects', 0)} scene-graph objects")
    if "n_instances" in stats:
        parts.append(f"{stats.get('n_instances', 0)} instances")
    return ", ".join(parts)


def stream_agent_update(agent: Any, backend: str) -> None:
    """One mapping step for the selected backend."""
    if backend == "svm":
        agent.update(move_head=False)
    else:
        agent.update()


def resolve_stream_backend(
    *,
    backend: str | None,
    cameras_only: bool,
) -> str | None:
    """Resolve ``--backend`` vs ``--cameras-only`` for ``emet stream``."""
    if cameras_only and backend:
        raise click.UsageError("Use either --cameras-only or --backend <name> (not both).")
    if cameras_only:
        return None
    return backend
