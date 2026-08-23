# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared memory-agent construction for interactive agent and paper harnesses."""

from __future__ import annotations

from typing import Any, Literal

from emet.config.embodied_agent_config import (
    GRAPH_EQA_FAMILY_BACKENDS,
    coerce_embodied_agent_for_memory_backend,
    normalize_memory_backend,
)
from emet.eval.benchmark_dynagraph import apply_dynagraph_harness
from emet.eval.memory_backends import DYNAGRAPH, STATIC_GRAPH
from emet.utils.logger import Logger

logger = Logger(__name__)

MemoryBackendName = Literal["dynagraph", "static_graph", "dynamem", "open_vocab"]
HarnessName = Literal[
    "interactive",
    "habitat_eqa",
    "habitat_ovmm_find",
    "ovmm_find_phase",
    "sqa3d",
    "dynamic_explore",
]


def compose_eqa_question(question: str, extra_instruction: str | None = None) -> str:
    """Append optional extra instruction to an EQA question (identical for harness and agent eval)."""
    q = str(question or "").strip()
    extra = str(extra_instruction or "").strip()
    if not extra:
        return q
    if not q:
        return extra
    return f"{q}\n\nAdditional instructions:\n{extra}"


def build_memory_agent(
    *,
    robot: Any,
    parameters: Any,
    backend: str = "dynagraph",
    harness: HarnessName | str = "interactive",
    semantic_sensor: Any | None = None,
    log: str | None = None,
    server_ip: str | None = None,
    mllm: bool = False,
    manipulation_only: bool = False,
    cpu_only: bool = False,
    eqa: bool = False,
    defer_eqa_vllm: bool = False,
    embodied_agent: Any | None = None,
    use_instance_graph: bool | None = None,
    use_sensor_perception: bool | None = None,
    apply_harness_profile: bool = True,
) -> Any:
    """Construct Dynamem / GraphEQA / Dynagraph / open-vocab controller with shared profile wiring.

    Interactive agent and paper runners should call this so merge/staleness/EQA flags stay aligned.
    At most one object-graph plug-in is attached (derived from ``backend``).
    """
    backend_key = normalize_memory_backend(backend)
    harness_key = str(harness or "interactive").strip().lower()
    embodied_agent = coerce_embodied_agent_for_memory_backend(embodied_agent, backend_key)

    if apply_harness_profile and backend_key in GRAPH_EQA_FAMILY_BACKENDS:
        method = DYNAGRAPH if backend_key == DYNAGRAPH else STATIC_GRAPH
        if harness_key == "interactive":
            from emet.eval.benchmark_dynagraph import apply_dynagraph_profile

            apply_dynagraph_profile(parameters, "interactive")
        else:
            apply_dynagraph_harness(parameters, harness_key, method)  # type: ignore[arg-type]

    if backend_key in GRAPH_EQA_FAMILY_BACKENDS:
        if isinstance(parameters, dict):
            parameters.setdefault("dynagraph_merge_xy_m", 0.45)
            parameters.setdefault("dynagraph_staleness_horizon", 256)
        else:
            if parameters.get("dynagraph_merge_xy_m") is None:
                parameters["dynagraph_merge_xy_m"] = 0.45
            if parameters.get("dynagraph_staleness_horizon") is None:
                parameters["dynagraph_staleness_horizon"] = 256

        inst = True if use_instance_graph is None else bool(use_instance_graph)
        sens = False if use_sensor_perception is None else bool(use_sensor_perception)
        if getattr(embodied_agent, "graph_eqa_memory", None) is not None:
            gcfg = embodied_agent.graph_eqa_memory
            if getattr(gcfg, "enabled", False):
                if use_instance_graph is None:
                    inst = bool(gcfg.use_instance_graph)
                if use_sensor_perception is None:
                    sens = bool(gcfg.use_sensor_perception)

        common = {
            "robot": robot,
            "parameters": parameters,
            "semantic_sensor": semantic_sensor,
            "log": log,
            "server_ip": server_ip,
            "mllm": mllm,
            "manipulation_only": manipulation_only,
            "cpu_only": cpu_only,
            "use_instance_graph": inst,
            "use_sensor_perception": sens,
            "eqa": True if eqa else None,
            "defer_eqa_vllm": defer_eqa_vllm if eqa else True,
        }
        if backend_key == "dynagraph":
            from emet.controller.controller_dynagraph import DynagraphController

            logger.info(
                f"build_memory_agent: dynagraph harness={harness_key} instance_graph={inst} sensor={sens} eqa={eqa}"
            )
            return DynagraphController(**common)
        from emet.controller.controller_graph_eqa import GraphEQAController

        logger.info(
            f"build_memory_agent: static_graph harness={harness_key} instance_graph={inst} sensor={sens} eqa={eqa}"
        )
        return GraphEQAController(**common)

    from emet.controller.controller_dynamem import RobotAgent

    logger.info(
        f"build_memory_agent: {backend_key} harness={harness_key} eqa={eqa} "
        f"ov={embodied_agent.open_vocab_scene_graph.enabled} "
        f"ge={embodied_agent.graph_eqa_memory.enabled}"
    )
    return RobotAgent(
        robot,
        parameters,
        semantic_sensor,
        log=log,
        server_ip=server_ip,
        mllm=mllm,
        manipulation_only=manipulation_only,
        cpu_only=cpu_only,
        use_instance_memory=parameters.get("use_instance_memory", True),
        eqa=eqa,
        defer_eqa_vllm=defer_eqa_vllm,
        embodied_agent=embodied_agent,
    )
