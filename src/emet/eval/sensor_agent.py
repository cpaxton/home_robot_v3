# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Bind the existing shared task agent to the measured-pose evaluation client."""

import base64
import json
import time
from io import BytesIO

import httpx
import numpy as np
from openai import OpenAI
from PIL import Image

from emet.llms.openai_vllm_client import OpenaiVLLMClient


class AuditedPNGClient(OpenaiVLLMClient):
    """Lossless remote transport with complete requests/responses for inspection."""

    def __init__(self, *, base_url, model, trace_path):
        super().__init__(base_url=base_url, model=model, image_max_side=0, max_tokens=2048)

        def record(response):
            response.read()
            with trace_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "timestamp_ns": time.monotonic_ns(),
                            "request": json.loads(response.request.content),
                            "http_status": response.status_code,
                            "response": response.json(),
                        }
                    )
                    + "\n"
                )

        self._openai.close()
        self._openai = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=240,
            max_retries=0,
            http_client=httpx.Client(event_hooks={"response": [record]}),
        )

    def _rgb_to_jpeg_data_url(self, rgb):
        encoded = BytesIO()
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(encoded, format="PNG")
        return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")


def build_sensor_agent(robot, model_config, output, *, max_images):
    """Construct LazyGraphController normally, then bind the supplied model API.

    This preserves the production graph, voxel retrieval, canonical tool loop,
    image selection and conversation. Robot, transport and planning geometry
    share the qualified boundary. Native GraphEQA and emet retain their
    respective semantic memory construction algorithms.
    """
    from emet.core.parameters import get_parameters
    from emet.eval.benchmark_dynagraph import apply_habitat_eqa_method_parameters
    from emet.eval.stack import build_memory_agent
    from emet.llms.graph_eqa_vlm import GraphEQAVLMClient
    from emet.llms.prompts.hmeqa_eqa_prompt import HMEQA_EQA_PROMPT
    from emet.motion.algo.a_star import AStar

    params = apply_habitat_eqa_method_parameters(get_parameters("dynav_config.yaml"), "lazy_graph")
    params.set("enable_tts", False)
    params.set("depth_source", "sensor")
    params.set("min_depth", 0.1)
    params.set("max_depth", 6.0)
    params.set("pad_obstacles", 0)
    params.set("add_local_radius_points", False)
    params.set("filters/smooth_kernel_size", 0)
    params.set("floor_evidence", dict(robot.navigation.map.parameters["floor_evidence"]))
    params.set("motion_planner/obstacle_map_mode", "physical")
    params.set("eqa/habitat_perfect_nav", False)
    params.set("eqa/agentic_decision_policy", "model_final")
    params.set("eqa/agentic_verify", True)
    params.set("eqa_vl/eqa_max_images", max_images)
    params.set("eqa_vl/answer_max_new_tokens", 2048)
    agent = build_memory_agent(
        robot=robot,
        parameters=params,
        backend="lazy_graph",
        harness="habitat_eqa",
        manipulation_only=True,
        eqa=True,
        defer_eqa_vllm=True,
        use_instance_graph=False,
        use_sensor_perception=True,
        log=str(output / "memory"),
        apply_harness_profile=False,
        server_ip="127.0.0.1",  # Required by the controller's unused manipulation REQ socket.
    )
    # Keep the semantic voxel memory, but plan on the same dense, measured
    # geometry as the reference. Otherwise the sparse semantic cadence loses
    # under-footprint floor returns and rejects the safe stationary spawn.
    # Footprint sweeps remain mandatory; there is no extra clearance padding.
    agent.space = robot.navigation.space
    agent.planner = AStar(agent.space, min_clearance_m=0.0, clearance_cost_weight=0.0)
    agent._min_clearance_m = 0.0
    agent._clearance_cost_weight = 0.0
    remote = AuditedPNGClient(
        base_url=model_config["base_url"], model=model_config["name"], trace_path=output / "model_calls.jsonl"
    )
    keyword = GraphEQAVLMClient(remote, system_prompt=None, max_tokens=64)
    eqa = GraphEQAVLMClient(remote, system_prompt=HMEQA_EQA_PROMPT, max_tokens=2048)
    for memory in (agent.graph_memory, agent.voxel_map):
        memory.image_description_client, memory.eqa_client = keyword, eqa
    if agent.sensor_builder is not None:
        agent.sensor_builder._perception = keyword
        agent.sensor_builder._lazy_vl_client = keyword
    (output / "parameters.json").write_text(json.dumps(params.data, indent=2, default=str) + "\n")
    (output / "navigation_binding.json").write_text(
        json.dumps(
            {
                "geometry": "shared_dense_observed_space",
                "semantic_memory": "production_lazy_graph_and_voxel_retrieval",
                "footprint_radius_m": robot.navigation.footprint.radius,
                "additional_clearance_m": 0.0,
                "footprint_sweep_required": True,
            },
            indent=2,
        )
        + "\n"
    )
    return agent, remote
