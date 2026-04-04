# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Graph-based EQA memory: re-implementation inspired by GraphEQA
# (https://arxiv.org/abs/2412.14480). Object-centric scene graph + task-relevant
# images for embodied question answering. No code copied from closed-source repos.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from emet.core.parameters import Parameters


@dataclass
class GraphNode:
    """Single node in the scene graph: an object or region with label and position."""

    node_id: int
    labels: List[str]
    xyz: np.ndarray  # (3,) world position
    obs_id: int  # 1-based index into observations list


@dataclass
class GraphObservation:
    """One observation (image + pose + labels) used to build the graph."""

    obs_id: int  # 1-based
    rgb: np.ndarray  # (H, W, 3)
    xyz: np.ndarray  # (3,) e.g. mean of visible points or camera position
    labels: List[str]


def _near(p1: np.ndarray, p2: np.ndarray, max_dist: float = 1.5) -> bool:
    return float(np.linalg.norm(p1[:2] - p2[:2])) <= max_dist


def _on(p_lower: np.ndarray, p_upper: np.ndarray, z_thresh: float = 0.15) -> bool:
    """Heuristic: lower object is 'on' upper if roughly below and close in xy."""
    if p_lower[2] >= p_upper[2]:
        return False
    return (
        abs(p_lower[2] - p_upper[2]) <= z_thresh + 0.2
        and float(np.linalg.norm(p_lower[:2] - p_upper[:2])) < 0.5
    )


def _on_floor(p: np.ndarray, floor_z: float = 0.05) -> bool:
    return float(p[2]) <= floor_z


class GraphEQAMemory:
    """
    Graph-based semantic memory for Embodied Question Answering (EQA).

    Maintains an object-centric scene graph (nodes = objects/regions with labels and
    3D positions; edges = spatial relations). Uses the same EQA query contract as
    the DynaMem voxel map: query_answer(question, xyt, planner) returns
    (reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images).
    """

    def __init__(
        self,
        parameters: Optional[Parameters] = None,
        max_near_distance: float = 1.5,
        eqa_client: Optional[Callable[..., str]] = None,
        image_description_client: Optional[Callable[..., str]] = None,
        log_dir: str = "graph_eqa_log",
        defer_llm_clients: bool = False,
    ):
        self.parameters = parameters or {}
        self.max_near_distance = max_near_distance
        self._nodes: List[GraphNode] = []
        self._edges: List[Tuple[int, int, str]] = []  # (id1, id2, relation)
        self._observations: List[GraphObservation] = []
        self._next_obs_id = 1
        self._question: Optional[str] = None
        self._relevant_objects: Optional[List[str]] = None
        self._history_outputs: List[str] = []

        self.log_dir = log_dir
        self.eqa_client = eqa_client
        self.image_description_client = image_description_client
        self._defer_llm_clients = defer_llm_clients

        if not defer_llm_clients and (
            self.eqa_client is None or self.image_description_client is None
        ):
            self._init_clients()

    def _ensure_llm_clients(self) -> None:
        """Load Gemini + Qwen VL on first use when defer_llm_clients=True."""
        if self.eqa_client is not None and self.image_description_client is not None:
            return
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize EQA and image-description clients (same pattern as voxel_dynamem)."""
        try:
            from emet.llms.gemini_client import GeminiClient
            from emet.llms.prompts.eqa_prompt import EQA_PROMPT
            from emet.llms.qwen_client import Qwen25VLClient
        except ImportError as e:
            raise ImportError(
                "GraphEQA memory requires emet.llms (Gemini, Qwen) for EQA. Install extras and set GOOGLE_API_KEY."
            ) from e
        self.image_description_client = Qwen25VLClient(
            model_size="3B", quantization="int4", max_tokens=20
        )
        self.eqa_client = GeminiClient(EQA_PROMPT, model="gemini-2.5-flash")

    def add_observation(
        self,
        rgb: Union[np.ndarray, Image.Image],
        xyz: np.ndarray,
        labels: List[str],
    ) -> int:
        """
        Add one observation to the graph: create a node and update edges.

        Args:
            rgb: RGB image (H, W, 3) or PIL Image
            xyz: (3,) world position for this observation (e.g. camera or centroid)
            labels: list of object/region labels (e.g. from a VLM)

        Returns:
            obs_id: 1-based observation id (used as image id in EQA).
        """
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        obs_id = self._next_obs_id
        self._next_obs_id += 1
        node_id = len(self._nodes) + 1
        node = GraphNode(node_id=node_id, labels=labels, xyz=np.asarray(xyz, dtype=float), obs_id=obs_id)
        self._nodes.append(node)
        self._observations.append(
            GraphObservation(obs_id=obs_id, rgb=rgb, xyz=xyz, labels=labels)
        )
        self._update_edges()
        return obs_id

    def _update_edges(self) -> None:
        """Compute pairwise spatial relations (near, on, on_floor) from node positions."""
        self._edges.clear()
        for i, na in enumerate(self._nodes):
            if _on_floor(na.xyz):
                self._edges.append((na.node_id, -1, "on"))  # -1 = floor
            for j, nb in enumerate(self._nodes):
                if i >= j:
                    continue
                if _near(na.xyz, nb.xyz, self.max_near_distance):
                    if (nb.node_id, na.node_id, "near") not in self._edges:
                        self._edges.append((na.node_id, nb.node_id, "near"))
                if _on(na.xyz, nb.xyz):
                    self._edges.append((na.node_id, nb.node_id, "on"))
                elif _on(nb.xyz, na.xyz):
                    self._edges.append((nb.node_id, na.node_id, "on"))

    def to_string(self) -> str:
        """Serialize the scene graph to a string for mLLM prompts."""
        lines = []
        for n in self._nodes:
            lbl = ", ".join(n.labels) if n.labels else "object"
            lines.append(f"Node {n.node_id}: {lbl} at ({n.xyz[0]:.2f}, {n.xyz[1]:.2f}, {n.xyz[2]:.2f}) [Image {n.obs_id}]")
        for a, b, rel in self._edges:
            b_str = "floor" if b == -1 else str(b)
            lines.append(f"  {rel}({a}, {b_str})")
        return "SCENE_GRAPH:\n" + "\n".join(lines) if lines else "SCENE_GRAPH: (empty)"

    def extract_relevant_objects(self, question: str) -> None:
        """Extract keywords from the question for image selection (same idea as DynaMem)."""
        if self._question == question:
            return
        self._question = question
        prompt = (
            "Assume there is an agent doing Question Answering in an environment. "
            "When it receives a question, tell the agent few objects (preferably 1-3) to pay attention to. "
            "Example: Where is the pen? -> pen. Is there grey cloth on cloth hanger? -> grey cloth, cloth hanger"
        )
        out = self.image_description_client([prompt, question])
        self._relevant_objects = [s.strip() for s in out.split(",") if s.strip()]

    def _select_relevant_obs_ids(
        self, max_images: int = 6
    ) -> List[int]:
        """Select observation IDs whose labels match relevant_objects (1-based)."""
        if not self._relevant_objects or not self._observations:
            return [o.obs_id for o in self._observations[:max_images]]
        seen: set = set()
        out: List[int] = []
        for obj in self._relevant_objects:
            obj_lower = obj.lower()
            for o in self._observations:
                if o.obs_id in seen:
                    continue
                if any(obj_lower in l.lower() for l in o.labels):
                    seen.add(o.obs_id)
                    out.append(o.obs_id)
                    if len(out) >= max_images:
                        return out
        # If few matches, add remaining up to max_images
        for o in self._observations:
            if o.obs_id not in seen:
                seen.add(o.obs_id)
                out.append(o.obs_id)
                if len(out) >= max_images:
                    break
        return out

    def _get_image_descriptions_str(
        self, obs_ids: List[int]
    ) -> str:
        """Build IMAGE_DESCRIPTIONS string for the prompt (1-indexed image ids)."""
        options = []
        for i, obs in enumerate(self._observations, start=1):
            lbl = ", ".join(obs.labels) if obs.labels else "object"
            line = f"{i}. {lbl} at ({obs.xyz[0]:.2f}, {obs.xyz[1]:.2f});"
            if obs.obs_id in obs_ids:
                idx = obs_ids.index(obs.obs_id) + 1
                line += f" This observation is associated with Image {idx};"
            options.append(line)
        return "IMAGE_DESCRIPTIONS: " + "\n".join(options) if options else "IMAGE_DESCRIPTIONS: (none)"

    def parse_answer(self, answer_outputs: str) -> Tuple[str, str, bool, str, str]:
        """Parse mLLM output into reasoning, answer, confidence, action, confidence_reasoning."""
        def extract_between(text: str, start: str, end: str) -> str:
            try:
                return (
                    text.split(start, 1)[1]
                    .split(end, 1)[0]
                    .strip()
                    .replace("\n", "")
                    .replace("\t", "")
                )
            except IndexError:
                return ""

        def extract_after(text: str, start: str) -> str:
            try:
                return text.split(start, 1)[1].strip().replace("\n", "").replace("\t", "")
            except IndexError:
                return ""

        reasoning = extract_between(answer_outputs, "reasoning:", "answer:")
        answer = extract_between(answer_outputs, "answer:", "confidence:")
        confidence_text = extract_between(answer_outputs, "confidence:", "action:")
        confidence = "true" in confidence_text.replace(" ", "").lower()
        action = extract_between(answer_outputs, "action:", "confidence_reasoning:")
        confidence_reasoning = extract_after(answer_outputs, "confidence_reasoning:")
        return reasoning, answer, confidence, action, confidence_reasoning

    def _target_point_from_image_id(self, image_id: int) -> Optional[np.ndarray]:
        """Return (x, y, 1) for the observation's position when mLLM suggests navigating to that image."""
        for obs in self._observations:
            if obs.obs_id == image_id:
                return np.array([obs.xyz[0], obs.xyz[1], 1.0], dtype=float)
        return None

    def query_answer(
        self,
        question: str,
        xyt: Optional[Union[Any, np.ndarray, list]] = None,
        planner: Any = None,
    ) -> Tuple[
        str, str, bool, str, Optional[np.ndarray], List[Image.Image]
    ]:
        """
        Answer the question using the scene graph and task-relevant images.
        Same return contract as voxel_dynamem.SparseVoxelMap.query_answer.

        Returns:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images
        """
        self._ensure_llm_clients()
        self.extract_relevant_objects(question)
        obs_ids = self._select_relevant_obs_ids(max_images=6)
        graph_str = self.to_string()
        img_desc_str = self._get_image_descriptions_str(obs_ids)

        commands: List[Any] = ["Question: " + question]
        commands.append("HISTORY: ")
        for i, h in enumerate(self._history_outputs):
            commands.append("Iteration_" + str(i) + ":" + h)
        commands.append(graph_str)
        commands.append(img_desc_str)

        relevant_images: List[Image.Image] = []
        for obs in self._observations:
            if obs.obs_id in obs_ids:
                relevant_images.append(Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB"))
                commands.append(Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB"))

        raw = self.eqa_client(commands)
        answer_outputs = raw.replace("*", "").replace("/", "").replace("#", "").lower()

        reasoning, answer, confidence, action, confidence_reasoning = self.parse_answer(
            answer_outputs
        )

        target_point = None
        if not confidence and action.strip():
            # Parse action as image id (integer)
            match = re.search(r"\d+", action.strip())
            if match:
                image_id = int(match.group())
                # image_id from mLLM is 1-based observation index
                if 1 <= image_id <= len(self._observations):
                    obs = self._observations[image_id - 1]
                    target_point = np.array([obs.xyz[0], obs.xyz[1], 1.0], dtype=float)
                else:
                    target_point = self._target_point_from_image_id(image_id)
            self._history_outputs.append(
                "Answer:" + answer + "\nReasoning:" + reasoning
                + "\nConfidence:" + str(confidence) + "\nAction: Navigate to Image " + action.strip()
                + "\nConfidence_reasoning:" + confidence_reasoning
            )
        else:
            self._history_outputs.append(
                "Answer:" + answer + "\nReasoning:" + reasoning
                + "\nConfidence:" + str(confidence) + "\nAction:\nConfidence_reasoning: " + confidence_reasoning
            )

        return (
            reasoning,
            answer,
            confidence,
            confidence_reasoning,
            target_point,
            relevant_images,
        )

    def get_observations(self) -> List[GraphObservation]:
        return list(self._observations)

    def get_nodes(self) -> List[GraphNode]:
        return list(self._nodes)

    def get_edges(self) -> List[Tuple[int, int, str]]:
        return list(self._edges)
