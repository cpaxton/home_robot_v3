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
#
# Graph-based EQA memory: re-implementation inspired by GraphEQA
# (https://arxiv.org/abs/2412.14480). Object-centric scene graph + task-relevant
# images for embodied question answering. No code copied from closed-source repos.

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from emet.core.parameters import Parameters


@dataclass
class GraphNode:
    """Single node in the scene graph: an object or region with label and position."""

    node_id: int
    labels: list[str]
    xyz: np.ndarray  # (3,) world position
    obs_id: int  # 1-based index into observations list
    description: str | None = None  # optional VLM-generated description


@dataclass
class GraphObservation:
    """One observation (image + pose + labels) used to build the graph."""

    obs_id: int  # 1-based
    rgb: np.ndarray  # (H, W, 3)
    xyz: np.ndarray  # (3,) e.g. mean of visible points or camera position
    labels: list[str]
    description: str | None = None  # optional VLM-generated description


def _near(p1: np.ndarray, p2: np.ndarray, max_dist: float = 1.5) -> bool:
    return float(np.linalg.norm(p1[:2] - p2[:2])) <= max_dist


def _on(p_lower: np.ndarray, p_upper: np.ndarray, z_thresh: float = 0.15) -> bool:
    """Heuristic: lower object is 'on' upper if roughly below and close in xy."""
    if p_lower[2] >= p_upper[2]:
        return False
    return abs(p_lower[2] - p_upper[2]) <= z_thresh + 0.2 and float(np.linalg.norm(p_lower[:2] - p_upper[:2])) < 0.5


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
        parameters: Parameters | None = None,
        max_near_distance: float = 1.5,
        eqa_client: Callable[..., str] | None = None,
        image_description_client: Callable[..., str] | None = None,
        log_dir: str = "graph_eqa_log",
    ):
        self.parameters = parameters or {}
        self.max_near_distance = max_near_distance
        self._nodes: list[GraphNode] = []
        self._edges: list[tuple[int, int, str]] = []  # (id1, id2, relation)
        self._observations: list[GraphObservation] = []
        self._next_obs_id = 1
        self._question: str | None = None
        self._relevant_objects: list[str] | None = None
        self._history_outputs: list[str] = []

        self.log_dir = log_dir
        self.eqa_client = eqa_client
        self.image_description_client = image_description_client

        if self.eqa_client is None or self.image_description_client is None:
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
        self.image_description_client = Qwen25VLClient(model_size="3B", quantization="int4", max_tokens=20)
        self.eqa_client = GeminiClient(EQA_PROMPT, model="gemini-2.5-flash")

    def add_observation(
        self,
        rgb: np.ndarray | Image.Image,
        xyz: np.ndarray,
        labels: list[str],
        description: str | None = None,
    ) -> int:
        """
        Add one observation to the graph: create a node and update edges.

        Args:
            rgb: RGB image (H, W, 3) or PIL Image
            xyz: (3,) world position for this observation (e.g. camera or centroid)
            labels: list of object/region labels (e.g. from a VLM)
            description: optional text description of the scene (e.g. from VLM)

        Returns:
            obs_id: 1-based observation id (used as image id in EQA).
        """
        if isinstance(rgb, Image.Image):
            rgb = np.array(rgb)
        obs_id = self._next_obs_id
        self._next_obs_id += 1
        node_id = len(self._nodes) + 1
        node = GraphNode(
            node_id=node_id,
            labels=labels,
            xyz=np.asarray(xyz, dtype=float),
            obs_id=obs_id,
            description=description,
        )
        self._nodes.append(node)
        self._observations.append(
            GraphObservation(obs_id=obs_id, rgb=rgb, xyz=xyz, labels=labels, description=description)
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
            lines.append(
                f"Node {n.node_id}: {lbl} at ({n.xyz[0]:.2f}, {n.xyz[1]:.2f}, {n.xyz[2]:.2f}) [Image {n.obs_id}]"
            )
        for a, b, rel in self._edges:
            b_str = "floor" if b == -1 else str(b)
            lines.append(f"  {rel}({a}, {b_str})")
        return "SCENE_GRAPH:\n" + "\n".join(lines) if lines else "SCENE_GRAPH: (empty)"

    def to_tree_string(self, indent: str = "  ") -> str:
        """
        Format the 3D spatial scene graph as an indented tree (text).

        Root = Scene; Floor is a virtual node; objects on floor are children of Floor;
        objects on other objects are nested. "Near" relations are listed at the end.
        Includes object labels, (x,y,z), and optional descriptions.
        """
        edge_set = set(self._edges)
        node_by_id = {n.node_id: n for n in self._nodes}

        def on_floor(nid: int) -> bool:
            return (nid, -1, "on") in edge_set

        def has_on_parent(nid: int) -> int | None:
            """Return node_id that this node is 'on', or None if on floor or no 'on' edge."""
            for a, b, rel in edge_set:
                if rel == "on" and a == nid and b != -1:
                    return b
            return None

        def children_of(nid: int | None) -> list[GraphNode]:
            if nid is None:
                # Floor children: explicitly on floor, or no "on" relation (in-scene)
                out = [
                    node_by_id[n.node_id]
                    for n in self._nodes
                    if on_floor(n.node_id) or has_on_parent(n.node_id) is None
                ]
            else:
                out = [node_by_id[a] for a, b, rel in edge_set if rel == "on" and b == nid and a in node_by_id]
            return sorted(out, key=lambda n: n.node_id)

        near_pairs = [(a, b) for a, b, rel in self._edges if rel == "near" and a < b]

        lines: list[str] = []
        lines.append("Scene (3D spatial graph)")
        lines.append(f"{indent}Floor")

        def visit(node: GraphNode, depth: int) -> None:
            pref = indent * (depth + 1)
            x, y, z = float(node.xyz[0]), float(node.xyz[1]), float(node.xyz[2])
            lbl = ", ".join(node.labels) if node.labels else "object"
            line = f"{pref}[{node.node_id}] {lbl}  at ({x:.2f}, {y:.2f}, {z:.2f})"
            if node.description:
                line += f"  — {node.description}"
            lines.append(line)
            for c in children_of(node.node_id):
                visit(c, depth + 1)

        for node in children_of(None):
            visit(node, 1)

        if near_pairs:
            lines.append("")
            lines.append("Near relations:")
            for a, b in near_pairs:
                na, nb = node_by_id.get(a), node_by_id.get(b)
                la = ", ".join(na.labels) if na and na.labels else str(a)
                lb = ", ".join(nb.labels) if nb and nb.labels else str(b)
                lines.append(f"{indent}{la} — {lb}")

        return "\n".join(lines) if lines else "Scene (3D spatial graph): (empty)"

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

    def _select_relevant_obs_ids(self, max_images: int = 6) -> list[int]:
        """Select observation IDs whose labels match relevant_objects (1-based)."""
        if not self._relevant_objects or not self._observations:
            return [o.obs_id for o in self._observations[:max_images]]
        seen: set = set()
        out: list[int] = []
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

    def _get_image_descriptions_str(self, obs_ids: list[int]) -> str:
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

    def parse_answer(self, answer_outputs: str) -> tuple[str, str, bool, str, str]:
        """Parse mLLM output into reasoning, answer, confidence, action, confidence_reasoning."""

        def extract_between(text: str, start: str, end: str) -> str:
            try:
                return text.split(start, 1)[1].split(end, 1)[0].strip().replace("\n", "").replace("\t", "")
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

    def _target_point_from_image_id(self, image_id: int) -> np.ndarray | None:
        """Return (x, y, 1) for the observation's position when mLLM suggests navigating to that image."""
        for obs in self._observations:
            if obs.obs_id == image_id:
                return np.array([obs.xyz[0], obs.xyz[1], 1.0], dtype=float)
        return None

    def query_answer(
        self,
        question: str,
        xyt: Any | np.ndarray | list | None = None,
        planner: Any = None,
    ) -> tuple[str, str, bool, str, np.ndarray | None, list[Image.Image]]:
        """
        Answer the question using the scene graph and task-relevant images.
        Same return contract as voxel_dynamem.SparseVoxelMap.query_answer.

        Returns:
            reasoning, answer, confidence, confidence_reasoning, target_point, relevant_images
        """
        self.extract_relevant_objects(question)
        obs_ids = self._select_relevant_obs_ids(max_images=6)
        graph_str = self.to_string()
        img_desc_str = self._get_image_descriptions_str(obs_ids)

        commands: list[Any] = ["Question: " + question]
        commands.append("HISTORY: ")
        for i, h in enumerate(self._history_outputs):
            commands.append("Iteration_" + str(i) + ":" + h)
        commands.append(graph_str)
        commands.append(img_desc_str)

        relevant_images: list[Image.Image] = []
        for obs in self._observations:
            if obs.obs_id in obs_ids:
                relevant_images.append(Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB"))
                commands.append(Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB"))

        raw = self.eqa_client(commands)
        answer_outputs = raw.replace("*", "").replace("/", "").replace("#", "").lower()

        reasoning, answer, confidence, action, confidence_reasoning = self.parse_answer(answer_outputs)

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
                "Answer:"
                + answer
                + "\nReasoning:"
                + reasoning
                + "\nConfidence:"
                + str(confidence)
                + "\nAction: Navigate to Image "
                + action.strip()
                + "\nConfidence_reasoning:"
                + confidence_reasoning
            )
        else:
            self._history_outputs.append(
                "Answer:"
                + answer
                + "\nReasoning:"
                + reasoning
                + "\nConfidence:"
                + str(confidence)
                + "\nAction:\nConfidence_reasoning: "
                + confidence_reasoning
            )

        return (
            reasoning,
            answer,
            confidence,
            confidence_reasoning,
            target_point,
            relevant_images,
        )

    def fill_descriptions_from_vlm(
        self,
        prompt: str | None = None,
        max_tokens: int = 80,
    ) -> None:
        """
        Fill missing node/observation descriptions using the VLM (e.g. Qwen 2.5-VL / 3.5).
        Skips observations that already have a description. Can be slow for many images.
        """
        if self.image_description_client is None:
            self._init_clients()
        default_prompt = (
            "In one short sentence, describe what is visible in this image: "
            "main objects, their arrangement, and any notable spatial relationships. "
            "Be concise."
        )
        prompt = prompt or default_prompt
        for obs in self._observations:
            if obs.description:
                continue
            try:
                # VLM accepts list of text + image(s)
                out = self.image_description_client(
                    [prompt, Image.fromarray(obs.rgb.astype(np.uint8), mode="RGB")],
                    verbose=False,
                )
                if isinstance(out, str) and out.strip():
                    desc = out.strip()
                    # Update observation (same object as stored)
                    obs.description = desc
                    # Update corresponding node
                    for n in self._nodes:
                        if n.obs_id == obs.obs_id:
                            n.description = desc
                            break
            except Exception:
                continue

    def get_observations(self) -> list[GraphObservation]:
        return list(self._observations)

    def get_nodes(self) -> list[GraphNode]:
        return list(self._nodes)

    def get_edges(self) -> list[tuple[int, int, str]]:
        return list(self._edges)

    def print_memory(self) -> str:
        """
        Return the 3D scene graph as a human-readable tree (same as to_tree_string).
        Use this as the canonical "print" output for the graph memory.
        """
        return self.to_tree_string()
