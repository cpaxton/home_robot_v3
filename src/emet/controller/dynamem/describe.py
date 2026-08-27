# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""User-facing scene description (YoloE / OWL / VLM / memory)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from emet.agent.env_flags import env_agent_camera_debug, env_agent_model_debug
from emet.controller.dynamem.constants import (
    _DEFAULT_DESCRIBE_CONFIDENCE,
    _DEFAULT_DESCRIBE_MAX_LABELS,
    _DESCRIBE_SCENE_OWL_QUERIES,
    _DESCRIBE_SCENE_STRUCTURE_LABELS,
    _DESCRIBE_SCENE_YOLOE_LABELS,
)
from emet.controller.zmq_stream_control import paused_robot_streams
from emet.mapping.instance import instances_to_text
from emet.perception.detection.owl import OwlPerception
from emet.perception.detection.yoloe import YoloEPerception
from emet.utils.logger import Logger

logger = Logger(__name__)


def dump_memory_to_text(
    self,
    include_bounds: bool = True,
    class_names: dict[int, str] | None = None,
) -> str:
    """Return instance memory as human-readable text (for logging or CLI dump)."""
    if not self.voxel_map.use_instance_memory:
        return "Instance memory is disabled."
    instances = self.get_voxel_map().get_instances()
    if class_names is None and self.semantic_sensor is not None and self.semantic_sensor.is_semantic():
        class_names = {}
        for inst in instances:
            cid = inst.get_category_id()
            if cid is not None and cid not in class_names:
                name = self.semantic_sensor.get_class_name_for_id(cid)
                if name is not None:
                    class_names[cid] = name
    return instances_to_text(instances, class_names=class_names, include_bounds=include_bounds)


def describe_head_camera_scene_text(
    self,
    *,
    graph_memory: Any | None = None,
    memory_backend: Any | None = None,
    graph_memory_backend: Any | None = None,
) -> str:
    """User-facing answer for ``describe_scene`` (current view — not a motion skill).

    Priority for "what can you see":
    1. Caption the **current** head RGB (VLM when loaded).
    2. Ground / enrich with graph/map labels already known.
    3. Optional curated detector fallback if configured.
    Does **not** look around or explore — use ``scan_environment`` / ``explore`` for that.
    """
    mem_kw = {
        "graph_memory": graph_memory,
        "memory_backend": memory_backend,
        "graph_memory_backend": graph_memory_backend,
    }
    rgb, depth = self._describe_scene_capture_rgb()
    if isinstance(rgb, str):
        return rgb  # error string from capture

    det_cfg = self._detection_cfg()
    if env_agent_model_debug():
        print(
            "[model debug] describe_scene: caption current view + ground with graph/memory; "
            "no auto look-around "
            f"(detector_fallback={bool(det_cfg.get('describe_use_detector_fallback', False))})",
            flush=True,
        )

    text = self._describe_scene_try_sources(rgb, depth, det_cfg, **mem_kw)
    if text:
        return text

    return (
        "I don't have a captioner or mapped object labels for this view yet. "
        "I'm sending a photo of what is in front of me — ask me to look around or explore "
        "if you want me to map more of the room."
    )


def _describe_scene_capture_rgb(self) -> tuple[np.ndarray, np.ndarray | None] | tuple[str, None]:
    """Return (rgb, depth) or (error_message, None)."""
    if self.robot is None or not hasattr(self.robot, "get_observation"):
        return "No robot view available.", None
    obs = self.robot.get_observation()
    if obs is None or getattr(obs, "rgb", None) is None:
        return "No current image.", None
    rgb = np.asarray(obs.rgb)
    if rgb.dtype != np.uint8:
        if rgb.size and float(np.nanmax(rgb)) <= 1.0 + 1e-6:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return "Head camera image has an unexpected shape.", None

    from emet.llms.vl_image import downsample_rgb_hwc, eqa_vl_image_kwargs

    eqa_cfg: dict[str, Any] = {}
    if isinstance(self.parameters, dict):
        eqa_cfg = self.parameters.get("eqa", {}) or {}
    elif self.parameters is not None and hasattr(self.parameters, "get"):
        raw = self.parameters.get("eqa", {}) or {}
        eqa_cfg = raw if isinstance(raw, dict) else {}
    if not isinstance(eqa_cfg, dict):
        eqa_cfg = {}
    img_kw = eqa_vl_image_kwargs(eqa_cfg)
    rgb = downsample_rgb_hwc(rgb, max_side=img_kw["image_max_side"], max_pixels=img_kw["image_max_pixels"])

    if env_agent_camera_debug():
        from emet.agent.camera_debug import print_camera_frame_diagnostics

        print_camera_frame_diagnostics("describe_scene (head RGB)", rgb, force=True)

    depth = getattr(obs, "depth", None)
    if depth is not None:
        depth = np.asarray(depth)
    return rgb, depth


def _describe_scene_try_sources(
    self,
    rgb: np.ndarray,
    depth: np.ndarray | None,
    det_cfg: dict[str, Any],
    *,
    graph_memory: Any | None = None,
    memory_backend: Any | None = None,
    graph_memory_backend: Any | None = None,
) -> str | None:
    """Caption the current view first; ground with graph/map; optional detector fallback."""
    # Live caption is the primary answer for "what can you see".
    vlm_text = self._describe_scene_vlm(rgb)
    mem_text = self._describe_scene_from_memory(
        graph_memory=graph_memory,
        memory_backend=memory_backend,
        graph_memory_backend=graph_memory_backend,
    )
    parts = [p for p in (vlm_text, mem_text) if p]
    if parts:
        return " ".join(parts)

    if bool(det_cfg.get("describe_use_detector_fallback", False)):
        dm = self.detection_model
        try:
            if isinstance(dm, YoloEPerception):
                return self._describe_scene_yoloe(rgb, depth, dm)
            if isinstance(dm, OwlPerception):
                thr = float(dm.confidence_threshold) if dm.confidence_threshold is not None else 0.2
                thr = max(0.12, min(thr, 0.35))
                return self._describe_scene_owl(rgb, dm, thr)
        except Exception as e:
            if env_agent_model_debug():
                print(f"[model debug] describe_scene detector fallback failed: {e}", flush=True)
            return None
    return None


def _detection_cfg(self) -> dict[str, Any]:
    if isinstance(self.parameters, dict):
        raw = self.parameters.get("detection", {}) or {}
    elif self.parameters is not None and hasattr(self.parameters, "get"):
        raw = self.parameters.get("detection", {}) or {}
    else:
        raw = {}
    return raw if isinstance(raw, dict) else {}


def _describe_scene_vlm(self, rgb: np.ndarray) -> str | None:
    """Caption current RGB with DynaMem image_description / EQA VLM when present."""
    vm = self.get_voxel_map() if hasattr(self, "get_voxel_map") else None
    if vm is None:
        return None
    client = getattr(vm, "image_description_client", None) or getattr(vm, "eqa_client", None)
    if client is None:
        return None
    from emet.llms.vllm_factory import dynamem_vllm_call

    pil = Image.fromarray(rgb)
    prompt = (
        "Describe what is visible in this robot head-camera image in one short sentence. "
        "Name only clearly visible objects and surfaces. If the view is mostly empty floor/wall "
        "or the robot's own body, say that. Do not invent objects that are not clearly visible."
    )
    try:
        with paused_robot_streams(self.robot):
            out = dynamem_vllm_call(
                client,
                [pil, prompt],
                system_prompt="",
                max_new_tokens=64,
            )
    except Exception as e:
        if env_agent_model_debug():
            print(f"[model debug] describe_scene VLM caption failed: {e}", flush=True)
        return None
    text = (out or "").strip()
    if not text:
        return None
    if env_agent_model_debug():
        print(f"[model debug] describe_scene: VLM caption ({type(client).__name__})", flush=True)
    return f"From my head camera: {text}"


def _describe_scene_from_memory(
    self,
    *,
    graph_memory: Any | None = None,
    memory_backend: Any | None = None,
    graph_memory_backend: Any | None = None,
) -> str | None:
    """Summarize known object labels from graph / memory backends (not live detector)."""
    labels: list[str] = []
    for backend in (graph_memory_backend, memory_backend):
        if backend is not None and hasattr(backend, "list_objects"):
            try:
                labels = [str(x) for x in (backend.list_objects() or []) if str(x).strip()]
            except Exception:
                labels = []
            if labels:
                break
    if not labels and graph_memory is not None and hasattr(graph_memory, "get_nodes"):
        for n in graph_memory.get_nodes():
            if getattr(n, "is_viewpoint", False) or getattr(n, "is_frontier", False):
                continue
            for lab in getattr(n, "labels", None) or []:
                s = str(lab).strip()
                if s:
                    labels.append(s)
        labels = list(dict.fromkeys(labels))
    if not labels and hasattr(self, "get_voxel_map"):
        vm = self.get_voxel_map()
        get_inst = getattr(vm, "get_instances", None) if vm is not None else None
        if callable(get_inst):
            try:
                for inst in get_inst() or []:
                    # Prefer string category if present on instance
                    cat = getattr(inst, "category_name", None) or getattr(inst, "name", None)
                    if cat:
                        labels.append(str(cat))
            except Exception:
                pass
            labels = list(dict.fromkeys(labels))
    if not labels:
        return None
    shown = labels[:20]
    extra = f" (+{len(labels) - len(shown)} more)" if len(labels) > len(shown) else ""
    if env_agent_model_debug():
        print(
            f"[model debug] describe_scene: memory/graph labels n={len(labels)}",
            flush=True,
        )
    return f"From my map/scene graph ({len(labels)} object labels) I also know about: " + ", ".join(shown) + extra + "."


@staticmethod
def _normalize_scene_rgb_u8(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype != np.uint8:
        if out.size and float(np.nanmax(out)) <= 1.0 + 1e-6:
            out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def pick_interesting_scene_image(
    self,
    *,
    graph_memory: Any | None = None,
    live_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """Prefer a usable graph-object crop over live head RGB for Discord / chat photos.

    Returns ``(image_hwc_uint8, label_or_None)``. Label is set only for a real named
    object crop that passes the RGB usability gate; blank/white crops fall back to live RGB.
    """
    from emet.agent.camera_debug import rgb_frame_is_usable

    def _live() -> tuple[np.ndarray | None, str | None]:
        if live_rgb is None:
            return None, None
        arr = np.asarray(live_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return None, None
        out = self._normalize_scene_rgb_u8(arr.copy())
        return out, None

    gm = graph_memory if graph_memory is not None else getattr(self, "graph_memory", None)
    if gm is not None and hasattr(gm, "get_nodes") and hasattr(gm, "get_observations"):
        from emet.visualization.rerun import dynagraph_node_rgb_crop, node_has_detection_crop

        obs_rgb = {int(o.obs_id): np.asarray(o.rgb) for o in gm.get_observations()}
        # (named_bonus, support, area, label, arr)
        candidates: list[tuple[int, int, int, str, np.ndarray]] = []
        for n in gm.get_nodes():
            if getattr(n, "is_viewpoint", False) or getattr(n, "is_frontier", False):
                continue
            if not node_has_detection_crop(n, obs_rgb):
                continue
            crop = dynagraph_node_rgb_crop(n, obs_rgb)
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            arr = self._normalize_scene_rgb_u8(np.asarray(crop))
            if arr.ndim != 3 or arr.shape[2] != 3:
                continue
            if not rgb_frame_is_usable(arr):
                continue
            raw_labels = [str(x).strip() for x in (getattr(n, "labels", None) or []) if str(x).strip()]
            label = raw_labels[0] if raw_labels else ""
            if not label or label.lower() in ("object", "unknown", "none"):
                continue  # generic / empty — do not claim "closer look at object"
            score = int(getattr(n, "support_count", 1) or 1)
            area = int(arr.shape[0]) * int(arr.shape[1])
            candidates.append((1, score, area, label, arr))
        if candidates:
            candidates.sort(key=lambda t: (-t[0], -t[1], -t[2]))
            _nb, _score, _area, label, arr = candidates[0]
            return arr.copy(), label

    if hasattr(self, "get_voxel_map"):
        try:
            vm = self.get_voxel_map()
        except Exception:
            vm = None
        get_inst = getattr(vm, "get_instances", None) if vm is not None else None
        if callable(get_inst):
            best: tuple[int, str, np.ndarray] | None = None
            try:
                for inst in get_inst() or []:
                    view = getattr(inst, "get_best_view", lambda: None)()
                    crop_t = getattr(view, "cropped_image", None) if view is not None else None
                    if crop_t is None:
                        continue
                    from emet.mapping.instance.instance import _cropped_image_to_caption_input

                    arr = _cropped_image_to_caption_input(crop_t)
                    if arr is None or arr.size == 0:
                        continue
                    arr = self._normalize_scene_rgb_u8(np.asarray(arr))
                    if not rgb_frame_is_usable(arr):
                        continue
                    cat = getattr(inst, "category_name", None) or getattr(inst, "name", None) or ""
                    cat_s = str(cat).strip()
                    if not cat_s or cat_s.lower() in ("object", "unknown", "none"):
                        continue
                    area = int(arr.shape[0]) * int(arr.shape[1])
                    if best is None or area > best[0]:
                        best = (area, cat_s, arr)
            except Exception:
                best = None
            if best is not None:
                return best[2].copy(), best[1]

    return _live()


def _describe_scene_yoloe(self, rgb: np.ndarray, depth: np.ndarray | None, dm: YoloEPerception) -> str:
    """User-facing caption from YoloE — not the low-conf ScanNet dump used for mapping."""
    det_cfg: dict[str, Any] = {}
    if isinstance(self.parameters, dict):
        det_cfg = self.parameters.get("detection", {}) or {}
    elif self.parameters is not None and hasattr(self.parameters, "get"):
        raw = self.parameters.get("detection", {}) or {}
        det_cfg = raw if isinstance(raw, dict) else {}

    thr = float(det_cfg.get("describe_confidence_threshold", _DEFAULT_DESCRIBE_CONFIDENCE))
    thr = max(0.15, min(thr, 0.85))
    max_labels = int(det_cfg.get("describe_max_labels", _DEFAULT_DESCRIBE_MAX_LABELS) or _DEFAULT_DESCRIBE_MAX_LABELS)
    max_labels = max(1, min(max_labels, 30))
    use_curated = bool(det_cfg.get("describe_use_curated_vocab", True))

    old_vocab = None
    label_vocab: list[str]
    if use_curated:
        old_vocab = list(dm.class_list)
        label_vocab = list(_DESCRIBE_SCENE_YOLOE_LABELS)
        dm.class_list = label_vocab
    else:
        label_vocab = list(dm.class_list)
    try:
        # Pause ZMQ decode during GPU detect (same contention as chat LLM load).
        with paused_robot_streams(self.robot):
            _sem, _inst, task = dm.predict(
                rgb,
                depth=depth,
                draw_instance_predictions=False,
                confidence_threshold=thr,
            )
    finally:
        if old_vocab is not None:
            dm.class_list = old_vocab

    ic = task.get("instance_classes")
    scores = task.get("instance_scores")
    if ic is None or len(ic) == 0:
        return (
            "This view looks empty or unclear to me. "
            "Ask me to look around for a wider scan, or I can send a photo of what I see."
        )

    best: dict[str, float] = {}
    idxs = np.atleast_1d(np.asarray(ic)).astype(int).ravel()
    scs = (
        np.atleast_1d(np.asarray(scores, dtype=np.float64)).ravel()
        if scores is not None
        else np.ones(len(idxs), dtype=np.float64)
    )
    if len(scs) != len(idxs):
        scs = np.ones(len(idxs), dtype=np.float64)
    for idx, sc in zip(idxs, scs, strict=True):
        if float(sc) < thr:
            continue
        i = int(idx)
        if 0 <= i < len(label_vocab):
            name = label_vocab[i]
            prev = best.get(name)
            if prev is None or float(sc) > prev:
                best[name] = float(sc)
    if not best:
        return (
            "This view looks empty or unclear to me. "
            "Ask me to look around for a wider scan, or I can send a photo of what I see."
        )
    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:max_labels]
    names = [name for name, _sc in ranked]
    if names and set(names).issubset(_DESCRIBE_SCENE_STRUCTURE_LABELS):
        return (
            "Mostly empty from here — mainly " + ", ".join(names) + ". Ask me to look around if you want a wider view."
        )
    summary = ", ".join(names)
    return f"From my head camera I can make out: {summary}."


def _describe_scene_owl(self, rgb: np.ndarray, dm: OwlPerception, confidence_threshold: float) -> str:
    texts = list(_DESCRIBE_SCENE_OWL_QUERIES)
    res = dm.predict(rgb, texts, confidence_threshold=confidence_threshold)
    labels = res["labels"]
    if labels.numel() == 0:
        return "This view looks empty or unclear to me. Ask me to look around, or I can send a photo."
    scores = res["scores"]
    best_by_label: dict[int, float] = {}
    for lab, sc in zip(labels.cpu().tolist(), scores.cpu().tolist(), strict=True):
        li = int(lab)
        sf = float(sc)
        if li not in best_by_label or sf > best_by_label[li]:
            best_by_label[li] = sf
    picked = [texts[i] for i in sorted(best_by_label)]
    summary = ", ".join(picked)
    return f"From my head camera, open-vocabulary detection suggests: {summary}."
