#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026

"""Score a decision dataset with SigLIP/OWLv2/YoloE presence verifiers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from emet.eval.presence_verifiers import (
    OwlV2PresenceDetector,
    YoloEPresenceDetector,
    detector_crop_evidence,
    siglip_cosine,
)
from emet.eval.verifier_bakeoff import summarize_backend


def _load_rows(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("action_taken") != "verify_siglip" or not row.get("rgb_path"):
            continue
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _dense_max(encoder: Any, rgb: np.ndarray, phrase: str) -> float:
    import torch
    import torch.nn.functional as F

    text = encoder.encode_text(phrase).detach().float().reshape(-1)
    text = text / (text.norm() + 1e-12)
    inputs = encoder._to_model_inputs(encoder.processor(images=rgb, return_tensors="pt"))
    with torch.no_grad():
        output = encoder.model.vision_model(inputs["pixel_values"], output_hidden_states=True)
        features = F.normalize(output.last_hidden_state.float(), dim=-1)
        sims = features @ text.to(device=features.device, dtype=features.dtype).reshape(-1, 1)
    return float(sims.max().item())


def _label(row: dict[str, Any]) -> bool | None:
    gt = row.get("gt") or {}
    return gt.get("gt_in_view")


def _score_siglip(
    rows: list[dict[str, Any]],
    *,
    version: str,
    device: str,
    dtype: str,
) -> list[dict[str, Any]]:
    import torch

    from emet.perception.encoders.siglip_encoder import MaskSiglipEncoder

    encoder = MaskSiglipEncoder(version=version, device=device, dtype=dtype)
    output: list[dict[str, Any]] = []
    for row in rows:
        rgb = _load_rgb(row["rgb_path"])
        phrase = str(row.get("phrase") or "")
        t0 = time.perf_counter()
        full = siglip_cosine(encoder, rgb, phrase)
        dense = _dense_max(encoder, rgb, phrase)
        output.append(
            {
                "episode_id": row["episode_id"],
                "step_id": row["step_id"],
                "phrase": phrase,
                "gt_in_view": _label(row),
                "full_score": full,
                "dense_score": dense,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }
        )
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _score_detector(
    rows: list[dict[str, Any]],
    *,
    backend: str,
    device: str,
    crop_encoder: Any | None,
) -> list[dict[str, Any]]:
    import torch

    detector = (
        OwlV2PresenceDetector(device=device)
        if backend == "owlv2"
        else YoloEPresenceDetector(device=device)
    )
    output: list[dict[str, Any]] = []
    for row in rows:
        rgb = _load_rgb(row["rgb_path"])
        phrase = str(row.get("phrase") or "")
        evidence = (
            detector_crop_evidence(detector, crop_encoder, rgb, phrase)
            if crop_encoder is not None
            else detector.score(rgb, phrase)
        )
        output.append(
            {
                "episode_id": row["episode_id"],
                "step_id": row["step_id"],
                "phrase": phrase,
                "gt_in_view": _label(row),
                "detector_score": evidence.score,
                "crop_siglip_score": evidence.crop_siglip_sim,
                "bbox_xyxy": evidence.bbox_xyxy,
                "latency_ms": evidence.latency_ms,
            }
        )
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-o", "--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        default="siglip1,siglip2,owlv2,yoloe",
        help="Comma-separated: siglip1,siglip2,owlv2,yoloe",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--detector-crop-siglip", action="store_true")
    args = parser.parse_args()

    rows = _load_rows(args.dataset.expanduser(), args.max_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    reports: dict[str, Any] = {}
    crop_encoder = None
    if args.detector_crop_siglip and any(method in ("owlv2", "yoloe") for method in methods):
        from emet.perception.encoders.siglip_encoder import MaskSiglipEncoder

        crop_encoder = MaskSiglipEncoder(version="so400m", device=args.device, dtype=args.dtype)

    for method in methods:
        print(f"scoring {method}: n={len(rows)}", flush=True)
        if method in ("siglip1", "siglip2"):
            scored = _score_siglip(
                rows,
                version="so400m" if method == "siglip1" else "siglip2_so400m",
                device=args.device,
                dtype=args.dtype,
            )
            score_key = "dense_score"
            thresholds = [round(0.08 + i * 0.005, 3) for i in range(17)]
        elif method in ("owlv2", "yoloe"):
            scored = _score_detector(
                rows,
                backend=method,
                device=args.device,
                crop_encoder=crop_encoder,
            )
            score_key = "detector_score"
            thresholds = [round(i * 0.025, 3) for i in range(1, 21)]
        else:
            raise ValueError(f"unknown method {method}")
        scored_path = args.output_dir / f"{method}.jsonl"
        scored_path.write_text(
            "".join(json.dumps(row, default=str) + "\n" for row in scored),
            encoding="utf-8",
        )
        reports[method] = summarize_backend(
            scored,
            score_key=score_key,
            thresholds=thresholds,
        )
        reports[method]["rows_path"] = str(scored_path)

    report = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "n_rows": len(rows),
        "n_labeled": sum(_label(row) is not None for row in rows),
        "methods": reports,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
