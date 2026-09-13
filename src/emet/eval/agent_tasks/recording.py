"""Append-only evidence with explicit policy/evaluator boundaries."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .spec import fingerprint


def write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class Recorder:
    def __init__(self, output: Path, manifest: dict):
        output.mkdir(parents=True, exist_ok=False)
        self.output = output
        self.started = time.monotonic()
        self.step = 0
        self.last_hash = ""
        manifest["manifest_hash"] = fingerprint(manifest)
        write_json(output / "manifest.json", manifest)
        (output / "images").mkdir()
        self.stream = (output / "events.jsonl").open("x")

    def append(
        self,
        kind: str,
        *,
        policy: dict | None = None,
        evaluator: dict | None = None,
        images: dict | None = None,
        command_id: str | None = None,
    ):
        from PIL import Image

        paths = {}
        image_hashes = {}
        for name, image in (images or {}).items():
            if name not in {"head", "wrist", "overhead"}:
                raise ValueError(f"unsupported image channel {name}")
            rel = f"images/{self.step:05d}_{name}.png"
            Image.fromarray(image).save(self.output / rel)
            paths[name] = rel
            image_hashes[name] = hashlib.sha256((self.output / rel).read_bytes()).hexdigest()
        event = {
            "step": self.step,
            "elapsed_s": time.monotonic() - self.started,
            "kind": kind,
            "observation_id": f"obs:{self.step}",
            "command_id": command_id,
            "policy": policy or {},
            "evaluator": evaluator or {},
            "images": paths,
            "previous_hash": self.last_hash,
            "image_hashes": image_hashes,
        }
        event["hash"] = fingerprint(event)
        self.stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.stream.flush()
        self.last_hash = event["hash"]
        self.step += 1

    def close(self):
        self.stream.close()


def load_run(output: str | Path) -> tuple[dict, list[dict], dict]:
    root = Path(output)
    manifest = json.loads((root / "manifest.json").read_text())
    expected_manifest = manifest.pop("manifest_hash", None)
    if expected_manifest is not None and expected_manifest != fingerprint(manifest):
        raise ValueError("manifest hash mismatch")
    if expected_manifest is not None:
        manifest["manifest_hash"] = expected_manifest
    events = []
    previous_hash = ""
    previous_time = -1.0
    for line in (root / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        digest = event.pop("hash")
        if event["step"] != len(events) or event["previous_hash"] != previous_hash or fingerprint(event) != digest:
            raise ValueError("event sequence/hash mismatch")
        if event["elapsed_s"] < previous_time:
            raise ValueError("event time moved backwards")
        for name, image in event["images"].items():
            resolved = (root / image).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
                raise ValueError(f"missing or unsafe recorded image: {image}")
            expected_image = event.get("image_hashes", {}).get(name)
            if expected_image is not None and hashlib.sha256(resolved.read_bytes()).hexdigest() != expected_image:
                raise ValueError(f"image hash mismatch: {image}")
        previous_time, previous_hash = event["elapsed_s"], digest
        event["hash"] = digest
        events.append(event)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {"status": "interrupted"}
    if metrics.get("event_hash") and metrics["event_hash"] != previous_hash:
        raise ValueError("metrics do not match the recorded event stream")
    return manifest, events, metrics
