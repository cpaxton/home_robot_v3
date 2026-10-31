"""Resolve generated control evidence stored outside Git, with checksum checks."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PREFIX = Path("paper/data/whole_body_control")


def catalog(root=ROOT):
    path = Path(root) / PREFIX / "artifacts.json"
    rows = json.loads(path.read_text())["artifacts"] if path.exists() else []
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.is_relative_to(PREFIX):
            raise ValueError("Artifact path escapes declared generated-data root")
    return rows


def checked_path(path, entry):
    path = Path(path)
    if path.stat().st_size != entry["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"Control artifact checksum mismatch: {entry['path']}")
    return path


def resolve_artifact(path, *, root=ROOT, store=None):
    path, root = Path(path), Path(root)
    full = path if path.is_absolute() else root / path
    try:
        relative = full.relative_to(root)
    except ValueError:
        relative = None
    entry = next((r for r in catalog(root) if Path(r["path"]) == relative), None)
    if full.is_file():
        return checked_path(full, entry) if entry else full
    if entry:
        storage = Path(store) if store else root / "logs/artifacts/whole_body_control"
        target = storage / relative.relative_to(PREFIX)
        if target.is_file():
            return checked_path(target, entry)
        raise FileNotFoundError(
            f"Generated artifact is outside Git: {relative}. "
            "Restore with python scripts/control_artifacts.py restore --store /path/to/control-artifacts"
        )
    raise FileNotFoundError(full)
