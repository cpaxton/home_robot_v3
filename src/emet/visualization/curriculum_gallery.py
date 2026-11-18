"""Publish only catalogued curriculum media; raw physics/assets stay private."""

import json
from pathlib import Path

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.visualization.curriculum_cases import FAMILIES
from emet.visualization.curriculum_record import sha

TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".html": "text/html; charset=utf-8",
}


def publish(directory):
    directory = Path(directory)
    catalog = {
        "format": "curriculum-gallery-v1",
        "controller": "analytic_teacher",
        "hardware_used": False,
        "videos": [],
        "coverage_available": False,
        "assets": {},
        "expected_videos": len(FAMILIES),
    }
    audit_path = directory / "trajectory-spread.json"
    motion = json.loads(audit_path.read_text())["families"] if audit_path.exists() else {}
    for family in FAMILIES:
        path = directory / (family + "-video.json")
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        if row.get("format") != "curriculum-video-v1" or row["family"] != family:
            raise ValueError("invalid video receipt")
        for name, digest in row["assets"].items():
            if Path(name).name != name or Path(name).suffix not in TYPES or sha(directory / name) != digest:
                raise ValueError("unbound curriculum media")
            catalog["assets"][name] = digest
        item = {k: row[k] for k in ("family", "title", "controller", "duration_s", "resolution", "summary")}
        if family in motion:
            item["motion_summary"] = motion[family]
        catalog["videos"].append(item)
    if (directory / "coverage.json").exists():
        coverage = json.loads((directory / "coverage.json").read_text())
        catalog.update(coverage_available=True, coverage_summary=coverage["summary"])
        for name in ("coverage.html", "coverage.png", "coverage.svg"):
            catalog["assets"][name] = sha(directory / name)
    save(directory / "index.json", catalog)
    return catalog


class CurriculumAssets:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()

    def state(self):
        path = self.directory / "index.json"
        if not path.exists():
            return {"format": "curriculum-gallery-v1", "videos": [], "coverage_available": False, "phase": "rendering"}
        value = json.loads(path.read_text())
        if value.get("format") != "curriculum-gallery-v1":
            raise ValueError("invalid curriculum catalog")
        return {k: v for k, v in value.items() if k != "assets"}

    def asset(self, name):
        if not name or Path(name).name != name or Path(name).suffix not in TYPES:
            raise ValueError("unknown curriculum asset")
        catalog = json.loads((self.directory / "index.json").read_text())
        if name not in catalog.get("assets", {}):
            raise ValueError("unknown curriculum asset")
        path = self.directory / name
        if path.is_symlink() or not path.is_file() or sha(path) != catalog["assets"][name]:
            raise ValueError("curriculum media changed")
        return path, TYPES[path.suffix]
