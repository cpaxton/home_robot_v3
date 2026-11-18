"""Export sampled SE(3)/SE(2) target coverage, including rejected identities."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.rl.manipulation.arm_splines import ArmSplineTrackingEnv, load_manifest
from emet.visualization.curriculum_cases import FAMILIES, ROOT
from emet.visualization.curriculum_record import sha
from emet.visualization.curriculum_video import pose_rotation


def coverage_statistics(rows):
    admitted = [r for r in rows if r.get("hand")]
    active = [r for r in admitted if r.get("hand_active", True)]
    result = {
        "requested_paths": len(rows),
        "admitted_paths": len(admitted),
        "rejected_paths": len(rows) - len(admitted),
        "active_hand_paths": len(active),
        "passive_hand_sweeps": len(admitted) - len(active),
        "postures": dict(Counter(r["case"].get("posture", "other") for r in admitted)),
        "splits": dict(Counter(r["case"]["split"] for r in admitted)),
    }
    if active:
        points = np.concatenate([np.asarray(r["hand"])[:, :3] for r in active])
        rotations = np.concatenate([np.asarray([pose_rotation(p) for p in r["hand"]]) for r in active])
        relative = Rotation.from_matrix(rotations @ rotations[0].T).magnitude()
        result.update(
            position_min_m=points.min(0).tolist(),
            position_max_m=points.max(0).tolist(),
            position_span_m=np.ptp(points, axis=0).tolist(),
            occupied_2cm_voxels=int(len(np.unique(np.floor(points / 0.02).astype(int), axis=0))),
            orientation_from_first_max_deg=float(np.rad2deg(relative.max())),
        )
    result["interpretation"] = "Sample occupancy only; no reachable-workspace percentage or uniform coverage claim."
    return result


def export(output, *, full_splines=True):
    output = Path(output)
    rows = {}
    for family in FAMILIES:
        path = output / family / "report.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        for row in report["rows"]:
            if not row.get("recording"):
                continue
            asset = path.parent / row["recording"]
            if sha(asset) != row["assets"][asset.name]:
                raise ValueError("recording changed before coverage export")
            with np.load(asset, allow_pickle=False) as data:
                hand = data["planned_hand"] if "planned_hand" in data else data["hand_target"]
                base = data["planned_base"] if "planned_base" in data else data["base_target"]
                indices = np.unique(np.r_[np.arange(0, len(hand), max(1, len(hand) // 180)), len(hand) - 1])
                item = {
                    "family": family,
                    "hand_active": family != "base",
                    "case": row["case"],
                    "hand": hand[indices].tolist(),
                    "base": base[indices].tolist(),
                    "scope": "full announced reference" if "planned_hand" in data else "executed TAMP goals only",
                    "teacher_success": row["success"],
                    "initial_joints_rad": data["joints"][0].tolist(),
                }
            rows[row["case"]["id"]] = item
    if full_splines:
        spec = load_manifest(ROOT / "configs/benchmarks/mars_arm_splines_v1.json")
        env = ArmSplineTrackingEnv(spec["cases"])
        try:
            for case in spec["cases"]:
                item = {
                    "family": f"arm_splines_l{case['level']}",
                    "hand_active": True,
                    "case": case,
                    "scope": "full admitted reference; no rollout implied",
                }
                try:
                    _, info = env.reset(options={"case_id": case["id"]})
                    reference = env.reference
                    indices = np.unique(
                        np.r_[
                            np.arange(0, len(reference.times), max(1, len(reference.times) // 180)),
                            len(reference.times) - 1,
                        ]
                    )
                    item.update(
                        hand=reference.hand[indices].tolist(),
                        base=reference.base[indices].tolist(),
                        admission=info["spline_admission"],
                        rejections=info["spline_rejections"],
                        initial_joints_rad=env.unwrapped.control.positions().tolist(),
                        certificate=reference.certificate,
                        reference_sha256=info["reference_sha256"],
                    )
                except ValueError as exc:
                    item.update(rejected=True, reason=str(exc), rejections=env.rejections.copy())
                if case["id"] in rows:
                    item["teacher_success"] = rows[case["id"]].get("teacher_success")
                rows[case["id"]] = item
                print("COVERAGE", case["id"], not item.get("rejected", False), flush=True)
        finally:
            env.close()
    items = list(rows.values())
    report = {
        "format": "curriculum-coverage-v1",
        "rows": items,
        "hardware_used": False,
        "summary": coverage_statistics(items),
        "families": {f: coverage_statistics([r for r in items if r["family"] == f]) for f in FAMILIES},
    }
    save(output / "coverage.json", report)
    template = Path(__file__).with_suffix(".html").read_text()
    (output / "coverage.html").write_text(template.replace("__DATA__", json.dumps(report).replace("<", "\\u003c")))
    plot(report, output)
    return report


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure = plt.figure(figsize=(16, 9))
    xyz = figure.add_subplot(121, projection="3d")
    xy = figure.add_subplot(222)
    xz = figure.add_subplot(224)
    palette = plt.get_cmap("tab10")
    for row in report["rows"]:
        if not row.get("hand"):
            continue
        hand = np.asarray(row["hand"])
        base = np.asarray(row["base"])
        color = palette(FAMILIES.index(row["family"]))
        xy.plot(*base[:, :2].T, color=color, alpha=0.2, ls=":")
        if not row.get("hand_active", True):
            continue
        xyz.plot(*hand[:, :3].T, color=color, alpha=0.35, lw=0.7)
        xy.plot(*hand[:, :2].T, color=color, alpha=0.25, lw=0.6)
        xz.plot(hand[:, 0], hand[:, 2], color=color, alpha=0.25, lw=0.6)
        for i in np.linspace(0, len(hand) - 1, 3, dtype=int):
            rot = pose_rotation(hand[i])
            xyz.quiver(*hand[i, :3], *(rot[:, 0] * 0.015), color=color, alpha=0.4, linewidth=0.6)
    xyz.set(
        xlabel="World x (m)",
        ylabel="World y (m)",
        zlabel="World z (m)",
        title="Full hand target paths + sparse orientation arrows",
    )
    for ax, title in [(xy, "XY: hand paths; dotted base goals"), (xz, "XZ: hand paths")]:
        ax.set_title(title)
        ax.set_xlabel("World x (m)")
        ax.set_ylabel("World y (m)" if ax is xy else "World z (m)")
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")
    figure.legend(
        [Line2D([0], [0], color=palette(i)) for i in range(len(FAMILIES))], FAMILIES, loc="lower center", ncol=4
    )
    s = report["summary"]
    figure.suptitle(
        f"Mars curriculum coverage | {s['admitted_paths']}/{s['requested_paths']} admitted paths | sampled coverage, not all reachable poses"
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    figure.savefig(output / "coverage.png", dpi=160)
    figure.savefig(output / "coverage.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recordings-only", action="store_true")
    args = parser.parse_args()
    export(args.output, full_splines=not args.recordings_only)


if __name__ == "__main__":
    main()
