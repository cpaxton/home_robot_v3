"""Render the selected analytic Cartesian teacher case and unchanged wheel gate."""

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path("paper/data/whole_body_control")
report = json.loads(gzip.decompress((folder / "cartesian_pose_oracle_v5.json.gz").read_bytes()))
summary = json.loads((folder / "cartesian_pose_oracle_v5.json").read_text())
chosen = next(x for x in report["snapshots"] if x["version"] == "v6-braking")["report"]["rows"][1]
actual = np.array([t["hand"][:3] for t in chosen["trace"]])
target = np.array([t["hand_goal"][:3] for t in chosen["trace"]])
fig = plt.figure(figsize=(11, 4.2), layout="constrained")
ax = fig.add_subplot(121, projection="3d")
ax.plot(*target.T, label="Announced hand pose", color="tab:orange", alpha=0.8)
ax.plot(*actual.T, label="Measured hand pose", color="tab:blue", linewidth=1)
ax.scatter(*actual[0], marker="o", color="tab:green", label="Start")
ax.scatter(*actual[-1], marker="x", color="tab:red", label="After verified placement")
ax.set(
    xlabel="world x (m)",
    ylabel="world y (m)",
    zlabel="world z (m)",
    title="Analytic Cartesian teacher / one frozen 20 g case\nDelivery passes; return planning exhausts budget",
)
ax.legend(fontsize=7)
ax = fig.add_subplot(122)
for i, metric in enumerate([1.0, 0.15]):
    ys = []
    for version in ["v5", "v6-braking"]:
        row = next(r for r in summary["rows"] if r["version"] == version and r["rotation_length_m_per_rad"] == metric)
        ys.append(max(row["contact"]["max_wheel_gap"]) * 1000)
    ax.bar(np.array([0, 1]) + (i - 0.5) * 0.3, ys, width=0.3, label=f"Rotation metric {metric} m/rad")
ax.axhline(0.1, color="tab:red", linestyle="--", linewidth=1, label="Unchanged unloading gap threshold")
ax.set(
    xticks=[0, 1],
    xticklabels=["Before quiet braking", "After quiet braking"],
    ylabel="Maximum wheel gap (mm)",
    title="Same fixture and analytic controllers\nReported safety cause remains explicit",
)
ax.legend(fontsize=7)
ax.grid(alpha=0.2)
fig.savefig("paper/figures/mars_cartesian_oracle_20261025.png", dpi=180)
fig.savefig("paper/figures/mars_cartesian_oracle_20261025.pdf")
