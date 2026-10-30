"""Reproduce the matched hold-dynamics development figure from committed data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("paper/data/whole_body_control")
d = json.loads((root / "hold_dynamics_v3.json").read_text())
v = json.loads((root / "hold_cartesian_validation_v3.json").read_text())
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
colors = ["tab:blue", "tab:orange"]
for case, color in zip(["left:20g:1", "right:20g:2"], colors, strict=True):
    rows = [r for r in d["gain_ablation"] if r["case_id"] == case]
    gains = [r["wheel_feedback_gain_override"] for r in rows]
    dwells = [r["quiet_metrics"]["longest_quiet_pose_dwell_s"] for r in rows]
    base = next(r for r in d["baselines"] if r["case_id"] == case and r["selector"] == "checkpoint_policy")
    axs[0].plot(
        [*gains, 2], [*dwells, base["quiet_metrics"]["longest_quiet_pose_dwell_s"]], ".-", label=case, color=color
    )
    values = []
    for selector in ["zero", "explicit_hold"]:
        r = next(r for r in d["baselines"] if r["case_id"] == case and r["selector"] == selector)
        values.append(r["quiet_metrics"]["base_velocity_rms"][0] * 1000)
    axs[1].plot([0, 1], values, ".-", label=case, color=color)
axs[0].axhline(2, linestyle="--", color="tab:red", linewidth=1)
axs[0].set(
    xlabel="Extra wheel feedback gain",
    ylabel="Longest continuous quiet pose (s)",
    title="Fixed-weight gain ablation\nGain 2: 10 s; others: 6 s",
)
axs[0].legend(fontsize=8)
axs[1].set(
    xticks=[0, 1],
    xticklabels=["Zero request", "Explicit hold"],
    ylabel="Forward velocity RMS (mm/s)",
    title="Identical initial states / 10 s\nNative actuators remain active",
)
rows = [r for r in v["rows"] if r["kind"] == "hold" and r["selector"] == "checkpoint_policy"]
y = [r["quiet_metrics"]["longest_quiet_pose_dwell_s"] for r in rows]
axs[2].bar(np.arange(len(rows)), y, color="tab:blue")
for i, r in enumerate(rows):
    axs[2].text(i, y[i] + 0.12, f"{r['duration_requested_s']} s", ha="center", fontsize=8)
axs[2].axhline(2, linestyle="--", color="tab:red", linewidth=1)
axs[2].set(
    xlabel="Development fixture",
    ylabel="Longest continuous quiet pose (s)",
    title="Refined learned policy / gain 0.5\n5/5 stationary holds pass",
    ylim=(0, 11),
    xticks=range(5),
    xticklabels=range(1, 6),
)
for ax in axs:
    ax.grid(alpha=0.2)
fig.savefig("paper/figures/mars_hold_dynamics_20261025.png", dpi=180)
fig.savefig("paper/figures/mars_hold_dynamics_20261025.pdf")
