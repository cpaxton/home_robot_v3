import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path("paper/data/whole_body_control")
rows = json.loads((out / "hold_feedback_validation_v2.json").read_text())["rows"]
old = json.loads((out / "hold_repair_pilot_v1.json").read_text())["trained_results"]
fig, axs = plt.subplots(1, 4, figsize=(14, 3.6), layout="constrained")
ax = axs[0]
a = [x["hand_error_at_end_m"] * 1000 for x in old if x["initialization"] == "scratch" and x["kind"] == "hold"]
b = [x["max_hand_error_m"] * 1000 for x in rows if x["kind"] == "hold10"]
ax.scatter([0] * len(a), a, label="Velocity head, 2 s", color="tab:orange")
ax.scatter([1] * len(b), b, label="Feedback head, 10 s", color="tab:blue")
ax.set_xticks([0, 1], ["Velocity\n2 s (n=2)", "Feedback\n10 s (n=5)"])
ax.set_ylabel("Hand error (mm)")
ax.set_title("Stationary position\nDifferent durations / fixtures")
ax = axs[1]
for i, kind in enumerate(["near3", "near6"]):
    y = [x["hand_error_at_end_m"] * 1000 for x in rows if x["kind"] == kind]
    ax.scatter(np.full(len(y), i), y, color="tab:blue")
ax.axhline(5, color="tab:red", linestyle="--", linewidth=1)
ax.set_xticks([0, 1], ["3 s", "6 s"])
ax.set_ylabel("Endpoint position error (mm)")
ax.set_title("Nearby FK targets\nSame five fixtures")
ax = axs[2]
y = [x["orientation_error_at_end_rad"] for x in rows if x["kind"] == "rotation6"]
ax.scatter(range(1, 6), y, color="tab:blue")
ax.axhline(0.05, color="tab:red", linestyle="--", linewidth=1)
ax.set_ylim(0, 0.14)
ax.set_ylabel("Endpoint angle error (rad)")
ax.set_xlabel("Development fixture")
ax.set_title("Reachable 0.12 rad rotation\nOne deadline stop retained")
ax = axs[3]
y = [x["quiet_dwell"]["longest_quiet_pose_dwell_s"] for x in rows if x["kind"] == "hold10"]
ax.scatter(range(1, 6), y, color="tab:blue")
ax.axhline(2, color="tab:red", linestyle="--", linewidth=1)
ax.set_ylim(0, 2.2)
ax.set_ylabel("Longest quiet pose dwell (s)")
ax.set_xlabel("Development fixture")
ax.set_title("Continuous whole-body quiet\n0/5 holds reach 2 s")
for ax in axs:
    ax.grid(alpha=0.2)
fig.savefig("paper/figures/mars_hold_feedback_20261025.png", dpi=180)
fig.savefig("paper/figures/mars_hold_feedback_20261025.pdf")
