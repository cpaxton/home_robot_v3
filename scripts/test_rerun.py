#!/usr/bin/env python3
"""Minimal Rerun test - verify Rerun web viewer works without DynaMem/simulation.

Run: uv run python scripts/test_rerun.py
Then open http://localhost:9090 (or http://<this-host>:9090 from another machine).

You should see a test image and 3D points. If you see the landing page, click Connect.
"""
import os
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# Headless: no native viewer, serve web only
HEADLESS = not bool(
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET")
)

rr.init("test_rerun", spawn=not HEADLESS)
if HEADLESS:
    rr.serve(open_browser=False)
    print("Rerun web server: open http://<this-host>:9090?url=ws://<this-host>:9877")
else:
    rr.serve(open_browser=True)

# Log test data
rr.log("test/image", rr.Image(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)))
rr.log("test/points", rr.Points3D(positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial2DView(name="Test Image", origin="/test/image"),
        rrb.Spatial3DView(name="Test 3D", origin="test"),
    ),
    collapse_panels=False,
)
rr.send_blueprint(blueprint)

print("Streaming test data for 60s...")
for i in range(60):
    rr.set_time_seconds("realtime", time.time())
    rr.log("test/image", rr.Image(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)))
    time.sleep(1)

print("Done.")
