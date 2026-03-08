# Debug

## Headless and Rerun

When running without a display (SSH, servers, Docker):

- **Rerun**: The native viewer is disabled, but the web server starts automatically. Connect from another machine at `http://<server-ip>:9090` to view the visualization. Ports 9090 (HTTP) and 9877 (WebSocket) must be reachable.
- **SSH port forwarding**: If the server binds to localhost only, use `ssh -L 9090:localhost:9090 -L 9877:localhost:9877 user@server`, then open `http://localhost:9090` on your laptop.
- **Correct URL**: Open `http://<server-ip>:9090` (or `http://localhost:9090` when local)—the same machine where DynaMem is running. Do **not** open https://app.rerun.io; that is a different app. The viewer at :9090 auto-connects to the stream; there is no Connect button.
- **Native app spawning instead of web**: If the native Rerun viewer opens instead of the web viewer, use `--headless` or set `RERUN_HEADLESS=1`. The web server now always runs on :9090, so you can open that URL even when the native app spawns.
- **No blueprint panel**: Use `--rerun-show-panels` when running DynaMem to reveal the entity tree and view options.
- **Still not seeing anything**: Run `uv run python scripts/test_rerun.py` to verify Rerun works in isolation. Use `--rerun-debug` with DynaMem to print obs/servo status every 2s (confirms data is reaching Rerun). Ensure the MuJoCo server is running in another terminal before starting DynaMem.
- **winit / DISPLAY errors**: These occur when a GUI tries to spawn without a display. DynaMem and other apps now detect headless and disable the native Rerun viewer; the web server still runs.

## Apps for Debugging

- [Test Timing](#test-timing) - Test the timing of the robot's control loop over the network.
- [Camera Info](#camera-info) - Print out camera information.

#### Test Timing

Test the timing of the robot's control loop over the network. This will print out the time it takes to send a command to the robot and receive a response. It will show a histogram after a fixed number of iterations given by the `--iterations` flag (default is 500).

```bash
python -m stretch.app.timing --robot_ip $ROBOT_IP

# Headless mode - no display
python -m stretch.app.timing --headless

# Set the number of iterations per histogram to 1000
python -m stretch.app.timing --iterations 1000
```

#### Camera Info

Print out information about the cameras on the robot for debugging purposes:

```bash
python -m stretch.app.debug.camera_info --robot_ip $ROBOT_IP
```

This will print out information about the resolutions of different images sent by the robot's cameras, and should show something like this:

```
---------------------- Camera Info ----------------------
Servo Head RGB shape: (320, 240, 3) Servo Head Depth shape: (320, 240)
Servo EE RGB shape: (240, 320, 3) Servo EE Depth shape: (240, 320)
Observation RGB shape: (640, 480, 3) Observation Depth shape: (640, 480)
```
