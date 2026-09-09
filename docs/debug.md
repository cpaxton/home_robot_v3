# Debug

See also: [emet CLI](cli.md) for `emet run timing`, `emet show <rrd>`, etc.

## Headless and Rerun

When running without a display (SSH, servers, Docker):

- **Rerun**: The native viewer is disabled, but the web server starts automatically. Connect from another machine at `http://<server-ip>:9090?url=ws://<server-ip>:9877`. Ports 9090 (HTTP) and 9877 (WebSocket) must be reachable.
- **SSH port forwarding** (recommended for Tailscale/VPN): Rerun binds to localhost by default. To view from your laptop over Tailscale or VPN, use SSH port forwarding:
  ```bash
  # On your laptop — forwards local ports to the robot
  ssh -L 9090:localhost:9090 -L 9877:localhost:9877 user@100.74.38.77
  ```
  Then open **http://localhost:9090?url=ws://localhost:9877** in your browser. Works over Tailscale, WireGuard, or any SSH-accessible host.
- **Direct connection** (if Rerun binds to 0.0.0.0): Try `--rerun-bind` or `RERUN_BIND_ALL=1` when starting the app. If your Rerun version supports it, you can then connect at `http://<robot-ip>:9090?url=ws://<robot-ip>:9877`. If that fails, use SSH port forwarding above.
- **socat workaround** (on the robot, before starting the app): If you have `socat` installed and `--rerun-bind` doesn't work:
  ```bash
  socat TCP-LISTEN:9090,fork,bind=0.0.0.0 TCP:127.0.0.1:9090 &
  socat TCP-LISTEN:9877,fork,bind=0.0.0.0 TCP:127.0.0.1:9877 &
  ```
  Then connect at `http://<robot-ip>:9090?url=ws://<robot-ip>:9877`.
- **Correct URL**: Open `http://localhost:9090?url=ws://localhost:9877` (or `http://<server-ip>:9090?url=ws://<server-ip>:9877` when remote). The `?url=ws://...` tells the viewer which WebSocket to connect to. Do **not** open https://app.rerun.io.
- **Native app spawning instead of web**: Native and web are exclusive — spawning the desktop viewer skips `rr.serve` (otherwise the native window stays empty). Use `--headless` or `RERUN_HEADLESS=1` for the web UI at `http://localhost:9090?url=ws://localhost:9877`. See [rerun.md](rerun.md).
- **No blueprint panel**: Use `--rerun-show-panels` when running DynaMem to reveal the entity tree and view options.
- **Still not seeing anything** (but `[RERUN] obs=True servo=True`): Data is flowing. Use the full URL: `http://localhost:9090?url=ws://localhost:9877` (the `?url=ws://...` is required to connect to the stream). Also try: (1) `--rerun-show-panels` to reveal the entity tree; (2) check the timeline is playing; (3) hard refresh (Ctrl+Shift+R).
- **winit / DISPLAY errors**: These occur when a GUI tries to spawn without a display. DynaMem and other apps now detect headless and disable the native Rerun viewer; the web server still runs.

## Apps for Debugging

- [Test Timing](#test-timing) - Test the timing of the robot's control loop over the network.
- [Camera Info](#camera-info) - Print out camera information.

#### Test Timing

Test the timing of the robot's control loop over the network. This will print out the time it takes to send a command to the robot and receive a response. It will show a histogram after a fixed number of iterations given by the `--iterations` flag (default is 500).

```bash
emet run timing --robot-ip $ROBOT_IP
# or: python -m emet.app.timing --robot_ip $ROBOT_IP

# Headless mode - no display
emet run timing --robot-ip $ROBOT_IP --headless
```

#### Camera Info

Print out information about the cameras on the robot for debugging purposes:

```bash
python -m emet.app.debug.camera_info --robot_ip $ROBOT_IP
```

This will print out information about the resolutions of different images sent by the robot's cameras, and should show something like this:

```
---------------------- Camera Info ----------------------
Servo Head RGB shape: (320, 240, 3) Servo Head Depth shape: (320, 240)
Servo EE RGB shape: (240, 320, 3) Servo EE Depth shape: (240, 320)
Observation RGB shape: (640, 480, 3) Observation Depth shape: (640, 480)
```
