# PR #135 design review disposition

1. **Runtime duplication:** `src/emet` is canonical for `zmq_obs_codec`,
   `zmq_server_env`, and `compression`. Run `python scripts/sync_zmq_runtime.py`
   after changes; `--check` and `test_zmq_runtime_parity.py` reject divergence.
   Deployable files remain real copies because robot installs use `emet_core`
   without the full workstation package. `h264_zmq.py` currently exists only
   in `emet_core`; it is not a second divergent copy.
2. **DINOv3 hot loop — unresolved:** throttling reduces frequency but not
   publish stalls. Before advertising low-jitter onboard embeddings, move
   loading/inference to a single worker with one replaceable pending frame,
   copy the submitted pixels and timing together, publish embedding source
   timing separately, and close the worker on shutdown. Test nonblocking
   publication with a deliberately blocked encoder and bounded queue size.
   Leave onboard DINOv3 disabled for stream latency acceptance meanwhile.
   Onboard depth also needs a separate latency/provenance audit.
3. **RTSP GUI:** preview now runs on the main thread of a spawned child
   process, with termination/join when the parent stream exits. No OpenCV
   GUI calls from a daemon thread. Display-system testing remains pending.
4. **Experimental H.264:** added optional `video`/PyAV dependency and disable
   the channel when PyAV is unavailable or encoding fails. This remains an
   independent-frame encoder, not a persistent inter-frame video encoder;
   all frames are keyframes and cadence still follows the servo period.
   Do not use it to make bandwidth/latency claims about production H.264.
5. **Lidar geometry — unresolved:** current uniform angular ordering only
   supports the existing replicated MJCF. Follow-up must project each hit
   using its sensor site's origin and +Z direction, transforming both into
   an explicitly named common lidar/base frame. Test reordered sensors,
   nonzero starting angles and translated sites; deriving only angles would
   still miss translated origins. Do not claim arbitrary MJCF support.
6. **Environment parsing:** invalid/nonfinite rates and scaling warn and
   fall back. JPEG quality and port parsing share validation; invalid ports
   fall back rather than reaching socket bind with invalid numbers.
7. **Deployment default:** corrected DINOv3 remote checkout to
   `~/src/home_robot_v3`; `EMET_CALIBAN_REPO` remains the explicit override.

These changes do not constitute stationary hardware or sustained Jetson
latency acceptance. No deployment or robot motion is part of this review fix.
