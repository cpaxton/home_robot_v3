# emet-core

Core runtime for Emet: ZMQ server (BaseZmqServer), motion (kinematics, control, pinocchio), and a torch-free subset of utils. Use this on the **robot** or for **simulator bridges** so you only need lightweight dependencies (no torch, transformers, or mujoco).

- **Robot bridges** depend on emet-core: **Stretch** uses `stretch_ros2_bridge`, **Innate Mars** uses `innate_mars_bridge`.
- **Simulator bridges** subclass `emet.core.server.BaseZmqServer` and implement the same message contract; depend only on emet-core + the sim SDK.

Install: `pip install emet-core`
