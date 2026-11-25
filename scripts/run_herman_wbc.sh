#!/usr/bin/env bash
# Run from an extracted commissioning bundle on the Jetson.
set -eo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
native=${EMET_MARS_NATIVE:-"$HOME/innate-os"}
source /opt/ros/humble/setup.bash
source "$native/ros2_ws/install/local_setup.bash"
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_zenoh_cpp}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export EMET_MARS_WBC_CONFIG=${EMET_MARS_WBC_CONFIG:-"$root/commissioning.json"}
export PYTHONPATH="$root/src:$root/src/innate_mars_bridge:$root/python${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
    install-runtime)
        python3 -m pip install --only-binary=:all: --target "$root/python" 'numpy==1.26.4' 'mujoco==3.5.0'
        ;;
    build-driver)
        revision=4be109a501aa8fd5fb69d11394b878592e53789e
        if [[ $(git -C "$native" rev-parse HEAD) != "$revision" ]]; then
            echo "Native driver revision differs from the reviewed patch base" >&2
            exit 1
        fi
        scratch=$(mktemp -d "$root/native-build.XXXXXX")
        cd "$scratch"
        git -C "$native" archive "$revision" ros2_ws/src/mars_bot/mars_arm | tar -x
        git apply "$root/innate_os_mars_feedback_4be109a5.patch"
        CMAKE_BUILD_PARALLEL_LEVEL=2 colcon build --base-paths ros2_ws/src --packages-select mars_arm \
            --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release
        printf 'Patched driver overlay: %s/install/mars_arm\n' "$scratch"
        ;;
    probe)
        python3 - <<'PYTHON'
import json
import time

import rclpy

from emet.robots.innate_mars.wbc_commissioning import create_executor
from innate_mars_bridge.remote.ros import InnateMarsRosInterface
from innate_mars_bridge.remote.wbc import ARM_TOPIC, BASE_TOPIC

rclpy.init()
node = InnateMarsRosInterface()
try:
    executor = create_executor(node)
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    feedback = executor.wbc.driver.feedback()
    print(json.dumps({
        "fresh_feedback": feedback["valid"],
        "arm_healthy": feedback["arm_healthy"],
        "ready": executor.available(),
        "capabilities": executor.wbc.driver.capabilities(),
        "publishers": {
            topic: [endpoint.node_name for endpoint in node.get_publishers_info_by_topic(topic)]
            for topic in (ARM_TOPIC, BASE_TOPIC, "/cmd_vel")
        },
        "motion_commands_sent": 0,
    }, indent=2))
finally:
    node.destroy_node()
    rclpy.shutdown()
PYTHON
        ;;
    start)
        shift
        exec python3 -m innate_mars_bridge.remote.server --local \
            --wbc-config-factory emet.robots.innate_mars.wbc_commissioning:create_executor "$@"
        ;;
    *)
        echo "Usage: $0 {install-runtime|build-driver|probe|start [BRIDGE_ARGS...]}" >&2
        exit 2
        ;;
esac
