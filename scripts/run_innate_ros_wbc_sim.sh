#!/usr/bin/env bash
# Build only the ROS message needed by the headless Innate WBC smoke.
set -eo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: $0 INNATE_OS_CHECKOUT ROBOT_DESCRIPTION_URDF" >&2
    exit 2
fi
innate=$(realpath "$1")
description=$(realpath "$2")
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
source /opt/ros/humble/setup.bash
mkdir -p "$scratch/src/msg"
cp "$innate/ros2_ws/src/maurice_bot/maurice_msgs/msg/ArmStatus.msg" "$scratch/src/msg/"
cat > "$scratch/src/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.8)
project(mars_msgs)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME} "msg/ArmStatus.msg")
ament_package()
CMAKE
cat > "$scratch/src/package.xml" <<'XML'
<package format="3">
  <name>mars_msgs</name>
  <version>0.0.0</version>
  <description>Arm status interface for the isolated Innate simulation smoke</description>
  <maintainer email="sim@example.org">Simulation</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
</package>
XML
prefix="$scratch/install"
if ! {
    cmake -S "$scratch/src" -B "$scratch/build" -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 &&
    cmake --build "$scratch/build" -j2 &&
    cmake --install "$scratch/build"
} > "$scratch/build.log" 2>&1; then
    cat "$scratch/build.log" >&2
    exit 1
fi
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID="${EMET_SIM_ROS_DOMAIN:-87}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export ROS_LOG_DIR="$scratch/log"
export EMET_MARS_WBC_CONFIG="$scratch/sim-commissioning.json"
export PYTHONPATH="$root/src:$root/src/innate_mars_bridge:$prefix/local/lib/python3.10/dist-packages:$prefix/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python_bin=${EMET_SIM_PYTHON:-"$root/.venv/bin/python"}
"$python_bin" - "$description" <<'PYTHON'
import json
import os
import sys
from pathlib import Path

from emet.robots.innate_mars.wbc_commissioning import VERIFICATION_KEYS

record = {"robot_description": sys.argv[1], **dict.fromkeys(VERIFICATION_KEYS, True)}
Path(os.environ["EMET_MARS_WBC_CONFIG"]).write_text(json.dumps(record))
PYTHON
"$python_bin" "$root/scripts/smoke_innate_ros_wbc.py" --innate "$innate"
