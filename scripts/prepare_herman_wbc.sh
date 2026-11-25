#!/usr/bin/env bash
# Build a relocatable commissioning bundle without reaching the robot.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: $0 ROBOT_DESCRIPTION_URDF ACTOR_DIRECTORY OUTPUT_TGZ" >&2
    exit 2
fi
description=$(realpath "$1")
actor=$(realpath "$2")
output=$(realpath -m "$3")
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/actor"
cp "$description" "$scratch/robot_description.urdf"
cp "$actor"/{manifest.json,policy.zip} "$scratch/actor/"
cp "$root/docs/patches/innate_os_mars_feedback_4be109a5.patch" "$scratch/"
cp "$root/scripts/run_herman_wbc.sh" "$scratch/"
printf '%s\n' '{"robot_description":"robot_description.urdf"}' > "$scratch/commissioning.json"
tar -C "$root" --exclude=__pycache__ --exclude='*.pyc' --exclude=src/emet/assets -cf - src/emet |
    tar -C "$scratch" -xf -
tar -C "$root" --exclude=__pycache__ --exclude='*.pyc' -cf - \
    src/emet/assets/robot/innate_mars src/innate_mars_bridge/innate_mars_bridge |
    tar -C "$scratch" -xf -
(
    set -o noclobber
    tar -C "$scratch" -czf - . > "$output"
)
