"""Validate the proposed innate-os feedback patch entirely in a temporary tree.

Checks audited file hashes, applies the patch to copies, then compiles the real
patched Robot reader against an SDK test double. No ROS node, serial device,
upstream write, hardware connection or deployment occurs.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "docs/patches/innate_os_feedback_units.patch"
META = PATCH.with_suffix(".meta.json")

SDK = r"""
#pragma once
#include <cstdint>
#include <stdexcept>
#define COMM_SUCCESS 0
#define DXL_LOBYTE(x) (static_cast<std::uint8_t>((x) & 0xff))
#define DXL_HIBYTE(x) (static_cast<std::uint8_t>(((x) >> 8) & 0xff))
#define DXL_LOWORD(x) ((x) & 0xffff)
#define DXL_HIWORD(x) (((x) >> 16) & 0xffff)
namespace dynamixel {
class GroupSyncRead {
public:
    inline static int result = 0;
    inline static bool available = true;
    inline static int data_reads = 0;
    GroupSyncRead(void*, void*, int, int) {}
    bool addParam(int) { return true; }
    int txRxPacket() { return result; }
    bool isAvailable(int, int, int) { return available; }
    std::uint32_t getData(int, int address, int) {
        ++data_reads;
        if (address == 128) return static_cast<std::uint32_t>(-100);
        if (address == 132) return 2048;
        return 5;
    }
};
class GroupSyncWrite {
public:
    GroupSyncWrite(void*, void*, int, int) {}
    bool addParam(int, std::uint8_t*) { return true; }
    bool changeParam(int, std::uint8_t*) { return true; }
    int txPacket() { return 0; }
};
}
"""

DYNAMIXEL = r"""
#pragma once
#include "dynamixel_sdk/dynamixel_sdk.h"
namespace maurice_arm {
class Dynamixel {
public:
    void* portHandler() { return nullptr; }
    void* packetHandler() { return nullptr; }
    void reboot(int) {}
};
}
"""

CHECKS = r"""
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include "maurice_arm/robot.hpp"
#include "maurice_arm/feedback_units.hpp"
using dynamixel::GroupSyncRead;
using maurice_arm::dynamixelVelocityRadPerSecond;
void must_throw(const std::function<void()>& read) {
    const int before = GroupSyncRead::data_reads;
    bool threw = false;
    try { read(); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
    assert(GroupSyncRead::data_reads == before);
}
int main() {
    // Manufacturer register units: 100 counts = 22.9 rpm = 2.398... rad/s.
    assert(dynamixelVelocityRadPerSecond(0) == 0);
    assert(std::abs(dynamixelVelocityRadPerSecond(100) - 2.398082392240209) < 1e-12);
    assert(std::abs(dynamixelVelocityRadPerSecond(-100) + 2.398082392240209) < 1e-12);
    maurice_arm::Robot robot(std::make_shared<maurice_arm::Dynamixel>(), {1, 2});
    auto [q, dq, load] = robot.readState(0);
    assert(q.size() == 2 && q[0] == 2048 && dq[0] == -100 && load[0] == 5);
    // Failed packets must not fabricate a fresh, stationary all-zero sample.
    GroupSyncRead::result = -1;
    must_throw([&] { robot.readState(0); });
    must_throw([&] { robot.readPosition(0); });
    must_throw([&] { robot.readVelocity(); });
    // A nominally successful packet with missing servo data is invalid too.
    GroupSyncRead::result = 0;
    GroupSyncRead::available = false;
    must_throw([&] { robot.readState(0); });
    must_throw([&] { robot.readPosition(0); });
    must_throw([&] { robot.readVelocity(); });
    std::cout << "conversion, valid-state, communication-failure and missing-data checks passed\n";
}
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(upstream):
    metadata = json.loads(META.read_text())
    if digest(PATCH) != metadata["patch_sha256"]:
        raise RuntimeError("feedback patch differs from its audited manifest")
    for name, expected in metadata["before_sha256"].items():
        if digest(upstream / name) != expected:
            raise RuntimeError(f"audited upstream source differs: {name}")
    with tempfile.TemporaryDirectory(prefix="mars-feedback-patch-") as directory:
        scratch = Path(directory)
        for name in metadata["before_sha256"]:
            target = scratch / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(upstream / name, target)
        subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=scratch, check=True)
        subprocess.run(["git", "apply", str(PATCH)], cwd=scratch, check=True)
        include = scratch / "stub"
        (include / "dynamixel_sdk").mkdir(parents=True)
        (include / "maurice_arm").mkdir()
        (include / "dynamixel_sdk/dynamixel_sdk.h").write_text(SDK)
        (include / "maurice_arm/dynamixel.hpp").write_text(DYNAMIXEL)
        test = scratch / "checks.cpp"
        test.write_text(CHECKS)
        package = scratch / metadata["package"]
        binary = scratch / "checks"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-I",
                str(include),
                "-I",
                str(package / "include"),
                str(test),
                str(package / "maurice_arm/robot.cpp"),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        print(result.stdout.strip())
    for name, expected in metadata["before_sha256"].items():
        if digest(upstream / name) != expected:
            raise RuntimeError(f"upstream source changed during validation: {name}")
    print("Upstream checkout unchanged; ROS build and hardware validation remain required.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    validate(parser.parse_args().upstream.resolve())
