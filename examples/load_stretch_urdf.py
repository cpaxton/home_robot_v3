#!/usr/bin/env python3
"""
Load the Stretch URDF model for visualization/kinematics.

Requires: hello-robot-stretch-urdf (pip install hello-robot-stretch-urdf)

Run: python examples/load_stretch_urdf.py
"""

import sys


def main():
    try:
        from stretch.visualization.urdf_visualizer import URDFVisualizer
    except ImportError as e:
        print("Error: Could not import URDFVisualizer.")
        print("Install hello-robot-stretch-urdf: pip install hello-robot-stretch-urdf")
        print(f"  {e}")
        sys.exit(1)

    print("=== Stretch URDF Load Test ===\n")

    print("Loading URDF model...")
    visualizer = URDFVisualizer()
    urdf = visualizer.urdf

    print(f"  Robot name: {urdf.name}")
    print(f"  Links: {len(urdf.link_map)}")
    print(f"  Actuated joints: {len(urdf.actuated_joint_names)}")
    print(f"  Joint names: {urdf.actuated_joint_names[:5]}...")

    # Test FK (cfg=None uses default/zero config)
    print("\nTesting forward kinematics...")
    meshes = visualizer.get_tri_meshes(cfg=None, use_collision=False)
    print(f"  Got {len(meshes['mesh'])} mesh links")

    print("\n✓ URDF loaded successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
