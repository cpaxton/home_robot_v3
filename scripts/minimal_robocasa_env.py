#!/usr/bin/env python3
"""Minimal example: create Robocasa kitchen env and call reset().

Run: uv run python scripts/minimal_robocasa_env.py

Success: env is created and reset() completes; sim model XML is available.
If placement fails after many retries (RuntimeError), exits with a clear message.
"""
import sys

def main():
    import robocasa  # noqa: F401
    from robosuite import load_part_controller_config
    import robosuite

    config = {
        "env_name": "PickPlaceCounterToCabinet",
        "robots": "PandaMobile",
        "controller_configs": load_part_controller_config(default_controller="OSC_POSE"),
        "translucent_robot": False,
        "layout_and_style_ids": [[1, 1]],
    }
    print("Creating env...", flush=True)
    env = robosuite.make(
        **config,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    print("Calling env.reset()...", flush=True)
    try:
        env.reset()
    except RuntimeError as e:
        if "50 times" in str(e) and "could not initialize" in str(e).lower():
            print(
                "reset() hit max placement retries (scene built, placement sampling failed).",
                flush=True,
            )
            env.close()
            return 0
        raise

    print("reset() OK.", flush=True)
    assert env.sim is not None
    xml = env.sim.model.get_xml()
    print(f"Sim model XML length: {len(xml)} chars.")
    env.close()
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
