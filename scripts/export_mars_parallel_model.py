"""Export a portable, physically initialized Mars model for backend parity tests."""

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import mujoco

    from emet.rl.manipulation.mastery import MasteryEnv, load_suite

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = MasteryEnv(
        load_suite(ROOT / "configs/benchmarks/mars_tracking_mastery_restoring_yaw_v1.json"), stages=["base"]
    )
    try:
        _, info = env.reset(options={"case_id": "train:base:yaw_left"})
        if info["initialization_failed"]:
            raise RuntimeError(info["mastery"])
        xml = args.output / "model.xml"
        mujoco.mj_saveLastXML(str(xml), env.model)
        tree = ET.parse(xml)
        root = tree.getroot()
        compiler = root.find("compiler")
        meshdir = Path(compiler.get("meshdir", "")) if compiler is not None else Path()
        assets = args.output / "assets"
        assets.mkdir()
        for node in root.findall(".//asset/*"):
            if "file" not in node.attrib:
                continue
            source = Path(node.get("file"))
            source = source if source.is_absolute() else meshdir / source
            name = hashlib.sha256(source.read_bytes()).hexdigest()[:12] + "-" + source.name
            shutil.copy2(source, assets / name)
            node.set("file", "assets/" + name)
        if compiler is not None:
            for name in ["meshdir", "texturedir", "assetdir"]:
                compiler.attrib.pop(name, None)
        tree.write(xml, encoding="unicode")
        metadata = {
            "format": "emet-mars-backend-parity-v1",
            "mujoco_version": mujoco.__version__,
            "fingerprint": env.fingerprint,
            "nq": env.model.nq,
            "nv": env.model.nv,
            "nu": env.model.nu,
            "physics_dt": float(env.model.opt.timestep),
            "noslip_iterations": int(env.model.opt.noslip_iterations),
            "base_body": "base_link",
            "base_qpos": int(env.model.jnt_qposadr[env.model.body_jntadr[env.control.base]]),
            "wheel_act": env.control.wheel_act.tolist(),
            "wheel_gear": env.control.wheel_gear.tolist(),
            "wheel_radius": env.profile.wheel_radius,
            "wheel_separation": env.profile.wheel_separation,
            "base_speed": env.profile.base_speed,
            "base_yaw_speed": env.profile.base_yaw_speed,
            "qpos": env.data.qpos.copy().tolist(),
            "qvel": env.data.qvel.copy().tolist(),
            "ctrl": env.data.ctrl.copy().tolist(),
            "time": float(env.data.time),
            "hardware_used": False,
        }
        (args.output / "state.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("Portable native Mars model:", args.output, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
