"""Fit a shared Mars simulator actor from task-balanced demonstrations and corrections."""

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.arm_spline_benchmark import save
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle, export_actor
from emet.rl.manipulation.whole_body_observation import ACTOR_KEYS

FAMILIES = ("base", "arm", "coupled", "carry", "tamp", "lift", "preplace")
CHANNELS = {
    "base": (0, 1),
    "arm": (2, 3, 4, 5, 6),
    "tamp": (2, 3, 4, 5, 6),
    "preplace": (2, 3, 4, 5, 6),
    "lift": (2, 3, 4, 5, 6),
}
SHAPES = {"history": (20, 101), "preview": (7, 18), "motion": (9,), "action": (8,)}


def signature(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, device):
    with np.load(path, allow_pickle=False) as archive:
        result = {k: np.asarray(archive[k], dtype=np.float32) for k in SHAPES}
    n = len(result["action"])
    if not n or any(v.shape != (n, *SHAPES[k]) or not np.isfinite(v).all() for k, v in result.items()):
        raise ValueError(f"Invalid training tensors {path}")
    if np.any(abs(result["action"]) > 1.000001):
        raise ValueError(f"Out-of-range actions {path}")
    return {k: torch.as_tensor(v, device=device) for k, v in result.items()}


def atomic_state(path, state):
    temp = path.with_suffix(".partial")
    torch.save(state, temp)
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=6000)
    parser.add_argument("--updates", type=int, default=54000)
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 0 < args.bootstrap < args.updates or not np.isfinite(args.hours) or args.hours <= 0 or args.threads <= 0:
        parser.error("require 0 < bootstrap < updates, a positive time budget and positive thread count")
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261120)
    rng = np.random.default_rng(20261120)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this runtime")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assumed_end = time.time() + args.hours * 3600
    report = {
        "complete": False,
        "stage": "waiting_for_inputs",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "updates": 0,
        "planned_updates": args.updates,
        "bootstrap_updates": args.bootstrap,
        "planned_hours": args.hours,
        "datasets": {},
        "online_datasets": {},
        "checkpoints": [],
        "loss": [],
    }
    report_path = output / "status.json"
    weights = args.work / "initializer"
    load_inputs = args.work / "data"
    if report_path.exists() and json.loads(report_path.read_text()).get("complete"):
        raise ValueError(f"Training already completed in {output}")
    if not report_path.exists():
        save(report_path, report)
    wait_stop = min(time.time() + 1200, assumed_end)
    while not all((load_inputs / (name + ".npz")).exists() for name in FAMILIES):
        if time.time() > wait_stop:
            missing = [name for name in FAMILIES if not (load_inputs / (name + ".npz")).exists()]
            raise RuntimeError(f"Training input wait exceeded: {missing}")
        time.sleep(5)
    teacher = {name: load(load_inputs / (name + ".npz"), device) for name in FAMILIES}
    online = {}
    report["datasets"] = {
        name: {"rows": len(data["action"]), "sha256": signature(load_inputs / (name + ".npz"))}
        for name, data in teacher.items()
    }
    initial = WholeBodyBundle(weights, robot="innate_mars")
    report["initializer_sha256"] = initial.manifest["weights_sha256"]
    model = initial.actor.to(device).train()
    model.requires_grad_(True)
    model.grip.requires_grad_(False)
    masks = {name: torch.as_tensor(CHANNELS.get(name, tuple(range(7))), device=device) for name in FAMILIES}
    cfg = ManipulationConfig(**initial.manifest["environment"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.00012)
    state_path = output / "resume.pt"
    if state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=False)
        resume_keys = ("datasets", "initializer_sha256", "planned_updates", "bootstrap_updates")
        if any(saved["report"].get(key) != report[key] for key in resume_keys):
            raise RuntimeError("Cannot resume with different initial data, model, or update schedule")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        report = saved["report"]
        rng.bit_generator.state = saved["random_state"]
        assumed_end = saved["deadline"]
    start = int(report["updates"])
    accum = {name: [0.0, 0] for name in FAMILIES}
    if start <= args.bootstrap:
        report["stage"] = "distilling"
    save(report_path, report)
    for step in range(start + 1, args.updates + 1):
        if step > args.bootstrap and (step in (start + 1, args.bootstrap + 1) or step % 100 == 1):
            for family in FAMILIES:
                p = load_inputs / f"online-{family}.npz"
                if p.exists() and family not in online:
                    online[family] = load(p, device)
                    receipt = {"rows": len(online[family]["action"]), "sha256": signature(p)}
                    if family in report["online_datasets"] and report["online_datasets"][family] != receipt:
                        raise RuntimeError(f"Cannot resume with a replaced correction file for {family}")
                    report["online_datasets"][family] = receipt
            if set(report["online_datasets"]) - set(online):
                raise RuntimeError("Cannot resume without previously loaded correction files")
            report["stage"] = (
                "randomized_expert_replay_with_online_corrections" if online else "randomized_expert_replay"
            )
        names = rng.choice(FAMILIES, size=4, replace=False)
        group, targets, channel_masks = [], [], []
        for family in names:
            data = online[family] if family in online and rng.random() < 0.35 else teacher[family]
            indices = torch.as_tensor(rng.integers(0, len(data["action"]), size=32), device=device)
            group.append({k: data[k][indices] for k in ACTOR_KEYS})
            targets.append(data["action"][indices])
            channel_masks.append(masks[family])
        prediction = model({k: torch.cat([a[k] for a in group]) for k in ACTOR_KEYS}).split(32)
        losses = [
            (p[:, mask] - t[:, mask]).square().mean()
            for p, t, mask in zip(prediction, targets, channel_masks, strict=True)
        ]
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0:
            observed = torch.stack(losses).detach().cpu().tolist()
            if not np.isfinite(observed).all():
                raise RuntimeError("Nonfinite distillation loss")
            for name, value in zip(names, observed, strict=True):
                accum[name][0] += value
                accum[name][1] += 1
        deadline = time.time() >= assumed_end
        export = step in (1000, args.bootstrap) or step % 12000 == 0 or deadline or step == args.updates
        if export or step % 500 == 0:
            report["updates"] = step
            report["loss"].append({"step": step, "per_family_mse": {k: s / n for k, (s, n) in accum.items() if n}})
            accum = {name: [0.0, 0] for name in FAMILIES}
            if export:
                directory = output / f"actor-{step:06d}"
                fingerprint = initial.manifest["fingerprint"]
                exported_model = copy.deepcopy(model).cpu()
                if directory.exists():
                    # A crash can occur after the bundle is published but
                    # before the checkpoint is committed.
                    exported = WholeBodyBundle(directory, robot="innate_mars", fingerprint=fingerprint)
                    if exported.manifest["steps"] != step:
                        raise ValueError(f"Unexpected checkpoint step in {directory}")
                    for name, value in exported_model.state_dict().items():
                        torch.testing.assert_close(exported.actor.state_dict()[name], value, atol=0, rtol=0)
                else:
                    exported = export_actor(
                        exported_model,
                        directory,
                        cfg,
                        fingerprint,
                        steps=step,
                        training={"method": "multi_family_expert_distillation", "families": list(FAMILIES)},
                    )
                report["checkpoints"].append(
                    {"step": step, "weights": exported.manifest["weights_sha256"], "path": directory.name}
                )
            atomic_state(
                state_path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "report": report,
                    "random_state": rng.bit_generator.state,
                    "deadline": assumed_end,
                },
            )
            save(report_path, report)
            print("step", step, "stage", report["stage"], "online", list(online), flush=True)
        if deadline:
            break
    report.update(complete=True, stage="complete", finished_at=time.time())
    save(report_path, report)


if __name__ == "__main__":
    main()
