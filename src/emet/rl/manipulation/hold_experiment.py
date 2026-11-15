"""Compare explicit actor initializations, train two bounded hold-repair pilots.

One CPU environment per training job; jobs run sequentially. No hardware,
teacher residual, artifact upload or automatic checkpoint promotion.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from emet.eval.mars_policy_audit import stationary_trial
from emet.eval.mars_tamp_control import CANONICAL_SUITE
from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.env import ManipulationConfig, ManipulationEnv
from emet.rl.manipulation.initialization import transfer_actor
from emet.rl.manipulation.normalization import CommandVecNormalize
from emet.rl.manipulation.policy import ManipulationHistoryFeatures, ManipulationMLPFeatures, export


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def evidence(directory, config, manifests, **extra):
    # A later evaluation/UI edit must not relabel the training implementation.
    # Bind completed training to its recorded source snapshot; compatibility
    # checks decide which of those sources must still match at inference.
    training_run = extra.get("training_run")
    env_record = json.loads(json.dumps(asdict(config)))
    if training_run is not None and (
        json.loads(json.dumps(asdict(ManipulationConfig(**training_run.get("env", {}))))) != env_record
        or not training_run.get("source_hashes")
        or any(m.get("fingerprint") != training_run.get("fingerprint") for m in manifests.values())
    ):
        raise ValueError("training evidence configuration/model mismatch")
    report = {
        "run": {
            "env": training_run["env"] if training_run else env_record,
            "source_hashes": training_run["source_hashes"] if training_run else source_hashes(),
        },
        "checkpoints": {key: {"manifest": value} for key, value in manifests.items()},
        "safety_promoted": False,
        **extra,
    }
    raw = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    archive = directory / "checkpoint_evidence.json.gz"
    archive.write_bytes(gzip.compress(raw, mtime=0))
    summary = directory / "checkpoint_evidence.json"
    save(
        summary, {"full_report_archive": str(archive.resolve()), "uncompressed_sha256": hashlib.sha256(raw).hexdigest()}
    )
    return summary


def architecture_for(spec, source):
    if source is None:
        return spec["architecture"]
    feature = json.loads((Path(source) / "manifest.json").read_text())["feature_extractor"]
    return {"ManipulationHistoryFeatures": "history", "ManipulationMLPFeatures": "mlp", "GoalFeedbackFeatures": "mlp"}[
        feature
    ]


def initial_checkpoint(directory, cfg, source, spec):
    torch.manual_seed(spec["seed"])
    env = CommandVecNormalize(DummyVecEnv([lambda: ManipulationEnv(cfg)]), config=cfg, norm_reward=False, clip_obs=20)
    try:
        from emet.rl.manipulation.feedback_policy import FeedbackActorCriticPolicy, GoalFeedbackFeatures

        model = PPO(
            FeedbackActorCriticPolicy if cfg.feedback_policy else "MultiInputPolicy",
            env,
            device="cpu",
            seed=spec["seed"],
            n_steps=256,
            batch_size=256,
            policy_kwargs={
                "features_extractor_class": GoalFeedbackFeatures
                if cfg.feedback_policy
                else ManipulationHistoryFeatures
                if architecture_for(spec, source) == "history"
                else ManipulationMLPFeatures,
                **(
                    {
                        "features_extractor_kwargs": {
                            "robot": cfg.robot,
                            "retention_context": cfg.retention_context,
                            "orientation_scale": cfg.feedback_orientation_scale,
                        }
                    }
                    if cfg.feedback_policy
                    else {}
                ),
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
            },
        )
        with torch.no_grad():
            if cfg.feedback_policy:
                model.policy.action_net.zero_output()
            else:
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
        details = {"kind": "zero_output", "optimizer_restored": False}
        fingerprint = env.get_attr("fingerprint")[0]
        if source is not None:
            details = transfer_actor(
                model, env, source, cfg, fingerprint, prior_count=spec["normalization_prior_count"]
            )
        export(model, env, directory, cfg, fingerprint, steps=0)
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = evidence(directory, cfg, {"0": manifest}, initialization=details)
        return summary, details
    finally:
        env.close()


def cases_for(directory, summary, case, delta):
    return {
        "hold": stationary_trial(case, directory, summary, selector="checkpoint_policy"),
        "nearby": stationary_trial(
            case, directory, summary, selector="checkpoint_policy", joint_delta=delta, duration_s=3
        ),
    }


def score(result):
    rows = list(result.values())
    successes = sum(r.get("goal_reached_at_end", False) and r.get("completed_duration", False) for r in rows)
    cost = sum(r.get("hand_error_at_end_m", 1) + (0.1 if r.get("failure") else 0) for r in rows)
    return successes, -cost


def training_command(spec, directory, source, mode):
    command = [
        sys.executable,
        "-u",
        "-m",
        "emet.rl.manipulation",
        "train",
        "--architecture",
        architecture_for(spec, source),
        "--initialization-steps",
        "0",
        "--envs",
        "1",
        "--seeds",
        str(spec["seed"]),
        "--steps",
        str(spec["steps"]),
        "--eval-every",
        str(spec.get("eval_every", spec["steps"])),
        "--eval-episodes",
        str(spec.get("eval_episodes", 1)),
        "--learning-rate",
        str(spec["learning_rate"]),
        "--final-learning-rate",
        str(spec.get("final_learning_rate", spec["learning_rate"])),
        "--exploration-std",
        str(spec["exploration_std"]),
        "--task-mix",
        "tracking-retention",
        "--no-held-object-predictor",
        "--wandb-mode",
        mode,
        "--output",
        str(directory),
    ]
    optional_bool = {"trajectory_preview", "tracking_reward", "certify_reach_paths", "dynamic_references"}
    if spec.get("initial_eval_episodes") is not None:
        command += ["--initial-eval-episodes", str(spec["initial_eval_episodes"])]
    for name, value in spec["environment"].items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            elif name in optional_bool:
                command.append("--no-" + name.replace("_", "-"))
        else:
            command += [flag, *[str(v) for v in value]] if isinstance(value, list) else [flag, str(value)]
    if source is not None:
        command += [
            "--transfer-from",
            str(source),
            "--normalization-prior-count",
            str(spec["normalization_prior_count"]),
        ]
    if spec.get("freeze_observation_normalization", False):
        if source is None:
            raise ValueError("frozen observation normalization requires a transfer source")
        command.append("--freeze-observation-normalization")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/benchmarks/mars_hold_training_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-mode", choices=["offline", "disabled"], default="offline")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = json.loads(args.spec.read_text())
    cfg = ManipulationConfig(**spec["environment"])
    args.output.mkdir(parents=True, exist_ok=False)
    catalog = {c["id"]: c for c in json.loads(CANONICAL_SUITE.read_text())["cases"]}
    report = {
        "scope": __doc__,
        "spec": spec,
        "source_hashes": source_hashes(),
        "initializations": {},
        "trained": {},
        "complete": False,
        "safety_promoted": False,
    }
    destination = args.output / "report.json"
    save(destination, report)
    for name, source in {"scratch": None, **spec["initializers"]}.items():
        if cfg.feedback_policy and architecture_for(spec, source) != "mlp":
            report.setdefault("skipped_initializations", {})[name] = "feedback-head pilot requires an MLP context"
            continue
        directory = args.output / "initializations" / name
        directory.mkdir(parents=True)
        summary, details = initial_checkpoint(directory, cfg, source, spec)
        result = cases_for(directory, summary, catalog[spec["selection_case"]], spec["nearby_joint_delta"])
        report["initializations"][name] = {"source": source, "details": details, "results": result}
        save(destination, report)
        print("INITIAL", name, score(result), flush=True)
    tracking = max(
        (k for k in spec["initializers"] if k.startswith("tracking")),
        key=lambda k: score(report["initializations"][k]["results"]),
    )
    report["selected_tracking_initialization"] = tracking
    save(destination, report)
    if args.prepare_only:
        return
    for name, source in (("scratch", None), (tracking, spec["initializers"][tracking])):
        directory = args.output / "training" / name
        directory.mkdir(parents=True)
        command = training_command(spec, directory, source, args.wandb_mode)
        save(directory / "command.json", command)
        with (directory / "train.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        root = directory / f"seed-{spec['seed']}"
        checkpoints = {p.parent.name: json.loads(p.read_text()) for p in root.glob("step-*/manifest.json")}
        # The final numbered export is the exact scored bundle; last is a
        # separate zip serialization and is not substituted into this evidence.
        checkpoint = root / max(checkpoints, key=lambda key: int(key.split("-")[1]))
        summary = evidence(root, cfg, checkpoints, training_run=json.loads((root / "run.json").read_text()))
        results = {
            case_id: cases_for(checkpoint, summary, catalog[case_id], spec["nearby_joint_delta"])
            for case_id in spec["evaluation_cases"]
        }
        report["trained"][name] = {
            "checkpoint": str(checkpoint),
            "results": results,
            "run": json.loads((root / "run.json").read_text()),
            "checkpoint_evidence": str(summary),
        }
        save(destination, report)
        print("TRAINED", name, {k: score(v) for k, v in results.items()}, flush=True)
    if report["source_hashes"] != source_hashes():
        raise RuntimeError("experiment sources changed during training")
    report["complete"] = True
    save(destination, report)


if __name__ == "__main__":
    main()
