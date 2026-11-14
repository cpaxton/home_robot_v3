"""Experimental direct whole-body PPO, with source-bound simulator preflight.

python -m emet.rl.manipulation.whole_body_training --help
The actor has no teacher fallback, object truth, task logic or deployment motion.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.manipulation.multi_critic_ppo import MultiCriticPPO, PPOConfig, grouped_gae
from emet.rl.manipulation.whole_body_actor import MultiCriticPolicy

DEFAULT_SUITE = Path("configs/benchmarks/mars_whole_body_v2_development.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_development(path=DEFAULT_SUITE):
    from emet.rl.manipulation.mastery import load_suite

    suite = json.loads(Path(path).read_text())
    if (
        suite.get("format") != "mars-whole-body-v2-development-v1"
        or suite.get("general_stages") != ["base", "arm", "coupled"]
        or suite.get("arm_reference") != "s_curve"
        or suite.get("family_control_tick_weights") != [0.5, 0.5]
        or suite.get("general_stage_weights") != {"base": 0.3, "arm": 0.5, "coupled": 0.2}
        or suite.get("teacher_expected_failures") != {"development:left:50g:2": "object_changed_replan"}
    ):
        raise ValueError("unsupported whole-body development curriculum")
    seen, seeds = set(), set()
    for split, start in (("train_cases", 34600000), ("audit_cases", 34610000)):
        if not suite.get(split):
            raise ValueError("nonempty disjoint development and audit pools required")
        for case in suite[split]:
            if (
                set(case) != {"id", "layout", "seed", "mass_kg", "yaw_offset_rad"}
                or case["id"] in seen
                or type(case["seed"]) is not int
                or case["seed"] in seeds
                or not start <= case["seed"] < start + 10000
                or case["layout"] not in {"left", "right"}
                or case["mass_kg"] not in {0.02, 0.05, 0.1}
                or case["yaw_offset_rad"] != (0.6 if case["layout"] == "left" else -0.6)
            ):
                raise ValueError("invalid/overlapping TAMP development cases")
            seen.add(case["id"])
            seeds.add(case["seed"])
    general = load_suite(suite["general_suite"])
    return suite, general


def primitive_rollout(cases, *, policy=None, arm_reference="s_curve", initial_settle_s=0.0):
    from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv

    env = TampPrimitiveEnv(cases, oracle=policy is None, arm_reference=arm_reference, initial_settle_s=initial_settle_s)
    rows = []
    try:
        for case in cases:
            ticks, count, returns = 0, 0, np.zeros(3)
            try:
                obs, info = env.reset(options={"case_id": case["id"]})
                for _ in range(2000):
                    count += 1
                    if policy is None:
                        action = info.get("oracle_action", np.zeros(8))
                    else:
                        sensor = {k: torch.from_numpy(obs[k])[None] for k in ("history", "preview", "motion")}
                        with torch.no_grad():
                            action = policy(sensor)[0].numpy()
                    obs, _, done, truncated, info = env.step(action)
                    returns += info["reward_groups"]
                    ticks += info["elapsed_control_ticks"]
                    if done or truncated:
                        break
                else:
                    raise RuntimeError("TAMP primitive exceeded bounded decisions")
                rows.append(
                    {
                        "id": case["id"],
                        "success": info["primitive_success"],
                        "code": info["code"],
                        "stop_confirmed": info["stop_confirmed"],
                        "control_ticks": ticks,
                        "decisions": count,
                        "returns": returns.tolist(),
                        "commands": info["commands"],
                        "inference_counts": info["policy_calls"],
                        "full_pick_place_success": False,
                        "object_translation_m": info["object_translation_m"],
                        "object_rotation_rad": info["object_rotation_rad"],
                        "executor_result": info["executor_result"],
                        "initial_settle_s": info["initial_settle_s"],
                        "initialization_failed": info["initialization_failed"],
                        "safety_flags": {
                            k: info[k] for k in ("wheel_lift", "invalid_physics", "unsafe_contact", "retention_lost")
                        },
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "id": case["id"],
                        "success": False,
                        "code": str(exc),
                        "control_ticks": ticks,
                        "decisions": count,
                        "initialization_failed": count == 0,
                        "stop_confirmed": None,
                        "execution_error": True,
                        "full_pick_place_success": False,
                    }
                )
            print("PRIMITIVE", rows[-1]["id"], rows[-1]["success"], rows[-1]["code"], flush=True)
    finally:
        env.close()
    return rows


def tamp_training_readiness(rows, suite):
    """Experimental admission, explicitly distinct from all-success qualification.

    One source-bound, diagnosed safe-stop challenge is retained in training and
    in the denominator. Unknown failures, rejected plans or safety faults block.
    No acceptance success flag or threshold is changed to make this pass.
    """
    expected = [c["id"] for c in suite["train_cases"] + suite["audit_cases"]]
    if [r.get("id") for r in rows] != expected:
        return False
    for row in rows:
        if (
            row.get("stop_confirmed") is not True
            or row.get("initialization_failed") is not False
            or set(row.get("safety_flags", {})) != {"wheel_lift", "invalid_physics", "unsafe_contact", "retention_lost"}
            or any(row["safety_flags"].values())
            or not row.get("commands")
            or row.get("decisions", 0) <= 0
        ):
            return False
        if not row["success"] and suite["teacher_expected_failures"].get(row["id"]) != row["code"]:
            return False
    return True


def preflight(suite_path, output):
    from emet.rl.manipulation.mastery import evaluate_suite, save_report
    from emet.rl.regression_gate import ROOT, source_state

    suite, general = load_development(suite_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    bound_state = source_state(ROOT)
    general_report = evaluate_suite(general, stages=suite["general_stages"], population="admission")
    save_report(output / "general-teacher.json.gz", general_report)
    rows = primitive_rollout(suite["train_cases"] + suite["audit_cases"])
    if sources != source_hashes() or bound_state != source_state(ROOT):
        raise RuntimeError("sources changed during whole-body preflight")
    report = {
        "format": "whole-body-v2-preflight-v1",
        "source_hashes": sources,
        "source_state": bound_state,
        "suite_sha256": digest(suite_path),
        "general_suite_sha256": digest(suite["general_suite"]),
        "general_teacher_sha256": digest(output / "general-teacher.json.gz"),
        "general_cases": len(general_report["cases"]),
        "tamp_cases": rows,
        "ready_for_experimental_training": (
            all(r["mastered"] for r in general_report["cases"]) and tamp_training_readiness(rows, suite)
        ),
        "tamp_successes": sum(r["success"] for r in rows),
        "tamp_attempts": len(rows),
        "tamp_all_success": all(r["success"] for r in rows),
        "simulation_only": True,
        "safety_promoted": False,
        "full_pick_place_qualified": False,
    }
    write_json(output / "preflight.json", report)
    return report


def require_preflight(path, suite_path):
    from emet.rl.regression_gate import ROOT, source_state

    suite, _ = load_development(suite_path)
    path = Path(path)
    report = json.loads(path.read_text())
    expected = [c["id"] for c in suite["train_cases"] + suite["audit_cases"]]
    rows = report.get("tamp_cases", [])
    if (
        report.get("format") != "whole-body-v2-preflight-v1"
        or report.get("ready_for_experimental_training") is not True
        or report.get("source_hashes") != source_hashes()
        or report.get("source_state") != source_state(ROOT)
        or report.get("suite_sha256") != digest(suite_path)
        or report.get("general_suite_sha256") != digest(suite["general_suite"])
        or report.get("general_teacher_sha256") != digest(path.parent / "general-teacher.json.gz")
        or [r.get("id") for r in rows] != expected
        or not tamp_training_readiness(rows, suite)
    ):
        raise ValueError("fresh passing whole-body teacher preflight required")
    # Independently check the frozen teacher report, not just a summary boolean.
    from emet.rl.manipulation.mastery import read_report, validate_preflight

    _, general = load_development(suite_path)
    teacher = read_report(path.parent / "general-teacher.json.gz")
    validate_preflight(teacher, general, suite["general_stages"])
    if not all(r["mastered"] for r in teacher["cases"]):
        raise ValueError("general teacher failed its original gates")
    return report


def make_environment(suite_path, *, basic_cases=None):
    from emet.rl.manipulation.mastery import MasteryEnv
    from emet.rl.manipulation.tamp_primitive_env import TampPrimitiveEnv
    from emet.rl.manipulation.whole_body_env import ControlStepMixture, GeneralTrackingEnv

    suite, general = load_development(suite_path)
    if basic_cases is not None:
        from emet.rl.manipulation.basic_learning import load_basic_learning

        spec, selected, _ = load_basic_learning(basic_cases, general)
        native = MasteryEnv(
            selected,
            stages=("base", "arm"),
            split="train",
            population="training",
            training_stage_weights=spec["training_stage_weights"],
        )
        try:
            return ControlStepMixture({"general": GeneralTrackingEnv(native)}, [1.0])
        except BaseException:
            native.close()
            raise
    native = MasteryEnv(
        general,
        stages=suite["general_stages"],
        split="train",
        population="training",
        training_stage_weights=suite["general_stage_weights"],
    )
    try:
        return ControlStepMixture(
            {
                "general": GeneralTrackingEnv(native),
                "tamp": TampPrimitiveEnv(suite["train_cases"], arm_reference=suite["arm_reference"]),
            },
            suite["family_control_tick_weights"],
        )
    except BaseException:
        native.close()
        raise


def tensor_observation(obs):
    return {k: torch.from_numpy(np.asarray(v, dtype=np.float32))[None] for k, v in obs.items()}


def arrays_digest(arrays):
    state = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        state.update(name.encode() + b"\0" + str(array.shape).encode() + array.dtype.str.encode())
        state.update(array.tobytes())
    return state.hexdigest()


def policy_state_digest(policy):
    return arrays_digest({name: value.detach().cpu().numpy() for name, value in policy.state_dict().items()})


def verify_initial_pair(reference, current):
    """Reject an unpaired candidate before its first optimizer update."""
    keys = (
        "source_hashes",
        "seed",
        "suite_sha256",
        "preflight_sha256",
        "basic_cases_sha256",
        "requested_decisions",
        "initial_actor_weights_sha256",
        "initial_policy_state_sha256",
        "initial_optimizer_sha256",
        "rollout_rng_sha256",
        "initial_numpy_rng_sha256",
        "first_rollout",
    )
    if any(key not in reference or key not in current or reference[key] != current[key] for key in keys):
        raise ValueError("paired initial state or first pre-update rollout differs")
    first = {k: v for k, v in reference["ppo"].items() if k != "advantage_normalization"}
    second = {k: v for k, v in current["ppo"].items() if k != "advantage_normalization"}
    if first != second:
        raise ValueError("paired PPO settings differ beyond normalization scheme")


def collect_rollout(env, model, observation, steps, *, episode_callback=None):
    """Bootstrap timeouts; never propagate GAE across resets/primitive boundaries."""
    collected = {
        k: []
        for k in (
            "actions",
            "log_prob",
            "rewards",
            "values",
            "next_values",
            "terminated",
            "episode_end",
            "ticks",
            "case_ids",
        )
    }
    observations = {k: [] for k in observation}
    for _ in range(steps):
        tensors = tensor_observation(observation)
        with torch.no_grad():
            distribution = model.distribution(tensors)
            # Log probability of the sampled action, before actuator clipping.
            action = distribution.sample() * model.actor.action_mask(tensors)
            log, _, values = model.evaluate(tensors, action)
        next_obs, _, terminated, truncated, info = env.step(action[0].numpy())
        with torch.no_grad():
            next_values = model.values(tensor_observation(next_obs))[0].numpy()
        for k, v in observation.items():
            observations[k].append(v.copy())
        entries = {
            "actions": action[0].numpy(),
            "log_prob": log.item(),
            "rewards": info["reward_groups"],
            "values": values[0].numpy(),
            "next_values": next_values,
            "terminated": terminated,
            "episode_end": terminated or truncated,
            "ticks": info["elapsed_control_ticks"],
            "case_ids": info.get("case_id") or info.get("mastery", {}).get("case_id", "unknown"),
        }
        for k, v in entries.items():
            collected[k].append(v)
        if terminated or truncated:
            if episode_callback is not None:
                episode_callback(info)
            observation, _ = env.reset()
        else:
            observation = next_obs
    return {
        **{k: np.asarray(v) for k, v in collected.items()},
        "observations": {k: np.asarray(v) for k, v in observations.items()},
    }, observation


def train(args):
    from emet.rl.manipulation.whole_body_bundle import export_actor
    from emet.rl.regression_gate import verify_regressions

    receipt = verify_regressions(args.regression_receipt)
    require_preflight(args.preflight, args.suite)
    sources = source_hashes()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = PPOConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        advantage_normalization=args.advantage_normalization,
    )
    policy = MultiCriticPolicy(privileged=not args.no_privileged_critic)
    learner = MultiCriticPPO(policy, config)
    metadata = {
        "format": "whole-body-v2-training-v1",
        "source_hashes": sources,
        "suite_sha256": digest(args.suite),
        "preflight_sha256": digest(args.preflight),
        "seed": args.seed,
        "ppo": asdict(config),
        "requested_decisions": args.steps,
        "regression_receipt_sha256": digest(args.regression_receipt),
        "regression_tests": len(receipt["test_ids"]),
        "hardware_used": False,
        "safety_promoted": False,
        "teacher_actions_in_loss": False,
        "privileged_critic": not args.no_privileged_critic,
        "actor_inputs": ["history", "preview", "motion"],
        "full_pick_place_qualified": False,
        "basic_cases_sha256": digest(args.basic_cases) if args.basic_cases else None,
        "training_family_scope": "basic_general_only" if args.basic_cases else "general_tamp_50_50_ticks",
    }
    write_json(output / "run.json", metadata)
    run = None
    env = None
    try:
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="emet-whole-body-v2",
                name=output.name,
                mode=args.wandb_mode,
                dir=str(output),
                config=metadata,
                settings=wandb.Settings(disable_git=True, save_code=False),
            )
        env = make_environment(args.suite, basic_cases=args.basic_cases)
        native = env.environments["general"].unwrapped
        initial_bundle = export_actor(
            policy.actor,
            output / "initial/actor",
            native.config,
            native.fingerprint,
            steps=0,
            training={"scope": "initial_before_ppo", "seed": args.seed},
        )
        metadata.update(
            initial_actor_weights_sha256=initial_bundle.manifest["weights_sha256"],
            initial_policy_state_sha256=policy_state_digest(policy),
            initial_action_std=policy.log_std.detach().exp().tolist(),
        )
        if args.basic_cases:
            from emet.rl.manipulation.basic_learning import load_basic_learning
            from emet.rl.manipulation.mastery import evaluate_suite, read_report, save_report, summarize
            from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle

            _, general = load_development(args.suite)
            spec, _, evaluation_suite = load_basic_learning(args.basic_cases, general)
            teacher = read_report(Path(args.preflight).parent / "general-teacher.json.gz")
            write_json(output / "basic-cases.json", spec)

            def evaluate_basic(label, directory):
                report = evaluate_suite(
                    evaluation_suite, stages=("base", "arm"), policy=WholeBodyBundle(directory), teacher_report=teacher
                )
                save_report(output / f"{label}-evaluation.json.gz", report)
                write_json(output / f"{label}-evaluation-summary.json", summarize(report))

            evaluate_basic("initial", output / "initial/actor")
            # Baseline measurement must not change the paired PPO random stream.
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        observation, _ = env.reset(seed=args.seed)
        numpy_rng = np.random.get_state()
        numpy_state = [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]]
        initial_state = output / "initial/training.pt"
        torch.save(
            {
                "policy": policy.state_dict(),
                "optimizer": learner.optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": numpy_state,
            },
            initial_state,
        )
        metadata.update(
            initial_training_state_sha256=digest(initial_state),
            rollout_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            initial_numpy_rng_sha256=hashlib.sha256(json.dumps(numpy_state).encode()).hexdigest(),
            initial_optimizer_sha256=hashlib.sha256(
                json.dumps(learner.optimizer.state_dict(), sort_keys=True).encode()
            ).hexdigest(),
        )
        write_json(output / "run.json", metadata)
        episodes, decisions, ticks = [], 0, 0

        def episode(info):
            summary = {
                k: info.get(k)
                for k in (
                    "training_family",
                    "code",
                    "primitive_success",
                    "stop_confirmed",
                    "wheel_lift",
                    "retention_lost",
                    "invalid_physics",
                    "unsafe_contact",
                )
            }
            summary["case_id"] = info.get("case_id") or info.get("mastery", {}).get("case_id")
            episodes.append(summary)
            with (output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(summary, allow_nan=False) + "\n")

        while decisions < args.steps:
            rollout, observation = collect_rollout(
                env, policy, observation, min(args.rollout_steps, args.steps - decisions), episode_callback=episode
            )
            if args.basic_cases and not np.all(rollout["ticks"] == 1):
                raise RuntimeError("basic C1 training requires one physical tick per decision")
            if decisions == 0:
                flattened = {"obs/" + k: v for k, v in rollout["observations"].items()}
                flattened.update({k: v for k, v in rollout.items() if k != "observations"})
                raw_advantage, _ = grouped_gae(
                    rollout["rewards"],
                    rollout["values"],
                    rollout["next_values"],
                    rollout["terminated"],
                    rollout["episode_end"],
                    rollout["ticks"],
                    gamma=config.gamma,
                    lam=config.gae_lambda,
                )
                np.savez_compressed(output / "first-rollout.npz", **flattened, raw_advantage=raw_advantage)
                metadata["first_rollout"] = {
                    "arrays_sha256": arrays_digest(flattened),
                    "observations_sha256": arrays_digest(rollout["observations"]),
                    "actions_sha256": arrays_digest({"actions": rollout["actions"]}),
                    "rewards_sha256": arrays_digest({"rewards": rollout["rewards"]}),
                    "raw_gae_sha256": arrays_digest({"raw_gae": raw_advantage}),
                }
                write_json(output / "run.json", metadata)
                if getattr(args, "pair_reference_run", None):
                    reference_path = Path(args.pair_reference_run)
                    verify_initial_pair(json.loads(reference_path.read_text()), metadata)
                    metadata["paired_reference_run_sha256"] = digest(reference_path)
                    metadata["initial_pair_verified_before_update"] = True
                    write_json(output / "run.json", metadata)
            stats = learner.update(rollout)
            decisions += len(rollout["ticks"])
            ticks += int(rollout["ticks"].sum())
            row = {
                "decisions": decisions,
                "control_ticks": ticks,
                "mixture": env.ledger(),
                **stats,
                "reward_group_means": rollout["rewards"].mean(0).tolist(),
                "case_sequence": rollout["case_ids"].tolist(),
            }
            with (output / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if run is not None:
                scalar = {
                    "decisions": decisions,
                    "control_ticks": ticks,
                    "actor_loss": stats["actor_loss"],
                    "approx_kl": stats["approx_kl"],
                }
                scalar.update({f"critic/{i}": v for i, v in enumerate(stats["critic_losses"])})
                scalar.update({f"ticks/{name}": v["control_ticks"] for name, v in env.ledger().items()})
                scalar["combined_tracking_correlation"] = stats["combined_tracking_correlation"]
                run.log(scalar, step=decisions)
            print("UPDATE", json.dumps(row, allow_nan=False), flush=True)
            # A crash loses at most one rollout. This is training state only;
            # critic weights never enter the deployed actor bundle.
            checkpoint = output / "training.pt.tmp"
            torch.save(
                {
                    "policy": policy.state_dict(),
                    "optimizer": learner.optimizer.state_dict(),
                    "run": metadata,
                    "decisions": decisions,
                    "control_ticks": ticks,
                    "mixture": env.ledger(),
                    "torch_rng": torch.get_rng_state(),
                },
                checkpoint,
            )
            os.replace(checkpoint, output / "training.pt")
        if sources != source_hashes():
            raise RuntimeError("sources changed while training")
        verify_regressions(args.regression_receipt)
        training = {
            "algorithm": "direct_multi_critic_ppo_v2",
            "advantage_normalization": config.advantage_normalization,
            "basic_cases_sha256": metadata["basic_cases_sha256"],
            "training_family_scope": metadata["training_family_scope"],
            "initial_actor_weights_sha256": metadata["initial_actor_weights_sha256"],
            "initial_policy_state_sha256": metadata["initial_policy_state_sha256"],
            "decision_steps": decisions,
            "control_ticks": ticks,
            "mixture": env.ledger(),
            "source_hashes": sources,
            "preflight_sha256": digest(args.preflight),
            "privileged_critic": policy.privileged,
            "teacher_actions_in_loss": False,
            "full_pick_place_qualified": False,
        }
        export_actor(
            policy.actor, output / "last/actor", native.config, native.fingerprint, steps=decisions, training=training
        )
        if args.basic_cases:
            evaluate_basic("final", output / "last/actor")
        write_json(
            output / "summary.json",
            {**training, "episodes": len(episodes), "complete": True, "safety_promoted": False, "hardware_used": False},
        )
    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train"))
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--regression-receipt", type=Path)
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-privileged-critic", action="store_true")
    parser.add_argument("--advantage-normalization", choices=("shared", "per_head"), default="shared")
    parser.add_argument("--basic-cases", type=Path, help="explicit small general-only diagnostic case manifest")
    parser.add_argument(
        "--pair-reference-run", type=Path, help="verify initial state/rollout against control run before PPO"
    )
    parser.add_argument("--wandb-mode", choices=("offline", "disabled"), default="offline")
    args = parser.parse_args()
    if args.command == "preflight":
        report = preflight(args.suite, args.output)
        if not report["ready_for_experimental_training"]:
            raise SystemExit("teacher preflight failed; training remains blocked")
    else:
        if (
            args.preflight is None
            or args.regression_receipt is None
            or args.steps < 2
            or args.rollout_steps < 2
            or args.steps % args.rollout_steps == 1
        ):
            parser.error("training requires fresh preflight/regressions and rollouts with at least two samples")
        train(args)


if __name__ == "__main__":
    main()
