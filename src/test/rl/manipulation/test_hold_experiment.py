import gzip
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from emet.rl.manipulation import __main__ as entry
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.hold_experiment import evidence, training_command


@pytest.mark.parametrize("transfer", [False, True])
@pytest.mark.parametrize(
    "name",
    [
        "mars_hold_training_v1.json",
        "mars_hold_training_feedback_v2.json",
        "mars_hold_training_feedback_v3.json",
        "mars_hold_training_feedback_v4.json",
        "mars_hold_training_feedback_v5.json",
        "mars_hold_training_feedback_v6.json",
        "mars_general_continuation_v1.json",
        "mars_pose_translation_curriculum_v1.json",
    ],
)
def test_experiment_cli_preserves_config_and_uses_one_cpu_worker(tmp_path, monkeypatch, transfer, name):
    spec = json.loads((Path("configs/benchmarks") / name).read_text())
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"feature_extractor": "ManipulationMLPFeatures"}))
    if spec.get("freeze_observation_normalization") and not transfer:
        with pytest.raises(ValueError, match="source"):
            training_command(spec, tmp_path, None, "disabled")
        return
    command = training_command(spec, tmp_path, source if transfer else None, "disabled")
    captured = []
    monkeypatch.setattr(entry, "train", captured.append)
    monkeypatch.setattr("sys.argv", ["manipulation", *command[command.index("train") :]])
    entry.main()
    args = captured[0]
    assert args.envs == 1 and args.initialization_steps == 0
    assert args.transfer_from == (source if transfer else None)
    assert args.architecture == "mlp"
    assert args.reach_hold_probability == spec["environment"]["reach_hold_probability"]
    assert args.fixed_base_probability == spec["environment"]["fixed_base_probability"]
    assert args.categorical_features_unscaled and args.hold_pregrasp_fixture
    assert args.exploration_std == spec["exploration_std"]
    assert args.final_learning_rate == spec.get("final_learning_rate", spec["learning_rate"])
    assert args.eval_every == spec.get("eval_every", spec["steps"])
    assert args.held_object_predictor is False
    assert args.feedback_policy == spec["environment"].get("feedback_policy", False)
    assert args.fixed_base_contract == spec["environment"].get("fixed_base_contract", False)
    assert args.pose_progress_weight == spec["environment"].get("pose_progress_weight", 0)
    assert args.reach_local_translation_probability == spec["environment"].get("reach_local_translation_probability", 0)
    assert args.freeze_observation_normalization == spec.get("freeze_observation_normalization", False)
    assert ManipulationConfig(**spec["environment"]).scenario_weights[0] >= 0.5


def test_frozen_continuation_command_requires_source_and_preserves_its_flag(tmp_path):
    spec = json.loads(Path("configs/benchmarks/mars_general_continuation_v1.json").read_text())
    spec["freeze_observation_normalization"] = True
    with pytest.raises(ValueError, match="source"):
        training_command(spec, tmp_path, None, "disabled")
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"feature_extractor": "GoalFeedbackFeatures"}))
    assert "--freeze-observation-normalization" in training_command(spec, tmp_path, source, "disabled")


def test_completed_training_evidence_keeps_original_sources(tmp_path):
    cfg = ManipulationConfig()
    run = {
        "env": json.loads(json.dumps(asdict(cfg))),
        "source_hashes": {"original_module": "original_sha"},
        "fingerprint": "frozen-model",
    }
    summary = evidence(tmp_path, cfg, {"step-1": {"fingerprint": "frozen-model"}}, training_run=run)
    stored = json.loads(gzip.decompress(Path(json.loads(summary.read_text())["full_report_archive"]).read_bytes()))
    assert stored["run"]["source_hashes"] == run["source_hashes"]
    # Historical runs omit options introduced later. Preserve their original
    # metadata after validating that omitted values mean the same defaults.
    run["env"].pop("pose_progress_weight")
    summary = evidence(tmp_path, cfg, {"step-1": {"fingerprint": "frozen-model"}}, training_run=run)
    stored = json.loads(gzip.decompress(Path(json.loads(summary.read_text())["full_report_archive"]).read_bytes()))
    assert "pose_progress_weight" not in stored["run"]["env"]
    run["env"]["horizon"] = 100
    with pytest.raises(ValueError, match="configuration/model"):
        evidence(tmp_path, cfg, {"step-1": {"fingerprint": "frozen-model"}}, training_run=run)
