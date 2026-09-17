# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

"""Exercise the real shell driver with inert model/scorer executables."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[3] / "scripts/run_shared_grounding_pilot.sh"


@pytest.mark.parametrize("custom,agent_rc,score_rc", [(False, 0, 0), (True, 7, 0), (True, 0, 1)])
def test_sim_preset_selection_and_independent_process_status(tmp_path, custom, agent_rc, score_rc):
    repo = tmp_path / "checkout"
    repo.mkdir()
    for name in (
        "configs/emet/query_detector_segmented_pilot.yaml",
        "configs/sim/default_table_stretch.yaml",
        "configs/benchmarks/tabletop_physical_eval.json",
        "custom agent.yaml",
        "custom scene.yaml",
        "custom evaluator.json",
    ):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    shutil.copy(DRIVER, repo / "driver.sh")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    fake = tmp_path / "agent"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "assert 'EMET_SIM_NAV_TELEPORT' not in os.environ\n"
        "assert os.environ['EMET_MOLMOSPACES_NAV_TELEPORT'] == '0'\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if 'emet.app.run_agent' in sys.argv:\n"
        "    sys.exit(int(os.environ['AGENT_RC']))\n"
        "if 'emet.eval.manipulation_trace' in sys.argv:\n"
        "    sys.exit(int(os.environ['SCORE_RC']))\n"
    )
    fake.chmod(0o755)
    out = tmp_path / "result"
    calls = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PHASE": "manipulation",
        "OUT": str(out),
        "AGENT_PY": str(fake),
        "CALLS": str(calls),
        "AGENT_RC": str(agent_rc),
        "SCORE_RC": str(score_rc),
        "EMET_SIM_NAV_TELEPORT": "1",
        "EMET_MOLMOSPACES_NAV_TELEPORT": "1",
    }
    for name in ("SIM_CONFIG", "SIM_AGENT_CONFIG", "SIM_EVAL_CONFIG", "SIM_COMMAND", "SIM_SEED"):
        env.pop(name, None)
    config = repo / "configs/emet/query_detector_segmented_pilot.yaml"
    scene = repo / "configs/sim/default_table_stretch.yaml"
    evaluator = repo / "configs/benchmarks/tabletop_physical_eval.json"
    instruction = "Use pick_place to put the pear in the sink; report failure."
    if custom:
        config, scene = repo / "custom agent.yaml", repo / "custom scene.yaml"
        env.update(SIM_AGENT_CONFIG=str(config), SIM_CONFIG=str(scene))
        evaluator = repo / "custom evaluator.json"
        env.update(SIM_EVAL_CONFIG=str(evaluator), SIM_COMMAND=instruction, SIM_SEED="1")
    proc = subprocess.run(["bash", "driver.sh"], cwd=repo, env=env, capture_output=True, text=True)
    assert proc.returncode == bool(agent_rc or score_rc), proc.stdout + proc.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    agent = next(args for args in commands if "emet.app.run_agent" in args)
    assert agent[agent.index("--config") + 1] == str(config)
    assert agent[agent.index("--sim-config") + 1] == str(scene)
    if custom:
        assert agent[agent.index("-c") + 1] == instruction
        assert agent[agent.index("--sim-seed") + 1] == "1"
    else:
        assert "red cylinder on the blue cube" in agent[agent.index("-c") + 1]
    assert commands[-1][1] == "emet.eval.manipulation_trace"
    assert f"hybrid_learned_pick_place\t{agent_rc}\t" in (out / "process_status.tsv").read_text()
    assert (out / "sim_agent_config.yaml").read_bytes() == config.read_bytes()
    assert (out / "sim_config.yaml").read_bytes() == scene.read_bytes()
    assert str(scene) in (out / "sim_config_sha256.txt").read_text()
    assert (out / "physical_eval_config.json").read_bytes() == evaluator.read_bytes()
