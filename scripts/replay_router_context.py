# Fixed captured router input experiment; no robot actions, no semantic gold.
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from emet.agent.prompt import parse_tool_calls_response
from emet.core.parameters import get_parameters
from emet.llms.graph_eqa_vlm import build_graph_eqa_vlm_clients

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
parameters = get_parameters("dynav_config.yaml", robot="rby1")
(args.output / "provenance.json").write_text(
    json.dumps(
        {
            "client_source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "eqa_config": dict(parameters.get("eqa")),
            "source": str(args.source),
            "scope": "client-payload offline tool-choice diagnostic, no robot action or success claim",
        },
        indent=2,
        default=str,
    )
)
_, client = build_graph_eqa_vlm_clients(parameters=parameters, device="cuda")
paths = []
for phase in ("find_object_agentic_trace", "find_recep_agentic_trace"):
    paths.append(sorted((args.source / "router_context" / phase).glob("call_*/request.json"))[-1])
with (args.output / "responses.jsonl").open("x") as stream:
    for path in paths:
        packet = json.loads(path.read_text())
        trace = path.parents[3] / f"{path.parent.parent.name}.jsonl"
        history = []
        for line in trace.read_text().splitlines():
            row = json.loads(line)
            if row.get("event") == "router_call" and row.get("router_call_id") == packet["metadata"]["router_call_id"]:
                break
            if row.get("event") is None and row.get("tool"):
                history.append(
                    {
                        key: row[key]
                        for key in (
                            "round",
                            "tool",
                            "ok",
                            "n_hypotheses",
                            "n_detections",
                            "outcome",
                            "error",
                            "status",
                            "nav_outcome",
                        )
                        if key in row
                    }
                )
        images = [Image.open(path.parent / entry["path"]).convert("RGB") for entry in packet["images"]]
        clean_system = packet["system_prompt"].split("Rules:", 1)[0] + (
            "Choose an available tool using the task, supplied images and actual previous tool outcomes. "
            "Distinguish search hypotheses from observations. Semantic recognition is not geometry or motion authorization."
        )
        for repeat in range(3):
            arms = ["original", "with_tool_history", "neutral_rules"]
            if repeat % 2:
                arms.reverse()
            for arm in arms:
                user = packet["user_text"]
                if arm == "with_tool_history":
                    user += "\nRecorded earlier tool results (not independent scene evidence):\n" + json.dumps(history)
                system = clean_system if arm == "neutral_rules" else packet["system_prompt"]
                random.seed(repeat)
                np.random.seed(repeat)
                torch.manual_seed(repeat)
                raw = client([user, *images], system_prompt=system, max_new_tokens=packet["metadata"]["max_new_tokens"])
                result = {
                    "request": str(path),
                    "arm": arm,
                    "repeat": repeat,
                    "system_prompt": system,
                    "user_text": user,
                    "image_hashes": [entry["pixel_sha256"] for entry in packet["images"]],
                    "raw": raw,
                    "parsed": parse_tool_calls_response(raw),
                    "history": history,
                }
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {"request": path.parent.parent.name, "arm": arm, "repeat": repeat, "parsed": result["parsed"]}
                    ),
                    flush=True,
                )
