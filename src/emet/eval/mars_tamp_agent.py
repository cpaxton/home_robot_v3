"""Agent selection over Mars public tools, separately gated from oracle success.

Uses the normal prompt/parser/dispatcher. Observed mode never falls back to GT;
current missing observed support is reported before spending model calls. An
existing inference endpoint is required: no model download or server launch.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emet.agent.loop import _dispatch_tool_calls
from emet.agent.prompt import AgentPromptBuilder, parse_tool_calls_response
from emet.agent.tools import Tool
from emet.eval.mars_tamp_control import CANONICAL_SUITE, require_public_readiness, run_case, sources


class AgentToolDriver:
    def __init__(self, infer, *, input_mode="observed", max_turns=12):
        if input_mode not in {"observed", "privileged"} or not 1 <= max_turns <= 32:
            raise ValueError("invalid agent test configuration")
        self.infer, self.input_mode, self.max_turns = infer, input_mode, max_turns
        self.selector_label = "llm_" + input_mode
        self.trace = []
        self.terminal = "not_started"

    def __call__(self, call, goals, registry):
        self.trace = []
        if self.input_mode == "observed":
            response = call("scene_tasks", robot="innate_mars")
            if response["status"] != "ok":
                self.terminal = "observed_capability_unavailable"
                self.trace.append({"phase": "capability_preflight", "response": response, "model_called": False})
                return response
        tools = {}
        for name in ("scene_tasks", "plan_pick_place", "execute_pick_place_plan"):
            tool = registry[name]

            def forward(_name=name, **kwargs):
                return json.dumps(call(_name, **kwargs))

            tools[name] = Tool(tool.name, tool.description, tool.parameters, forward, returns_info=tool.returns_info)
        system = str(
            AgentPromptBuilder(list(tools.values()), context={"tamp_inputs": self.input_mode, "manip_mode": "physical"})
        )
        instruction = (
            "Move the blue cube onto destination support. Its entire footprint must fit the world XY region "
            + json.dumps(goals["placement_region"])
            + ". Inputs: "
            + self.input_mode
            + ". "
            "Discover tasks, plan with placement_region, and execute only a returned plan_ref. "
            "Use one available tool call per turn. Do not invent handles or change goals. "
            "Stop after partial execution; no held-object recovery skill is available."
        )
        history = []
        self.terminal = "turn_budget_exhausted"
        last = None
        for turn in range(self.max_turns):
            prompt = instruction + "\nPrevious tool outcomes:\n" + "\n".join(history)
            began = time.monotonic()
            raw = self.infer(system, prompt)
            parsed = parse_tool_calls_response(raw)
            calls = parsed.get("tool_calls", [])
            row = {"turn": turn, "response": raw, "calls": calls, "model_wall_s": time.monotonic() - began}
            self.trace.append(row)
            if len(calls) != 1 or calls[0].get("name") not in tools:
                row["error"] = "expected_one_available_tool_call"
                history.append(json.dumps({"status": "error", "code": row["error"]}))
                continue
            _, results, _ = _dispatch_tool_calls(calls, tools, SimpleNamespace())
            row["results"] = results
            history.extend(results)
            try:
                last = json.loads(results[0].split("] ", 1)[1])
            except (IndexError, ValueError):
                self.terminal = "invalid_tool_result"
                return None
            if last["status"] == "partial":
                self.terminal = "partial_execution_requires_recovery"
                return last
            if calls[0]["name"] == "execute_pick_place_plan" and last["status"] == "ok":
                self.terminal = "execution_returned_ok_pending_physical_score"
                return last
        return last


def evaluate_agents(catalog, output, *, infer, input_mode, max_turns=12, model_id="not_recorded"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("refusing to overwrite agent evidence")
    manifest = {**sources(), "eval/mars_tamp_agent.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    rows = []
    for case in catalog["cases"]:
        for agent_seed in (31, 32, 33):
            random.seed(agent_seed)
            np.random.seed(agent_seed)
            driver = AgentToolDriver(infer, input_mode=input_mode, max_turns=max_turns)
            row = run_case(case, tool_driver=driver, input_mode=input_mode)
            row.update(
                agent_seed=agent_seed,
                agent_trace=driver.trace,
                agent_terminal=driver.terminal,
                physics_during_inference=False,
            )
            if driver.terminal == "observed_capability_unavailable":
                report = {
                    "status": "blocked",
                    "reason": driver.terminal,
                    "complete": False,
                    "cases": [],
                    "planned_trials": 3 * len(catalog["cases"]),
                    "completed_trials": 0,
                    "capability_diagnostic": row,
                    "source_hashes": manifest,
                    "model_id": model_id,
                    "input_mode": input_mode,
                    "observed_agent_qualified": False,
                    "hardware_qualified": False,
                }
                (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                return report
            rows.append(row)
            current = {**sources(), "eval/mars_tamp_agent.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            if current != manifest:
                raise RuntimeError("agent execution sources changed during evaluation")
            report = {
                "scope": "LLM-selected public tools; independently stepped simulator; " + input_mode + " inputs",
                "input_mode": input_mode,
                "model_id": model_id,
                "source_hashes": manifest,
                "cases": rows,
                "complete": len(rows) == 3 * len(catalog["cases"]),
                "successes": sum(x["success"] for x in rows),
                "remote_sampling": "server defaults; recorded local seeds do not set remote sampler",
                "observed_agent_qualified": False,
                "hardware_qualified": False,
            }
            tmp = output / "report.tmp"
            tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            tmp.replace(output / "report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", choices=["observed", "privileged"], default="observed")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--remote-model", required=True)
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args()
    require_public_readiness(json.loads(args.oracle_report.read_text()))
    # Deliberately construct the model client only after oracle admission.
    from emet.agent.loop import _call_llm
    from emet.llms.openai_client import OpenaiClient

    client = None

    def infer(system, prompt):
        nonlocal client
        if client is None:
            client = OpenaiClient(system, model=args.remote_model, base_url=args.base_url)
            client._openai = client._openai.with_options(timeout=120, max_retries=0)
        client.reset()
        return _call_llm(client, prompt, None, False, reset_context=True)[0]

    evaluate_agents(
        json.loads(CANONICAL_SUITE.read_text()),
        args.output,
        infer=infer,
        input_mode=args.inputs,
        max_turns=args.max_turns,
        model_id=args.remote_model,
    )


if __name__ == "__main__":
    main()
