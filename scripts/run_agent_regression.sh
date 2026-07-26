#!/usr/bin/env bash
# Copyright (c) Chris Paxton 2026
# Fast no-sim pack: Discord / Herman / emet run agent must stay green while
# Graph-Driven / agentic EQA work lands. Prefer this over full pytest.
#
# Usage (from repo root):
#   ./scripts/run_agent_regression.sh
#   uv run emet test -q …   # same paths; see below
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Agent loop + tools (Discord image stash, query_memory, dispatch)
# CLI defaults (--discord/--no-discord, robot/memory)
# Herman connection preset resolve
# Classic GraphEQA answer path (agentic OFF by default)
# Memory backend smoke (SVM / DynaMem / GraphEQA adapters)
exec uv run emet test -q \
  src/test/agent/test_agent_prompt_and_tools.py \
  src/test/agent/test_dispatch_tool_calls.py \
  src/test/agent/test_run_agent_loop_mock.py \
  src/test/agent/test_call_llm.py \
  src/test/agent/test_thinking_status.py \
  src/test/agent/test_dynagraph_import_cycle.py \
  src/test/agent/test_manual_find_command.py \
  src/test/cli/test_run_agent_defaults.py \
  src/test/app/test_stream_dynav_resolve.py \
  src/test/controller/test_graph_eqa_answer_only.py \
  src/test/eval/test_agentic_eqa_verification.py \
  src/test/memory/test_memory_backends_smoke.py \
  src/test/memory/test_graph_eqa_beliefs.py \
  -m 'not sim' \
  "$@"
