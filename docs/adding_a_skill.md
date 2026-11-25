# How to Add a New Agent Skill

This is the **current** way to give the LLM agent a new tool (a *skill*). If you are
looking at [`adding_a_new_task.md`](adding_a_new_task.md), that page documents the
older executor/task model and is kept only for historical reference.

## The model in one paragraph

The agent exposes tools to the LLM. A skill is declared **once** as a
[`SkillSpec`](../src/emet/agent/skills/specs.py) — name, mode(s), description, a
JSON-Schema `parameters` object, and a `returns_info` flag — and that declaration is
the single source of truth. Two orchestrators assemble different tool packs from the
same library:

| Mode | Orchestrator | Tool pack assembled by |
|------|--------------|------------------------|
| `CHAT` | Discord / terminal agent | `emet.agent.tools.build_chat_tools` |
| `EQA_EPISODE` | scored GraphEQA verify/explore loop | `emet.memory.graph_eqa.agentic_tools.build_agentic_eqa_tools` |

See [`AgentMode`](../src/emet/agent/skills/modes.py) and
[`build_skill_pack`](../src/emet/agent/skills/registry.py).

The pieces:

- **`SkillSpec`** (`src/emet/agent/skills/specs.py`) — declarative metadata.
- **`Tool`** (`src/emet/agent/tools.py`) — the runtime object: name, description,
  parameters, a callable `func`, an optional `executor_commands` bridge, and
  `returns_info`. `Tool.schema()` emits an OpenAI-style function schema.
- **`ToolOutcome`** (`src/emet/agent/tool_outcome.py`) — the normalized result shape
  (`ok` / `status` / `note` / `tool` / `payload`) both loops report.

## Skill inventory

`CHAT_EXCLUSIVE_TOOL_NAMES` and `EQA_EXCLUSIVE_TOOL_NAMES` are **derived automatically**
from the spec tuples below — do not edit them by hand.

### CHAT skills (`CHAT_SKILL_SPECS`, 30 tools)

| Skill | Bound via | `returns_info` |
|-------|-----------|----------------|
| `query_memory` | `func` | ✓ |
| `send_image` | `func` | ✓ |
| `explore` | `func` | ✓ |
| `navigation_diagnostics` | `func` | ✓ |
| `send_map_snapshot` | `func` | ✓ |
| `pick_place` | `func` | ✓ |
| `plan_pick_place` | `func` | ✓ |
| `execute_pick_place_plan` | `func` | ✓ |
| `scene_tasks` | `func` | ✓ |
| `find_objects` | `func` + executor (`find`) | ✓ |
| `describe_scene` | `func` | ✓ |
| `say` | `func` + executor (`say`) | — |
| `wave` | `func` + executor (`wave`) | — |
| `nod_head` | `func` + executor (`nod_head`) | — |
| `shake_head` | `func` + executor (`shake_head`) | — |
| `avert_gaze` | `func` + executor (`avert_gaze`) | — |
| `go_home` | `func` + executor (`go_home`) | — |
| `scan_environment` | `func` | ✓ |
| `rotate_base` | `func` | ✓ |
| `face_toward` | `func` | ✓ |
| `move_forward` | `func` | ✓ |
| `take_picture` | `func` | ✓ |
| `take_ee_picture` | `func` | ✓ |
| `aim_arm_at` | `func` | ✓ |
| `hand_over` | `func` + executor (`hand_over`) | — |
| `query_scene_graph` | `func` | ✓ |
| `list_scene_relations` | `func` | ✓ |
| `send_object_image` | `func` | ✓ |
| `ask_user` | `func` | ✓ |
| `quit` | `func` + executor (`quit`) | — |

The **executor-bridge** skills (`say`, `find_objects`, `hand_over`, `go_home`, the
four emotes, `quit`) are thin wrappers over the legacy executor: they also set
`executor_commands` so the loop can route the call to the old `controller/task/*`
executor. See [Binding patterns](#binding-patterns) below.

### EQA skills (`EQA_SKILL_SPECS`, 8 tools)

EQA tool names are **trace-stable — do not rename them** (traces and paper tooling
depend on the names).

| Skill | Terminal | `returns_info` |
|-------|----------|----------------|
| `inspect_graph` | — | ✓ |
| `explore_frontier` | — | — |
| `investigate` | — | — |
| `navigate_to_obs` | — | — |
| `look_around` | — | — |
| `verify_siglip` | — | ✓ |
| `submit_answer` | `eqa_answer_only` | — |
| `finish` | `eqa_explore_only` | — |

## Adding a CHAT skill

Two edits, then run the tests. Pick the template below that matches the kind of
skill you are adding.

### Step 1 — declare the `SkillSpec`

Add one entry to `CHAT_SKILL_SPECS` in `src/emet/agent/skills/specs.py`. For example,
a no-arg fire-and-forget action:

```python
SkillSpec(
    name="my_action",
    modes=frozenset({AgentMode.CHAT}),
    description="What this action does and when the LLM should call it.",
    parameters=_NO_PARAMS,
    returns_info=False,
),
```

Notes:

- `CHAT_EXCLUSIVE_TOOL_NAMES` updates itself from `CHAT_SKILL_SPECS`; nothing else to edit.
- `returns_info=True` means the tool result is fed back to the LLM for a follow-up
  response (e.g. `query_memory`, `describe_scene`). Leave it `False` for fire-and-forget
  actions (speech, emotes, `hand_over`, `quit`).
- Use `_NO_PARAMS` for a no-argument tool.

### Step 2 — bind the `Tool`

In `build_chat_tools(context)` in `src/emet/agent/tools.py`, define a `func` closure
over `context` (or a lambda) and `tools.append(Tool(...))` with the **same name** as the
spec.

## Binding patterns

Every real skill is one of these three shapes. Use the verbatim examples as templates.

### 1. Direct `func` (parameterized, returns info)

The common case: a skill that computes from `context` and returns text the LLM should
see. Example — `aim_arm_at` (abridged `func` body):

```python
def aim_arm_at(object_label: str):
    # ... localize + kinematic EE aim; returns a ToolOutcome
    return result.to_tool_outcome()

tools.append(
    Tool(
        name="aim_arm_at",
        description=(
            "Point the arm / wrist camera at a named object (localize + kinematic EE aim when "
            "available). On success you may call take_ee_picture once. If aim fails or is "
            "unavailable, prefer face_toward + describe_scene (head camera) instead of claiming "
            "a closer look."
        ),
        parameters={
            "type": "object",
            "properties": {
                "object_label": {
                    "type": "string",
                    "description": "Object or region to aim at (e.g. cables, red cup).",
                }
            },
            "required": ["object_label"],
        },
        func=aim_arm_at,
        returns_info=True,
    )
)
```

### 2. Executor bridge, parameterized

A skill that delegates to the legacy executor with an argument. Example — `say`:

```python
tools.append(
    Tool(
        name="say",
        description="Speak text to the user via TTS (text-to-speech). Use to announce what you are doing.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The message to speak aloud."}},
            "required": ["text"],
        },
        func=lambda text: _exec("say", text),
        executor_commands=lambda args: [("say", args.get("text", ""))],
    )
)
```

`find_objects` uses the same shape (`executor_commands=lambda args: [("find", args.get("text", ""))]`).

### 3. Executor bridge, no-arg

A skill with no arguments that dispatches a fixed executor command. Example —
`hand_over` (the canonical legacy-bridge skill):

```python
tools.append(
    Tool(
        name="hand_over",
        description="Hand the held object to a person (find person, navigate, extend arm).",
        parameters=_NO_PARAMS,
        func=lambda: _exec("hand_over", ""),
        executor_commands=_simple_exec_mapping("hand_over"),
    )
)
```

`go_home`, `wave`, `nod_head`, `shake_head`, `avert_gaze`, and `quit` follow the same
no-arg bridge shape (`quit` uses `func=lambda: None` since it needs no result).

### Details that matter

- **`func` vs `executor_commands`.** `func` is the direct implementation. When a skill
  is a thin wrapper over the legacy executor, also pass `executor_commands`, and the
  loop will map the call to an executor `(command, args)` pair via `Tool.to_executor()`.
  When `executor_commands` is omitted (the common case), `func` runs directly.
- **The spec is authoritative.** At the end of `build_chat_tools`, a drift check raises
  `RuntimeError` if the bound tool names differ from `CHAT_SKILL_SPECS`, and every
  tool's `description` / `parameters` / `returns_info` are overwritten from the spec.
  So keep the `Tool` constructor values consistent with the spec, but know that a typo
  there cannot silently diverge the prompt.
- **`_exec(cmd, args)`** dispatches to the legacy executor; `_simple_exec_mapping(cmd)`
  builds a no-arg `executor_commands` mapping.

### Step 3 — run the tests

```bash
uv run pytest src/test/agent/test_skill_packs.py src/test/agent/test_tool_outcome.py
```

`test_chat_tools_match_skill_specs` fails if your `Tool` metadata drifts from the spec;
`test_chat_pack_excludes_eqa_only_tools` keeps the CHAT and EQA surfaces disjoint.

## Adding an EQA skill

1. Add a `SkillSpec` to `EQA_SKILL_SPECS` in `specs.py`. Use `eqa_answer_only=True` for
   terminal answer tools (`submit_answer`) or `eqa_explore_only=True` for
   explore-terminal tools (`finish`); `eqa_specs_for_submode` filters them per submode.
2. Bind it in `build_agentic_eqa_tools` (`src/emet/memory/graph_eqa/agentic_tools.py`).
3. Keep `EQA_EXCLUSIVE_TOOL_NAMES` (derived) in sync by declaring the spec; then
   `test_eqa_answer_pack_names_match_registry` and
   `test_build_skill_pack_eqa_matches_agentic_builder` enforce parity.

## Enforcement summary

| Mechanism | What it prevents |
|-----------|------------------|
| Drift check in `build_chat_tools` | Tool names diverging from `CHAT_SKILL_SPECS` |
| `test_skill_packs.py` | Metadata drift, CHAT/EQA name collisions, pack-vs-builder mismatch |
| `ToolOutcome` | Divergent result shapes between the two loops |

## Related

- [`attempt_ledger.md`](attempt_ledger.md) — how tool outcomes are written into the
  graph-memory action ledger.
- [`apis/tamp.md`](apis/tamp.md) — the versioned manipulation tool API
  (`scene_tasks` / `plan_pick_place` / `execute_pick_place_plan` / `pick_place`).
- [`adding_a_new_task.md`](adding_a_new_task.md) — the legacy executor/task model
  (historical).
