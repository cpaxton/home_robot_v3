# Native TAMP composition and CHAT outcome integration

This follow-up is stacked on #209. It changes CHAT outcome accounting and batch
control, plus the native CPU evaluation harness; it does not change motion
planning, controllers, collision tolerances, or input provenance.

## Integration defect and fix

`ToolOutcome.coerce` previously interpreted TAMP JSON as free-form text. An
`error` or `partial` envelope generally lacked the prose substrings used to detect
failure, so the attempt ledger could record success. The dispatcher also ran
subsequent calls from the same model-generated batch without returning that
failure to the model first.

CHAT now reads the versioned envelope, preserves its code/data/recovery, and
rejects malformed TAMP outcomes. A failed TAMP call stops the remainder of the
batch and emits a JSON `batch_stopped` result listing unattempted tools. The
session remains alive for a later inspection/replanning turn. This is not a
persistent safety interlock, and it does not establish that an LLM will choose
correct recovery. EQA and other legacy tool outcomes retain existing behavior.

The tool descriptions now explain candidate-versus-feasible-task semantics,
input/execution provenance, and rediscovery after execution. The CPU harness
uses the actual CHAT dispatcher and public registry; no LLM selects its calls.

## Development diagnostics (not acceptance denominators)

The initial harness changes were uncommitted on `ed42dc4a`; their emitted HEAD
field alone does not identify the changed harness. Keep these diagnostics
separate from the frozen validation below:

- `20261013_174953_a9fd13`, `native-composition-r1`: box and cylinder each completed
  two consecutive transfers, without resets (2/2 episodes). Direct tool calls.
- `20261013_175203_17d24d`, `native-composition-r2`: two objects separated 0.20 m
  laterally. Box transfer passed; cylinder pregrasp planning failed with
  `ik_failed`. No second execution was attempted.
- `20261013_175257_39b331`, `native-composition-r3`: objects separated 0.15 m in
  depth. Box transfer passed; cylinder pregrasp planning failed with `ik_failed`.
  No second execution was attempted.

The last candidate's IK failure is not a proof of global infeasibility, nor proof
that a blocker must be moved. These fixtures lack a validated complete reference
sequence and must not enter an LLM success denominator yet.

## Next acceptance boundary

Repeated transfers test controller state, refreshed planning, and tool integration;
they do not constitute a semantic rearrangement benchmark. The next planner PR
must validate a genuinely solvable multi-object reference sequence, with explicit
source/target regions and final goal predicates. First distinguish initial grasp
infeasibility from an obstruction created by an earlier placement. Retain
per-candidate rejection reasons and consider alternative grasp/place combinations
before claiming that another high-level action is required.

Only after a complete native reference sequence passes should we compare scripted
selection, task search, and `run agent` LLM selection. Keep missing geometry,
partial execution, stale handles, and unknown command outcomes as separate fault
controls. Observed geometry, exploration, live ZMQ composition, and native
articulation remain separate acceptance gates.

## Frozen validation at `9b44dd7f`

The managed CPU suite `20261013_175433_17acd0` runs through the actual CHAT
registry and dispatcher. Under `/home/cpaxton/runs/emet/native-composition-final/`:

- `repeated/`: **4/4 two-step episodes passed** (two fresh episodes per shape,
  eight transfers total). Each episode retains physics/controller state between
  transfers, rediscovers before each plan, and verifies released support after
  successful execution. These repeated-object controls are distinct from the
  earlier 4/4 live single-transfer result; do not pool their denominators.
- `two-objects/`: the forward-order depth-separated fixture is retained as a
  diagnostic, not an admitted agent benchmark: **0/1 complete sequences**, with
  the box transfer successful and cylinder pregrasp planning rejected. The managed
  job exits 1 for this unsolved task; this is a scored planning failure, not an
  infrastructure crash.

Reverse-order job `20261013_175545_006b09`, artifacts in
`/home/cpaxton/runs/emet/native-composition-order/`, rejects the cylinder at
pregrasp planning before either transfer. Thus the fixture is not failing solely
because of the previous box placement. Neither order supplies a passing reference
sequence. Budgeted planning rejection is not a proof that no physical solution
exists.

**99 targeted regression tests passed, no failures/skips**: CHAT dispatch,
versioned outcome handling, attempt accounting, manipulation sequences, skill
pack separation, native adapter and API contracts. Ruff passes on all changed
Python files. Tests include malformed JSON, misleading success prose, partial
execution, stale scene, batch cancellation and a subsequent recovery turn.
