# TAMP API readiness implementation (2026-10-05)

## Disk cleanup

Before cleanup, the shared filesystem had about 436 MB free. The installed pip
cache command removed 88 legacy cache files, reclaiming about 282 MB. Its older
version did not recognize the HTTP-v2 cache. After verifying that directory and
all entries were not symlinks, cleanup removed only
`/home/cpaxton/.cache/pip/http-v2`, reclaiming **24,483,827,712 bytes**. Immediately
afterward, free space was **25,233,096,704 bytes**. Other active jobs can change it.
No datasets, installed environments, worktrees or experiment evidence were deleted.
The cleanup target was met without removing experiment source checkouts.

## Branch assessment

At inspection, local HEAD was `ce4c565c`; upstream `feat/tamp-agent-execution`
was `4a65277c`. PR #178 is open against main. Main is still `a9a5f1d1`; the parent
physical-acceptance branch is `2a0c5ec1`. The follow-up includes its parent branch
changes, so review/merge order matters. The unrelated lockfile edit observed during
planning was absent at implementation start; this task did not change `uv.lock`.

## Implementation and checks

The [canonical API](../apis/tamp.md) defines the consistent JSON contract and
recovery actions for all four tools. New guards cover server boot identity,
finite full poses, endpoint revalidation, stale measured state, and failed
placement submotions. The incorrect 1-rad shoulder-stall diagnosis was corrected
against the original log. No object-specific runtime behavior was introduced.

The full regression run passed **206 tests**, including placement submotion
regressions. Ruff and whitespace checks pass. Live readiness is **not yet established**.

## Live protocol

`scripts/run_tamp_api_readiness.sh OUTPUT ORIGINAL_FROZEN_REGISTRY` runs only
inside `emet jobs --cpu-safe --gpu-exclusive`. Use an immutable checkout.
It records source and the predeclared three IDs; runs the actual CHAT tool smoke;
then freshly admits scene00, scene02 and scene12 cleanup0 and executes three
fresh-process GT/MCTS latch repeats of each admitted fixture. Failed admission
is retained and blocks readiness. The CHAT smoke and MCTS matrix are separate
measurements; the matrix is not nine learned-agent trials.

Readiness requires a passing API smoke, three admitted cases, and 9/9 successful
full cleanup tasks. Final machine-readable output is `readiness.json`. Historical
scores remain unchanged. The shared GPU queue already contains another agent's
running simulator and queued evaluation; those jobs must not be interrupted.
