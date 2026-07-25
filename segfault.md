# Segfault investigation log

Last updated: 2026-07-25 11:10 EDT

## Recover first (do this before anything else)

```bash
tail -n 12 ~/runs/emet/STATUS.log   # state + literal next command
ls -l ~/runs/emet/latest            # newest OUT dir
uv run emet jobs                    # is a managed job still alive?
```

If `STATUS.log` says `RUNNING` / `OK` and `emet jobs` still shows the job, **do not relaunch** — the detached job may still be fine after the Cursor agent died.

If `STATUS.log` says `CRASH` / `EGL` / `EXIT` / `BLOCKED`, follow its `next:` line. The copy-paste resume for this bal-32 crash is also below under **Active experiment**.

Do **not** run `emet eval kill-stale` while a managed HM-EQA job is still intended to live.

## Active experiment

| Field | Value |
|-------|-------|
| Job | `20260725_101522_3b3b11` (`hmeqa-bal32-rerun`) — **FAILED** exit 139 |
| OUT | `~/runs/emet/hmeqa_agentic_bal32_20260725_101519` |
| Commit | `e7db059` |
| Arms | classic, then agentic |
| IDs | `2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84` |
| Coverage figure QIDs | `15,28,47` |
| Progress at abort | **5/64** (classic q2, q6, q8, q11, q12 done; q14 crashed) |
| Abort | classic **q14**, ~10:39 EDT, phase `native-crash` |
| Capsule | `$OUT/native_crash_classic_q14.log` |
| Orchestrator log | `$OUT/orchestrator.log` |
| Progress JSON | `$OUT/progress.json` → `{"units_done":5,"units_total":64,"phase":"native-crash","current_id":"14"}` |
| Status seed | `~/runs/emet/STATUS.log` (backfilled CRASH record; see `next:`) |

GPU after abort (checked ~10:53): idle, Xorg/gnome-shell only, ~40C / ~24W. No Xid / ECC / OOM.

### Exact resume (affinity off CPUs 8+9)

```bash
cd ~/src/home_robot_v4
uv run emet eval status
uv run emet eval diagnose
NEED_MIB=12000 uv run emet eval wait
# Exclude logical CPUs 8+9 (same P-core, 6.0 GHz max):
uv run emet jobs run --name hmeqa-bal32-aff --need-mib 12000 -- \
  env EMET_ALLOW_SDPA_ATTN=1 EMET_EQA_TRACE=1 RESUME=1 \
  ARMS=classic,agentic \
  HOLDOUT_IDS=2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84 \
  COVERAGE_QIDS=15,28,47 \
  taskset -c 0-7,10-31 \
  ./scripts/run_hmeqa_agentic_h2h.sh \
  /home/cpaxton/runs/emet/hmeqa_agentic_bal32_20260725_101519
```

`RESUME=1` skips non-empty `${arm}_q*.jsonl` (q2/q6/q8/q11/q12 stay scored). q14 was empty → will re-run.

### After resume is launched

```bash
tail -n 12 ~/runs/emet/STATUS.log
uv run emet jobs status <NEW_JOB_ID>
uv run emet jobs logs <NEW_JOB_ID> --tail 40
```

## Two distinct crashes this morning

### A. Cursor CLI `agent` — illegal instruction (10:30 EDT)

User-visible:

```
Fatal error in , line 0
Check failed: fixed_size_above_fp + (stack_slots * kSystemPointerSize) …
… v8::internal::Deoptimizer::Deoptimizer …
Illegal instruction (core dumped)
```

Binary: `~/.local/bin/agent` → Cursor `2026.07.23-e383d2b`.

Kernel record:

```
[Sat Jul 25 10:30:04 2026] traps: MainThread[1964009] trap invalid opcode
  ip:2e76edc sp:7ffc28e273b0 error:0 in node[f33000+27cf000]
```

This is the Cursor/Node/V8 process itself, not Habitat/`emet`. Same class of
fault (invalid opcode) as the MuJoCo `_structs` trap at 09:06.

### B. HM-EQA classic q14 — Python SIGSEGV mid-VLM decode (10:39 EDT)

Orchestrator aborted after one native crash (by design). Episode had loaded
Habitat + Qwen3-VL-8B int4, finished look-around, then crashed during the first
multimodal `query_answer` decode (~65 tokens in).

Faulthandler top of current thread:

- `transformers…/modeling_qwen3_vl.py:133` `rotate_half`
- → `apply_rotary_pos_emb` → attention forward → `generate`
- → `emet.llms.qwen3_vl_client._generate_ids` / `generate_multimodal`

No matching kernel segfault/trap line around 10:39 (unlike the Cursor agent
crash). Faulthandler alone; `timeout: the monitored command dumped core`.

## Evidence collected

The kernel journal since the 2026-07-24 boot contains many unrelated native
process failures:

- `emet`: null/invalid instruction addresses, MuJoCo `_structs`, and libc.
- `python`: repeated faults in `libcuda.so.595.84` and MuJoCo `_structs`.
- `cicc`: two executable-address faults.
- 2026-07-25 09:06: Python first trapped `invalid opcode` in MuJoCo `_structs`,
  then another Python process segfaulted at an executable-looking address.
- 2026-07-25 10:30: Cursor `agent` (Node) `invalid opcode` (above).
- 2026-07-25 10:39: HM-EQA episode SIGSEGV in Qwen3-VL RoPE path (userspace
  faulthandler only; no kernel line found).

Every recorded **kernel** native crash before 10:30 was running on logical CPU
**8 or 9**. Those are the two SMT threads of the **same** i9-14900KF P-core
(`lscpu`: CORE 4), and that core is one of the two configured for a **6.0 GHz**
maximum (`cpuinfo_max_freq`). Neighboring cores max at 5.7 GHz; E-cores 4.4.
Processes are allowed on CPUs 0–31, so this concentration is not explained by
task affinity.

Apport has crash reports for earlier Python failures, including
`.venv/bin/python -m emet.cli install robocasa --help` with MuJoCo loaded and a
full `pytest -m "not sim"` process. Core capture is disabled in the launching
shell (`ulimit -c` is 0); Apport is the configured core handler.

There are no kernel MCE/EDAC, NVIDIA Xid, OOM, or thermal-fault reports. Their
absence does not rule out marginal CPU voltage/boost stability. A later
`perf: interrupt took too long` at 10:43 is a soft latency note, not proof.

## Working diagnosis

Highest-probability cause: **host CPU instability on the favored 6.0 GHz
P-core (logical 8/9)**, exposed by native CUDA / MuJoCo / V8 / compiler
workloads — not a single deterministic HM-EQA Python bug.

Supporting pattern:

- Random fault addresses across `libcuda`, MuJoCo `_structs`, libc, `cicc`,
  and now Cursor Node/V8.
- Exclusive kernel placement on one physical core’s SMT pair.
- Invalid-opcode traps (corrupt instruction stream / speculative execution
  artifacts) in addition to ordinary segfaults.

The q14 crash site (`rotate_half` during CUDA decode) is a plausible place for
a corrupted CPU-side tensor op or a GPU→host sync path to surface; it does not
by itself prove a transformers bug.

Immediate containment: relaunch the managed bal-32 resume under
`taskset -c 0-7,10-31` so episode workers inherit affinity away from CPUs 8/9.
Longer-term host checks: BIOS defaults / Intel baseline power, current BIOS and
microcode, focused CPU/RAM stress after the experiment.

## Prior distinct issue

The earlier `pytest -m "not sim"` Open3D offscreen-renderer SIGSEGV was a real,
separate deterministic bug. It was fixed by validating `.sens` input before
constructing the renderer and by gating renderer tests. Do not conflate that
fixed test bug with the cross-library CPU8/9 crash cluster above.

## How to keep this file useful next time

Before any risky GPU / sim action (Habitat episode, MuJoCo serve, VLM load,
`emet eval kill-stale`, resume, affinity experiment), append enough that a cold
session can recover from **only** this file + `STATUS.log`:

1. Job id, OUT path, commit, arms, ID list, progress.
2. Exact launch / resume command (copy-pasteable).
3. What just failed (signal, capsule path, one-line stack top).
4. What **not** to do (`kill-stale`, hard-kill Habitat, relaunch while job alive).

Prefer writing through `scripts/status_log.sh` so `tail ~/runs/emet/STATUS.log`
carries the same `next:` instruction; keep this markdown as the durable
investigation narrative.
