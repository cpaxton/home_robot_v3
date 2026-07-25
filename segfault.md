# Segfault investigation log

Last updated: 2026-07-25 17:12 EDT

### Holdout-8 updated agentic (launching)

| Field | Value |
|-------|-------|
| Job | `20260725_171348_116e60` (`hmeqa-holdout8-owlv2`) |
| OUT | `~/runs/emet/hmeqa_agentic_holdout8_20260725_owlv2` |
| IDs | `15,56,65,68,79,88,104,105` (paper holdout-8) |
| Arms | **agentic only** (updated evidence policy + OWLv2) |
| Flags | `--agentic-verifier owlv2 --require-verified --no-agentic-router` |
| Paper ref | classic 5/8, old agentic 8/8 (unverified fallbacks — not the bar) |
| Monitor | `uv run emet jobs status 20260725_171348_116e60` |
| Resume | `uv run emet hmeqa resume /home/cpaxton/runs/emet/hmeqa_agentic_holdout8_20260725_owlv2` |
| Do not | `kill-stale` while job live; hard-kill Habitat; second GPU job |

---

### Cursor agent death mid Graph-Driven plan (2026-07-25 ~17:04 EDT)

| Field | Value |
|-------|-------|
| Class | **A** — Cursor/`agent` process died (not Habitat/`emet` SIGSEGV; no kernel line this boot) |
| Plan | `Graph-Driven EQA Agent-da93b7e0.plan.md` |
| Session | `da93b7e0` — implement re-sent 16:19 and again 17:00 after prior agent exits |
| Last work | unit-test fix (`run()` resets policy state) + HMEQA question load check — **CPU tests only** |
| Kernel since 13:30 reboot | **0** segfault / invalid-opcode lines |
| GPU | idle (~23.8 GiB free); **no** active `emet jobs` |

**Detached jobs that finished despite agent EXIT markers:**

| Job / OUT | Result |
|-----------|--------|
| Bakeoff `~/runs/emet/hmeqa_verifier_bakeoff` | Done; 24 frames; **`n_labeled=0`** so PR curves empty. SigLIP1 dense ~0.08–0.11; OWLv2 det max ~0.34 |
| Graph probe `~/runs/emet/hmeqa_graph_probe_20260725` (`20260725_163432_ffb43d`) | **4/4 scored, failed=0**; agentic **1/4** (only q56 correct). q17/q18 empty predict (abstain-ish); q12 confident wrong |

**Do not** relaunch Habitat. **Do not** `kill-stale`. Resume: continue Graph-Driven todos (fix evidence labels / policy gaps / persistence) from code already in the working tree; STATUS `next` was summarize probe.

```bash
cd ~/src/home_robot_v4
bash scripts/status_log.sh tail
uv run emet jobs   # should be empty
uv run emet hmeqa summarize /home/cpaxton/runs/emet/hmeqa_graph_probe_20260725
```

---

### Verify-gate calib (2026-07-25 15:39)

Offline on saved `h2h_agentic` RGB (q12/17/18/56): full-frame max ~0.01–0.12; dense patch max ~0.12–0.14 — **never ≥0.21**.
So image-space PRESENT must use ~0.12 (MaskSigLIP matching default), not DynaMem voxel 0.21.
Probe: `hmeqa-verify-probe` with `REQUIRE_VERIFIED=1` + dense/voxel upgrade + no same-view reverify.

## STOPPED bal-32 — verify gate broken (investigating)

| Field | Value |
|-------|-------|
| Cancelled job | `20260725_141057_c18716` |
| OUT (progress saved) | `~/runs/emet/hmeqa_agentic_bal32_20260725_101519` |
| Frozen progress | classic **14/32**, agentic **7/20** (52/64 units) |
| Resume later | `uv run emet hmeqa resume /home/cpaxton/runs/emet/hmeqa_agentic_bal32_20260725_101519` |

**Why stopped:** agentic never hits SigLIP `PRESENT` (0/103 verifies on bal-32; max full-frame sim ~0.12 vs bar 0.21/0.28). All submits were `picked_by=fallback`. Root cause: full-frame `encode_image` cosine ≠ DynaMem per-point feature space that 0.21 was calibrated on. Holdout 7/8 also had `verified=0/8` — sample win was unconfirmed guesses.

**Probe now:** job `20260725_153447_3bdbf7` (`hmeqa-verify-probe`)
OUT `~/runs/emet/hmeqa_verify_probe_20260725_153444` — QIDs `12,17,18,56`, agentic-only,
`EMET_EQA_AGENTIC_REQUIRE_VERIFIED=1` + voxel-obs PRESENT upgrade.
Monitor: `uv run emet jobs status 20260725_153447_3bdbf7`

---


### Verify-gate experiment (active)

| Step | Result |
|------|--------|
| Offline calib | `~/runs/emet/hmeqa_verify_calib_20260725_153823` — dense max ~0.12–0.14 on saved frames, never ≥0.21 |
| Image PRESENT bar | `SIGLIP_IMAGE_PRESENT_THRESHOLD=0.12` — **high-recall / high-FP** (not a precision oracle); see `docs/experiments/agentic_scale.md` § SigLIP role |
| Policy | explore → move → verify once → assess; `REQUIRE_VERIFIED=1`; no same-view reverify |
| Habitat probe | job `20260725_154049_fa93bd` OUT `~/runs/emet/hmeqa_verify_probe_20260725_154046` QIDs 12,17,18,56 |
| Bal-32 | frozen classic 14/32 agentic 7/20 — do not resume until probe looks sane |

Note: HM-EQA has no `voxel_map.pkl` scene caches (those are Robocasa/Molmo). Tuning used saved episode RGB under `~/.cache/habitat_eqa/episodes/h2h_agentic_q*`.

## Recover first (do this before anything else)

```bash
cd ~/src/home_robot_v4                 # owning checkout — not v2/v3
bash scripts/status_log.sh tail        # state + literal next command
bash scripts/status_log.sh latest      # newest OUT for this checkout
uv run emet jobs                       # is a managed job still alive?
```

Do **not** `tail ~/runs/emet/STATUS.log` — that flat path is shared with sibling trees
(`home_robot_v2` / `v3` / `v4`) and would mix recovery instructions across agents.
Per-repo log: `~/runs/emet/status/home_robot_v4/STATUS.log`.

If `STATUS.log` says `RUNNING` / `OK` and `emet jobs` still shows the job, **do not relaunch** — the detached job may still be fine after the Cursor agent died.

If `STATUS.log` says `CRASH` / `EGL` / `EXIT` / `BLOCKED` / host-freeze, follow its `next:` line.

Do **not** run `emet eval kill-stale` while a managed HM-EQA job is still intended to live.

## Active experiment

| Field | Value |
|-------|-------|
| Last job | `20260725_141057_c18716` (`hmeqa-bal32-harden`) — **RUNNING** (resume after host freeze) |
| Prior jobs | `20260725_114158_32d3a0` (freeze at q48); `20260725_101522_3b3b11` (exit 139 on q14) |
| OUT | `~/runs/emet/hmeqa_agentic_bal32_20260725_101519` |
| Commit (at resume) | `6426bab` + local harness/CLI harden (uncommitted) |
| Arms | classic, then agentic |
| IDs | `2,6,8,11,12,14,15,16,17,18,21,25,27,28,29,31,32,33,34,38,39,40,41,43,44,47,48,49,57,76,80,84` |
| Coverage figure QIDs | `15,28,47` |
| Progress at resume | **26/64** classic scored; empty q48 will re-run |
| Policy | `NATIVE_CRASH_POLICY=skip`, `NATIVE_CRASH_STREAK_ABORT=2`, `EPISODE_COOLDOWN_SEC=30`, cpu-safe `0-7,12-31` |

GPU after reboot (~13:33): idle, Xorg/gnome-shell only, ~37C / ~18W. No Xid / ECC / OOM in journal (journal simply stops).

### Exact resume (affinity now baked into H2H — exclude ALL 6 GHz cores)

```bash
cd ~/src/home_robot_v4
uv run emet eval recover --need-mib 12000
uv run emet hmeqa resume /home/cpaxton/runs/emet/hmeqa_agentic_bal32_20260725_101519
# equivalents: emet hmeqa status ; emet jobs
```

`RESUME=1` skips non-empty `${arm}_q*.jsonl` (26 classic episodes stay scored). q48 is empty → will re-run.
Default crash policy is **skip** with **streak-abort=2** (early exit if consecutive native crashes).

### After resume is launched

```bash
bash scripts/status_log.sh tail
uv run emet jobs status <NEW_JOB_ID>
uv run emet jobs logs <NEW_JOB_ID> --tail 40
```

## Three distinct failures today

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

This is the Cursor/Node/V8 process itself, not Habitat/`emet`.

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

### C. Host hard freeze on classic q48 (13:02 EDT) — **this crash**

- Affinity resume (`hmeqa-bal32-aff`) had completed classic q14→q47 under
  `taskset -c 0-7,10-31` (21 episodes after the morning q14 SIGSEGV).
- q48 started 13:02:17; by planning step 4 / VLM decode ~161 tokens the
  machine hard-froze. Journal ends at 13:02:41 with only warp/tailscale noise —
  **no** oops, soft lockup, NMI, Xid, or OOM.
- Forced reboot ~13:30. Job marked failed (`exited without DONE`).
- Root cause class: same **host CPU instability** as morning, plus **incomplete
  turbo exclusion** (CPUs 10–11 still allowed). Possible contributor: queued
  MuJoCo TAMP job from `home_robot_v3` waiting on this HM-EQA PID.

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
- 2026-07-25 13:02: **hard freeze** mid classic q48 (no kernel line; boot gap).

Every recorded **kernel** native crash before 10:30 was running on logical CPU
**8 or 9**. Those are SMT threads of one 6.0 GHz P-core. The other 6.0 GHz
P-core is logical **10–11**. Both must be excluded.

Apport has crash reports for earlier Python failures. Core capture is disabled
in the launching shell (`ulimit -c` is 0); Apport is the configured core handler.

There are no kernel MCE/EDAC, NVIDIA Xid, OOM, or thermal-fault reports for the
13:02 freeze (journal simply stops). Absence does not rule out marginal CPU
voltage/boost stability.

## Working diagnosis

Highest-probability cause: **host CPU instability on the 6.0 GHz P-cores
(logical 8–11)**, exposed by native CUDA / MuJoCo / V8 / Habitat+VLM workloads —
not a single deterministic HM-EQA Python bug.

Immediate containment (landed in tree):

1. `emet.utils.cpu_affinity` — auto-exclude CPUs with max freq ≥ 6000 MHz
   (`0-7,12-31` on this box).
2. H2H applies that affinity at start (no manual incomplete `taskset`).
3. `EPISODE_COOLDOWN_SEC` (default 20) + `sync` between episodes.
4. Resume writes `host_freeze_*.log` when empty mid-episode jsonl + NUL-padded log.

Longer-term host checks: BIOS defaults / Intel baseline power, microcode,
focused CPU/RAM stress after the experiment.

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
4. What **not** to do (`kill-stale`, hard-kill Habitat, relaunch while job alive,
   incomplete `taskset` that leaves any 6 GHz core online, queue MuJoCo beside Habitat).

Prefer writing through `scripts/status_log.sh` so
`bash scripts/status_log.sh tail` (per-repo under
`~/runs/emet/status/<repo>/`) carries the same `next:` instruction; keep this
markdown as the durable investigation narrative.
