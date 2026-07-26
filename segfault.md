# Segfault investigation log

Last updated: 2026-07-26 10:58 


### Fresh bal-32 #2 under paper-router (significance re-check) — 2026-07-26 10:58

| Field | Value |
|-------|-------|
| Goal | Second independent bal-32 classic vs agentic under the policy that went 8/8 on holdout (overnight `hmeqa_overnight_20260726_022227` gave classic 10/32 vs agentic 12/32, McNemar p≈0.75 — not significant) |
| Policy | `--preset paper-router` (owlv2 + allow-unverified + agentic-router=1) |
| Launch | `uv run emet hmeqa h2h ~/runs/emet/hmeqa_agentic_bal32r2_<stamp> --preset paper-router --job-name hmeqa-bal32r2` |
| Resume | `uv run emet hmeqa resume <OUT> --preset paper-router` |
| Do not | kill-stale while job live; hard-kill Habitat mid-episode |
| Next | `uv run emet jobs` · `uv run emet hmeqa status <OUT>` · compare vs `hmeqa_overnight_20260726_022227/bal32` |

### Re-run overnight ladder (Qwen router + letter anti-hijack) — 2026-07-26 09:38 

| Field | Value |
|-------|-------|
| Goal | Fresh holdout-8 → bal-32 after VLM-first submit fixes (working tree) |
| Policy | owlv2 + allow-unverified + **agentic-router=1** |
| Script | `scripts/run_hmeqa_overnight_ladder.sh` |
| Launch | `uv run emet jobs run --name hmeqa-overnight-router --need-mib 12000 -- ./scripts/run_hmeqa_overnight_ladder.sh` |
| Do not | kill-stale while job live; hard-kill Habitat mid-episode |
| Next | `bash scripts/status_log.sh tail` · `uv run emet jobs` · `uv run emet hmeqa status OUT` |

### Host freeze during q104 frontier-viz debug (2026-07-25 ~22:32 → reboot 22:42)

| Field | Value |
|-------|-------|
| Class | **Host hard freeze** (journal ends mid-warp; no OOM/Xid/hung_task; `last` marks sessions `crash`) |
| Trigger window | After q104 frontierviz **DONE** at 22:15:53 — Warp GUI `crash_report` events from 22:15:59; syslog lasts until ~22:32 then reboot 22:42 |
| Job at freeze | `20260725_221017_11296e` already finished (not mid-episode) |
| OUT | `~/runs/emet/hmeqa_agentic_q104_frontierviz_20260725_221014` — agentic **0/1**, pred empty, `require_verified` abstain; **all** `explore_frontier` had `frontier_xyz=null` |
| Root explore bug | `pick_habitat_exploration_target` permanently `blocked.add` on soft-recent goals → empty frontier set. Fix: soft-skip only (`5ab8d1d`) |
| GPU now | idle ~23.5 GiB free; no registered jobs |
| Do not | relaunch q104 unless validating soft-recent; `kill-stale` not needed |
| Next | optional q104 re-smoke with pick panels; then `emet hmeqa resume` bal-32 `~/runs/emet/hmeqa_agentic_bal32_20260725_101519` (classic 14/32, agentic 4/10 @ 52/64) |

```bash
bash scripts/status_log.sh tail
uv run emet jobs
uv run emet eval status
# validate fix (optional):
# uv run emet jobs run --name hmeqa-q104-softrecent --need-mib 12000 -- \
#   env EMET_ALLOW_SDPA_ATTN=1 ./scripts/run_hmeqa_agentic_h2h.sh --ids 104 --arms agentic
```

---


### STOPPED 2026-07-25 23:09 — user cancel bal-32; debug q104/105 frontier

| Field | Value |
|-------|-------|
| Cancelled | `20260725_230214_8e1494` at **54/64** agentic q40 |
| Bal-32 live | classic 14/32 (11 empty); agentic **8/22** (7 empty); **verified=2/22**; many free-form XYZ answers |
| Do not | relaunch Habitat / resume bal-32 until frontier mid-floor pick is fixed |
| Frontier bug | picks land in **already-explored open floor** (panel waypoint mid-green); selectable `n_frontier≈2`; viz blue ≠ planner-reachable frontier; frequent `NO_NEW_OBS` after 4 m nav |

### q104 soft-recent smoke — PASSED (2026-07-25 23:00)

| Field | Value |
|-------|-------|
| Job | `20260725_225326_893b65` done |
| OUT | `~/runs/emet/hmeqa_agentic_q104_softrecent_20260725_225212` |
| Frontier | **4 non-null** / 2 null (was 6/6 null); 4 `frontier_picks/iter_*.png` |
| Score | still 0/1 empty pred (verify gate) — explore fix only |
| Next | resume bal-32 |

### bal-32 resume (launching)

| Field | Value |
|-------|-------|
| OUT | `~/runs/emet/hmeqa_agentic_bal32_20260725_101519` |
| Progress | classic 14/32, agentic 4/10 @ 52/64 — resume fills empties |
| Commit | soft-recent + INCOMPLETE DONE gate |
| Do not | kill-stale; second GPU job |
| Job | `20260725_230214_8e1494` (`hmeqa-bal32-resume`) |
| Monitor | `uv run emet jobs status 20260725_230214_8e1494` |



| Field | Value |
|-------|-------|
| OUT | `~/runs/emet/hmeqa_agentic_q104_softrecent_20260725_225212` |
| Commit | `dd9c622` |
| Arms | agentic only, ids=104 |
| Expect | non-null `frontier_xyz` in trace; `bundles/agentic_q104/frontier_picks/iter_*.png` |
| Do not | kill-stale; second GPU job; hard-kill Habitat |
| After | if OK → `emet hmeqa resume ~/runs/emet/hmeqa_agentic_bal32_20260725_101519` |


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

### q104 explore-fix (launching 2026-07-25T23:49:14-04:00)

| Field | Value |
|-------|-------|
| OUT | `/home/cpaxton/runs/emet/hmeqa_agentic_q104_explorefix_20260725_234914` |
| Commit | `888c069` |
| Fixes | reachable-adjacent frontier snap; habitat max_depth=4.5/pad=1/smooth=1; look-around on NO_NEW_OBS |
| Arms | agentic only, ids=104 |
| Expect | explored_area grows; frontier picks on rim not mid-floor; fewer NO_NEW_OBS |
| Do not | kill-stale; second GPU job; hard-kill Habitat |
| Monitor | `uv run emet jobs` / `emet hmeqa status /home/cpaxton/runs/emet/hmeqa_agentic_q104_explorefix_20260725_234914` |


### q104 explore-fix — FAIL on exploration (2026-07-25 23:55)

| Field | Value |
|-------|-------|
| Job | `20260725_235021_0c17be` done |
| OUT | `~/runs/emet/hmeqa_agentic_q104_explorefix_20260725_234914` |
| Score | agentic **1/1 correct=D** (allow-unverified) — not an explore win |
| explored_area_m2 | **1.27 identical** to softrecent (md5 match) |
| Motion | trajectory path ~0.5 m, mostly spin; nav_attempts: already_at_goal |
| Frontier picks | still mid-green; look_around_on_no_new_obs fired 2×; NO_NEW_OBS 5/7 |
| Do not | claim explore fixed; resume bal-32 |
| Next | fix Habitat `pick_uncovered` / navmesh snap so goals leave spawn blob |


### q104 floor-area rerun (launching 2026-07-26T00:22:30-04:00)

| Field | Value |
|-------|-------|
| OUT | `/home/cpaxton/runs/emet/hmeqa_agentic_q104_floorarea_20260726_002230` |
| Commit | `888c069` |
| Why | explorefix bundle export crashed (frontier_picks SameFileError) → maps/explored_2d were STALE from 22:15; explore verdict unknown |
| New | floor_area.jsonl + floor_area_growth.png per episode; same-file copy skip |
| Expect | fresh explored_2d + per-step area curve to judge growth |
| Do not | kill-stale; second GPU job |


### q104 floor-area rerun — EXPLORATION CONFIRMED (2026-07-26 00:27)

| Field | Value |
|-------|-------|
| Job | `20260726_002233_976c0a` done, correct=D |
| OUT | `~/runs/emet/hmeqa_agentic_q104_floorarea_20260726_002230` |
| Explored | **8.84 m² final, peak 13.2** (old stale claim of 1.27 was a bundle-export bug) |
| Motion | 6.5 m path, 27 unique poses (was 0.5 m) |
| Fixed | frontier_picks SameFileError aborted bundle export → maps/explored were stale from 22:15 |
| New | per-step `floor_area.jsonl` + `floor_area_growth.png` in every episode bundle |
| Open issue | explored area **non-monotone** (5.4→0.8→13.2→9.1) — DynaMem frustum culling erases coverage; frontier regenerates near robot → NO_NEW_OBS churn |


### q104 spin-no-erase validation (launching 2026-07-26T01:19:12-04:00)

| Field | Value |
|-------|-------|
| OUT | `/home/cpaxton/runs/emet/hmeqa_agentic_q104_spinfix_20260726_011912` |
| HEAD | `888c069` (+ uncommitted voxel/frontier/spin fixes) |
| Why | prior floorarea confirmed explore but **non-monotone** area (spin cleared without re-add) |
| Arms | agentic only, ids=104, allow-unverified |
| Expect | `floor_area.jsonl` mostly monotone; no opening-scan erase; frontier picks leave spawn |
| Do not | kill-stale; second GPU job; hard-kill Habitat |
| Monitor | `uv run emet jobs` / `emet hmeqa status /home/cpaxton/runs/emet/hmeqa_agentic_q104_spinfix_20260726_011912` |


### q104 clear-logic validation (launching 2026-07-26T02:05:58-04:00)

| Field | Value |
|-------|-------|
| OUT | `/home/cpaxton/runs/emet/hmeqa_agentic_q104_clearfix_20260726_020558` |
| HEAD | `888c069` (+ uncommitted clear_points fix: strict-past carve + 2x2 + validity mask) |
| Why | validate static floor no longer shrinks; expect monotone `floor_area.jsonl` |
| Arms | agentic only, ids=104, allow-unverified |
| Expect | no 5.4→0.8 / 13→8 / mid-episode map collapse; explored mostly monotone |
| Do not | kill-stale; second GPU job; hard-kill Habitat |
| Monitor | `uv run emet jobs` / `emet hmeqa status /home/cpaxton/runs/emet/hmeqa_agentic_q104_clearfix_20260726_020558` |


| Job | `20260726_020704_b9edb3` |

### q104 clear-logic validation — PASS (2026-07-26 02:11)

| Field | Value |
|-------|-------|
| Job | `20260726_020704_b9edb3` done, correct=D |
| OUT | `~/runs/emet/hmeqa_agentic_q104_clearfix_20260726_020558` |
| Explored | **5.2 → 36.3 m² at step 20** (final floor_metrics **37.0 m²**); free floor 3.8 → 26.4 |
| Curve | monotone through last live stride sample (0/5/10/15/20) |
| Note | maps `step_0025+` in the shared episode cache were **stale leftovers** from earlier runs (timestamps 00:27 / 22:15); not a 20→25 reset. Flush now wipes old stride PNGs. |


### Overnight ladder (job 20260726_022333_7b5326)

BASE: `/home/cpaxton/runs/emet/hmeqa_overnight_20260726_022227`

| Field | Value |
|-------|-------|
| Script | `scripts/run_hmeqa_overnight_ladder.sh` |
| Phase 1 | holdout-8 `15,56,65,68,79,88,104,105` classic+agentic |
| Gate | if agentic << classic or n&lt;6 → one agentic-only retune |
| Phase 2 | fresh bal-32 classic+agentic |
| Policy | owlv2 + **allow-unverified** + no-router (explore fix; verify abstain was broken) |
| Do not | kill-stale; second GPU job; hard-kill Habitat |
| Monitor | `uv run emet jobs` / `bash scripts/status_log.sh tail` |


| Job | `20260726_022333_7b5326` |
| BASE | `/home/cpaxton/runs/emet/hmeqa_overnight_20260726_022227` |
