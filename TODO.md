# Living TODOs (small / near-term)

Short checklist for agent/hardware polish that is not worth a full plan doc yet.
Strike through or move to a PR when done.

## Embodied agent / Herman

- [ ] **Arm IK “closer look”**: point the wrist / EE camera at a named object (or image region), then capture.
  - Stub tool today: `aim_arm_at` in chat pack (returns not-implemented + suggests `describe_scene`).
  - Needs: Mars/Stretch IK + wrist frame + safety (collision, joint limits); then `take_ee_picture` only *after* aim.
  - Until then: “closer look / inspect X” → **change viewpoint** (rotate / small drive) then `describe_scene` / `send_image` (head), never raw `take_ee_picture`.
- [ ] **Chat verify ≈ EQA look**: prefer `face_toward` + `describe_scene` (not blind +45°). Full EQA-style `navigate_to_obs` + `verify_siglip` in CHAT still optional later.
- [ ] **`emet run` + `--connection`**: wrapper always injects `--robot-ip 127.0.0.1`, overriding connection profiles — use `--robot-ip 192.168.1.43` (or fix CLI so connection wins when robot-ip was defaulted).
- [ ] **Interruptible explore**: Discord messages queue until `explore` finishes; drain `unified_input_queue` mid-nav.
- [ ] **VRAM for Discord house chat**: default Mars preset loads tool LLM + EQA 8B VL + SigLIP (+ DA3). Easy OOM on 24 GiB when map update / DA3 overlaps generate. Mitigations: `--onboard-da3`, share one VL (`--llm qwen3-vl-eqa --share-memory-vllm`), or lazy-unload captioner between turns.

## Mapping / safety

- [x] Map-clip `move_forward` (including 0.1 m); refuse when map empty/blank.
- [x] Empty cloud guard in `list_objects_in_an_image` (navigate crash).

## Docs / ops

- [ ] Document Herman Discord happy path with `EMET_BASE_ROTATE_ONLY` + `EMET_ALLOW_SDPA_ATTN` / flash-attn in hardware checklist (partially done).
