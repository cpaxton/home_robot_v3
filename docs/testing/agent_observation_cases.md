# Agent observation and movement case catalogue

Qualify the same harness across EQA and OVMM. Model correctness, geometry
admission, motion safety and task completion are separate outcomes.

## Fast cross-tool integration gate

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 12 .venv/bin/python -m pytest -q \
  src/test/eval/test_agent_observation_cases.py \
  src/test/controller/test_validated_relative_motion.py \
  src/test/agent/test_floor_recovery.py \
  src/test/eval/test_agentic_floor_recovery.py
```

Choose an allowed stable CPU on other hosts. No GPU/model download required.
Use pytest `--collect-only -q` to list cases and `--junitxml=...` to save results.

| ID | Scenario and required outcome | Coverage |
| --- | --- | --- |
| OBS-01 | Visible but far: 3.3 m target exceeds 2.5 m gate; no object geometry. Closer observation is allowed, camera position is not object position. | Parameterized far-target case in `test_agent_observation_cases.py`; live microwave recovery still unqualified |
| OBS-02 | Unknown floor: reject before motion, publish checked pose, recheck same pose after mapping; sensing never authorizes motion without replanning. | Cross-tool rejection/recovery/retry test in that file |
| OBS-03 | Real obstruction: floor inspection cannot clear an occupied footprint or permit motion. | Obstruction case in that file |
| OBS-04 | Absent target: semantic abstention creates no geometry; do not retry just to override abstention. | Parameterized absent-target case in that file; full-agent refusal is a separate live gate |
| OBS-05 | Partial/no motion: report measured rather than requested travel; capture actual outcome even after failure. | `test_validated_relative_motion.py`, `test_agent_relative_motion.py` |
| OBS-06 | Stale recovery: new attempts have distinct plan identity; receipt mutation cannot alter recovery geometry. | Cross-tool receipt-isolation case and `test_agentic_floor_recovery.py` |
| OBS-07 | Bounded sensing: measured head arrival and fresh acquisition required; missing depth is not free floor. | `test_floor_recovery.py` |

The new cross-tool cases use real movement/recovery/grounding functions with
controlled planner, semantic-response and map inputs. They are not physics,
sensor or model-accuracy tests. Mapping changes are explicit fixture changes;
the tests do not manufacture sensor evidence.

## Serial live simulation gate

Use `scripts/run_shared_room_pairs.py --help` for required frozen source
checkouts/output arguments. Run managed CPU-safe, GPU-exclusive jobs serially.

- S0: `--episodes default_table_s0_distinct_recep` (Stretch object/destination find).
- Recovery: `--episodes molmo_rby1_ithor_s2_idx0` (rby1 bowl/microwave).
- Freeze model/quantization, seed, physics, scan/navigation budgets and source
  SHA/assets. Candidate-only is diagnostic; parity claims require both arms.
- Preserve complete requests, receipts, image roles, base/head trajectories,
  footprint before/after, measured motion, geometry admission, final-answer
  source, wall and available simulation time. Empty/incomplete artifacts fail.
- Require independently scored object AND destination localization. Textual
  correctness is not OVMM localization; find is not pick/place.
- Follow recovery with S0 and fixed EQA regression gates. Keep original failures.

The current Molmo layout combines far-target and unknown-start-footprint
conditions. Dedicated live variants isolating each remain to be authored;
the synthetic contracts do not claim those scenes exist.

Historical failure: frozen `fb8af512`, job `20261023_085419_4ed1b5`, outputs
`/home/cpaxton/runs/emet/relative-motion-molmo-20261023`. Qwen requested motion,
then correctly requested targeted floor inspection. Receipt footprint was not
published to shared navigation state: recovery returned
`missing_rejected_footprint`. The repair publishes it and resets attempt identity;
it does not weaken clearance.

Next integrations: [EXPRESS-Bench and AbstainEQA](../environments/embodied_qa_extensions.md).
