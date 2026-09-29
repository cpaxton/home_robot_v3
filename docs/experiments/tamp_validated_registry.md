# Scene-validated TAMP controls

The historical 200-row registry is retained unchanged. It contains construction
errors and is not a denominator of 200 valid planning tasks. The new workflow
separates candidate construction, executed reference validation, and fresh MCTS
evaluation. It does not turn teleport/latch controls into physical acceptance.

## Admission contract

- Resolve actual scene furniture, a real trash receptacle, and enough movable
  objects from ground-truth scene metadata. Missing requirements are coverage gaps.
- Construct bounded deterministic clutter layouts without overlapping their
  navigation-model disks or embedding them in unselected obstacle disks.
- For navigation tasks, require a free start/goal, a blocked initial route, and
  a route opened by the declared relocations. Keep all other obstacle disks,
  including the destination furniture; never erase occupied endpoint cells.
- For cleanup, reject an already-satisfied task.
- Execute a deterministic grounded reference sequence through the declared
  oracle/latch backend, without calling MCTS. Verify every object at the target
  and the navigation goal reached when required. Preserve the reference steps.
- Admit only successful references. Retain construction failures, reference
  failures, crashes and timeouts separately; none is a scored MCTS failure.
- Freeze exact body identities, clutter positions/orientations, goal, receptacle,
  initial base XY, scene fingerprint, backend, clearance and reference evidence
  under a content hash. Evaluate MCTS in a fresh simulator on that fixture.
  Refuse changed scenes, backend overrides, changed starts and stale certificates.

The reference shares low-level grounding and execution with the tested planner,
but does not use its MCTS search. Admission demonstrates a solution within this
backend and can bias the corpus toward its representable actions; report coverage
and rejections alongside success rates. Disk geometry is an explicit approximation.
Latch uses its existing IK/attachment contract, and oracle mode teleports objects;
neither proves physical grasps, contact-safe manipulation, or full robot geometry.

## Commands

Generate construction requests (not yet scored episodes):

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_tamp_clutter_registry.py \
  --output /tmp/tamp-candidates.yaml
```

From a clean frozen checkout, submit each live stage through
`emet jobs run --cpu-safe --gpu-exclusive --need-mib 8000`, with
OMP/OPENBLAS/MKL thread counts set to one. Dry-run the runner first:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_tamp_experiments.py \
  --suite full --registry /tmp/tamp-candidates.yaml --validate-fixtures \
  --output-dir /tmp/tamp-admission --dry-run
```

Remove `--dry-run` inside the managed job. `--case-id` can select pilot candidates.
Admission writes `ledger.json` for every requested candidate and
`validated_registry.yaml` containing only admitted fixtures. Once admission is
terminal, freeze that registry and use the same source for evaluation:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_tamp_experiments.py \
  --suite full --registry /tmp/tamp-admission/validated_registry.yaml \
  --output-dir /tmp/tamp-evaluation --dry-run
```

Admission success is `fixture_admitted`, not task success. Report the number of
requested candidates, admitted fixtures, rejected fixtures by reason, and evaluated
fixtures separately. Do not substitute the historical 200-row denominator or
silently replace a rejected task at evaluation time.

## Status, September 28

Fixture construction, admission and fresh replay are implemented. The final
source (`5b619590`) passes 116 targeted tests. The eight-case pilot admits four
fixtures and rejects four. All four admitted fixtures passed fresh MCTS replay.
The full 200-candidate admission/replay run is active; no 200-task solvability
claim is established.

The first eight-case pilot exposed certificate serialization of simulator NumPy
arrays after successful oracle execution, RBY1 attachment verification failures,
and layouts rejected by the disk geometry. These are pilot diagnostics, not a
scored MCTS result. Serialization now normalizes arrays/scalars before hashing.
Construction diagnostics distinguish object overlap from scene overlap.

Certificates include hashes of the source scene XML and relevant implementation
files. The builder saves an initial scene snapshot and freezes measured settled
clutter poses. Replay checks XYZ within 3 cm and orientation within 0.1 rad
(accounting for quaternion sign). Final scoring uses measured object locations;
navigation additionally requires a route from the **original** start through the
final scene, a verified relocation, and actual arrival at the goal. Escaping the
clutter ring with a base teleport is insufficient. These checks remain within the
explicit oracle/latch and disk-model scope above.

Pilot r1 (`a6cbcb2d`, job `20260928_215816_11d8d0`) is terminal: two
RBY1 reference rejections, two Mars construction rejections, and four oracle
certificate-serialization errors. Its artifacts and construction requests are
archived at `~/runs/emet/tamp-validated-fixtures-20260928/`. Pilot r2
(`75db7026`, job `20260928_221033_b565a2`) rechecks the same eight candidates
and runs fresh MCTS replay only after admission terminates. The current contract
also binds task mode, object count, success radius, backend, bin query and seed;
changing scoring settings requires a new witness.

### Completed pilot and active full run

Pilot r2 is terminal: **4/8 admitted; 4/4 admitted fixtures passed fresh MCTS**.
This is one scene and two oracle-backed robots, not a robustness estimate.

| Robot | Cleanup admission | Navigation admission | Fresh MCTS |
| --- | --- | --- | --- |
| RBY1, latch | Rejected: apple attachment verification; 2/3 relocated | Rejected: bottle/knife failures; 6/8 relocated | Not scored |
| Stretch, oracle | Admitted | Admitted | 2/2 passed |
| Mars, oracle | Rejected: all four layouts overlap scene disks | Rejected: all 288 layouts overlap scene disks | Not scored |
| Nori, oracle | Admitted | Admitted | 2/2 passed |

RBY1 reference failures do not prove the tasks impossible. Mars construction
failures are specific to the bounded builder and its conservative disk model.
No rejection was relabeled as an MCTS failure or silently replaced.

Durable evidence: `~/runs/emet/tamp-validated-fixtures-20260928/` contains
`tamp-fixture-pilot-20260928-r2/` (snapshots, reference traces, certificates,
`fixture_topdown.png` per admitted case) and `tamp-fixture-replay-20260928-r2/`
(fresh MCTS traces and final-state metrics). The historical r1 evidence is retained.

Full run: job `20260928_222600_2cf5c6`, clean source `5b619590`, 900 seconds
per case, serial CPU-safe/exclusive-GPU execution. Outputs are written directly
to the durable archive's `full-r3/`, followed by `replay-r3/`. Admission must
finish all 200 candidates before `frozen_registry.yaml` is copied for replay.
`admission_evaluation_summary.json` will report requested, admitted, not admitted,
evaluated, successes, and evaluation status counts. Full results are pending.
