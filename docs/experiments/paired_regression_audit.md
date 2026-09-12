# September 9 regression investigation

Status: **no-regression acceptance is NOT established**. Passing unit tests and
merging the stack established implementation checks, not task-quality parity.

## Checked artifacts

Historical lazy random-16: `/home/cpaxton/runs/emet/merge_strategies/20260904_085805/a_lazy_graph.jsonl`
and adjacent manifest. Current: `/tmp/emet-integrated-eqa-random16-fixed/lazy_arrival.jsonl`.

- Historical 8/16 versus current 6/16: losses are exactly q15 and q25; no gains;
  the other fourteen correctness outcomes match.
- Same question IDs, Qwen3-VL-8B/int4, 20/10 top-level budgets, frontier weight,
  no HM3D semantics/enriched labels, and matching question/start-pose hashes.
- Historical source is **592eb8a4 plus dirty modifications**, not a clean
  47a30ed9 run. The latter is a committed reproduction proxy, not the exact
  historical source. Current source is clean 0c2dc7f0.
- Mean object/observation counts: old 13.5625/25.0625, current 12.625/23.9375.
  These failures are not accompanied by a return of instance flooding.

## Failure localization

q15 (fruit bowl next to microwave): old answered No, current Yes. Current raw
reasoning asserts proximity based on graph/view evidence. This is an incorrect
spatial answer; grounding and selected-view evidence need checking.

q25 (bathroom towels): initial camera image and all four first-turn attached
images have identical SHA256 hashes across runs. Spawn pose matches. Yet the
first prompt's frontier candidates/coordinates and weak SigLIP scores differ.
The first navigation goal matches; the second already differs. Old trace has
9 navigation attempts, current 19, all reported successful. Current answer
history repeatedly describes bedrooms without towels; the final low-confidence
answer is modified by the existing memory-location fallback. The old run
answered correctly from a bathroom towel view. The fallback implementation is
unchanged between 47a30ed9 and 0c2dc7f0, so it is not by itself a newly introduced
code change. It remains an answer-quality risk.

`voxel_dynamem.py` subsamples geometry/features using the global torch RNG
(`torch.randperm`); HM-EQA's episode runner does not initialize run RNGs.
This is a concrete reproducibility gap, not proof that stochasticity explains
the whole regression. Seeds also cannot guarantee deterministic GPU kernels or
identical global RNG consumption across changed implementations.

## Submitted checks (serial GPU-exclusive)

- `20260909_083034_a93310`: committed historical proxy 47a30ed9, q15/q25 plus
  unchanged q16 control; `/tmp/emet-regression-old-20260909`.
- `20260909_083035_1283fe`: main e8564ac6, same IDs/model/top-level settings;
  `/tmp/emet-regression-current-20260909`.
- `20260909_083206_53a933`: post-merge query-driven Habitat OVMM lamp/bed,
  12 tool rounds / 8 navigation budget; `/tmp/emet-ovmm-query-postmerge-20260909`.

Do not tune to these two failed questions and then report them as holdout
evidence. Next fix: explicit per-episode evaluation seeds recorded in manifests
and results, with matched-seed/repeat checks. Inspect mapping/frontier changes
before altering admission thresholds or reverting safety fixes. Keep native
DynaMem and shared-agent comparisons labeled separately. TAMP execution controls
and learned manipulation remain separate gates; no physical robustness claim.
