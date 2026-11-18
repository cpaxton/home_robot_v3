# Versioned scoped native wheel feedback

The [complete report](../../../../docs/experiments/scoped_wheel_feedback_20261116.md)
interprets this matched 113-case-per-contract comparison at source
`4b3c405960876825c57f5925300f001ee1d1ae78`.

`results.json` retains every completed case outcome and metric, original raw
report hashes, source/archive/receipt identities, the separately bound baseline
completion producer, and the offline W&B run. It records **1,538 passing tests**,
all 226 measured-stop confirmations, and the cold v4 viewer check. Paired TAMP
initial states, compiled models, terminal poses and object motion are identical.
Repeated deterministic pairs are not independent reliability trials.

`protocol.json` fixes all cohorts, audit seeds and mask event times.
`comparison.svg` shows paired success and the selected physics-boundary gain.
`*.py.txt` preserves exact producers. The original baseline runner aborts on an
incorrect masked-motion-hint assertion after 109 valid cases; the separate
completion producer retains those results and runs the four corrected mask
audits. Candidate and baseline producer identities remain distinct. Physical
reference samples, contracts and audit identities are unchanged by that repair.

The scoped option improves moving goals 10/12 → 12/12 and preserves coupled/TAMP
success, but larger coupled remains 6/10, repeated mask handoffs remain 2/4, and
moving linear jerk worsens. No PPO, default promotion, loaded-task qualification
or hardware motion occurs. Offline W&B: `w2ld2dmi`.

Raw traces, actor weights, generated extras and source archives stay ignored.
A source snapshot also needs a Git HEAD pinned to its recorded revision for
four existing report tests; the runtime used an isolated Git view with objects
from the canonical repository. No source bytes changed during evaluation.
