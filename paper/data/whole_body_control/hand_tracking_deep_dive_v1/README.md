# Hand-tracking accuracy deep dive

The [report](../../../../docs/experiments/hand_tracking_accuracy_20261116.md)
explains a fixed **24-episode, failure-enriched** diagnostic at runtime
`4b3c405960876825c57f5925300f001ee1d1ae78`.

- `results.json` preserves every outcome, source/weight/raw-trace hashes, exact
  primary-metric replay agreement, feedback cancellation, local Jacobian checks,
  counterfactual sensitivity, contact pairs, static separation bounds and reward
  source findings. All 24 stops confirm. No learning or hardware motion occurs.
- `protocol.json` fixes the nine actor/teacher cases and six interventions.
- `hand-tracking.svg` shows error history, cancelling feedback, probe outcomes
  and departure from a safe nominal self-collision witness.
- `*.py.txt` preserves exact audit and analysis producers. Compressed raw state
  and observation traces remain ignored under local logs.

The unchanged source still verifies its prior 1,538-test receipt. That suite was
not rerun or renamed as a new test result. This audit is not generalization,
loaded manipulation or hardware qualification. The gain/feature interventions
are diagnostic and are not exported or promoted. Scored contact-force maxima
are separate from callback peaks which can grow during unscored stopping.
