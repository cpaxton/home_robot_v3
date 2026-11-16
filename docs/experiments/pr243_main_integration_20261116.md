# Named partial-feedback integration — November 16

Integrated the reviewed native foundations on runtime source `2bb806b3`, then
landed #242 main `5e2a643e` at `407e763c`. The complete `src`, `scripts` and
`configs` trees remain identical to the tested runtime after parent validation
records and squash ancestry were integrated. No production conflict resolution
was required for the original parent update.

An independent review agent found no production blocker in effect-verified
completed prefixes, per-failed-step dispatch feedback, public copy/privacy
boundaries, native selectors, unavailable-tool batch stops and evaluator-only
fault/reauthorization seams.

**205 focused integration tests pass**, zero failures/errors/skips. Coverage
includes API/provider/task receipts, dispatch and continuation, native task CLI,
recovery scoring and partial-agent evaluation. Ruff, whitespace and full paper
build (70 pages) pass. The first test launcher used a nonexistent test path and
exited before tests ran; its log is retained alongside the corrected successful
run. No application or physical trial is counted from that setup failure.

The [compact record](pr243_main_integration_20261116.json) retains source/job and
artifact hashes; local raw artifacts are under
`.tamp-runs/native-stack-20261116/pr243/`. Tests overlap parent suites and are not
additive totals. Parent #242's new native/actual-agent shifted-swap checks verify
the underlying motion integration; they are not reruns of #243's feedback cohort.
The final #244 live integration gate exercises named partial feedback plus the
new contextual-retry behavior through actual `run agent`.

The original #243 recovery scorer does not automatically enforce fresh discovery,
a new plan and exactly one remaining execution; the recorded restated-goal trace
was manually checked. #244 adds automatic sequence checks and contextual retry.
Historical prospective and feedback successes/failures remain unchanged. No
contact-loss, held-object, unknown-dispatch, observed or hardware recovery claim.
