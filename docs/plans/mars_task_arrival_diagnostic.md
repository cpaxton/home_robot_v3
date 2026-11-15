# Bounded Mars task-arrival experiment

The retained actor reaches the matched actual TAMP pregrasp with about
3.01 mm / 0.02966 rad final error. The current 3 mm / 0.02 rad endpoint contract
stops execution before closure. Test whether accepting a nearby pregrasp pose
helps the actual task, without relabeling the original precision result.

Four fixed full public-tool attempts, all on `development:right:50g:1`:

1. Analytic teacher, original precision contract.
2. Retained frozen actor, original precision contract.
3. Same teacher, explicit pregrasp arrival region 4 mm / 0.04 rad.
4. Same actor, same pregrasp region.

No checkpoint selection, new reset seed, retries, training or hardware. Current
wheel feedback is enabled from construction, with the previously validated
gain, physical wheel slew and turn guard. Each attempt owns one isolated native
worker on CPU14, with one numerical thread and CUDA hidden. CPU15 handles
focused checks. Freeze source and scripts before native attempts.

The region changes **endpoint acceptance only**. The existing IK/witness
tolerance, reference geometry, speed limits, phase retiming scales, execution
time budget and independent 10 mm / 0.1 rad path tube remain unchanged. The
experiment is pregrasp-only: grasp, lift and placement keep 3 mm / 0.02 rad.
Completion still requires measured quiet hold. Per-cycle collision, whole-base
drift, static support, contact-force, retention and wheel-unloading checks stay
active. Invalid regions fail closed before actor inference.

The planned pregrasp is 10 mm above the grasp pose. This makes the nearby arrival
region worth testing, not a geometric robustness certificate. Record signed
finger/source-support and finger/object distances at the initial and each
executed stage endpoint, using copied private simulator state. Benign object
contact is permitted. Runtime configuration checks and actual measured grasp
replanning remain authoritative; refuse motion if they reject it. Do not claim
the whole uncertainty region is certified by a few sampled distances.

Run `scripts/probe_mars_task_arrival.py --output logs/eval/mars-task-arrival-v1`
from a frozen checkout. Preserve every public tool response, stage result,
raw per-tick trace, final task predicate and measured-stop receipt. A successful
pick/place requires completed grasp/place operations, independent placement,
support, release, retention/safety and stillness predicates, plus two further
seconds of physical stability. Later-stage failure remains failure; pregrasp
arrival alone is not a successful grasp or task.

Publish compact outcomes, endpoint errors/geometry, closure/lift progress,
retention/contact/support metrics and provenance. Raw `.json.gz` files stay
ignored. The prior full regression receipt does not qualify changed source.
These four development attempts cannot qualify hardware or broad task success.
