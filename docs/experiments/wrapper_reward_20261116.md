# Whole-body wrapper reward: baseline and experimental hand precision

The training wrappers use `RewardMeter`, not the scalar reward returned by the
native manipulation environment. Native precision/progress/endpoint settings
therefore do not change this wrapper objective. This applies to general mastery,
coupled tracking, and the spawned TAMP primitive bridge.

## Existing default

At every 0.05 s physical control tick (20 Hz), the three critic groups receive:

- Tracking: `dt * (hand + base)`, plus terminal credit on the final tick.
- Stability: `-dt * (0.01 * robust_normalized_jerk + 0.001 * action_effort)
  - 0.01 * squared_action_change`. Motion action channels exclude the gripper.
- Contact safety: `-dt * 0.1 * (max(force_N - 3, 0) / 3)^2 - 10 * guard_failed`.
  Force is the current conservative body-pair load, excluding ordinary wheel
  support. Historical maximum force is not charged again on later quiet ticks.

For an active hand, pose credit is `exp(-position_error/0.01 - rotation_error/0.1)`.
Velocity credit is independently `0.5 * exp(-||linear_twist_error/0.02||^2
- ||angular_twist_error/0.2||^2)`. Units are metres, radians, m/s and rad/s.
Base pose credit is `exp(-outside_xy_region/0.02 - heading_error/0.05)`;
velocity credit is `0.5 * exp(-sum((base_velocity_error/[0.02,0.02,0.1])^2))`.
Existing target masks and the coupled-v2 heading acceptance region still apply.
Maximum tracking density with both targets active is 3 per simulated second.

Terminal success is +2. General/coupled failure is -2; TAMP primitive failure
is **-10**, an existing difference now recorded explicitly. Guard penalties
are additional. Individual rewards from bundled TAMP executor-hold ticks,
including the final terminal credit, remain individually discounted in PPO.

At a stationary hold, a perfectly placed base and a hand 10 mm / 0.02 rad wrong
still receive 2.301 per second: 76.7% of the maximum tracking density. Exact hand
placement is rewarded more, but independently earned velocity/base credit dilutes
this signal. This does not establish that the wrong pose is globally optimal or
that every historical checkpoint was trained with this wrapper.

## Explicit experimental profile

The default is unchanged. Opt in with the complete versioned configuration:

```text
--reward-config configs/benchmarks/whole_body_reward_contract_v1.json
```

`WholeBodyRewardConfig` validates every field before worker allocation. The same
record reaches all three wrappers, run/summary JSON, offline W&B configuration,
training checkpoints and exported actor training metadata. Normalization/noise
pairing rejects different reward configurations. The profile JSON is included
in the source-bound regression manifest.

For the experimental profile, hand pose credit is:

```text
4 * (0.25 * exp(-position_error/0.02 - rotation_error/0.2)
   + 0.75 * exp(-position_error/position_tolerance
                - rotation_error/rotation_tolerance))
```

The fine scales come from each environment's actual hand contract, rather than
from the curriculum level: for example 3 mm / 0.02 rad in mastery/coupled tests,
and 5 mm / 0.05 rad in the native TAMP fixture. These are reward length scales;
the existing executable-trajectory success gates remain independent.
Hand velocity credit and (when the hand is active) base velocity credit are
multiplied by the fine pose exponential. The hand therefore retains a whole-body
correction incentive inside the coarse base region. Base pose credit is retained;
no base observation or motion authority is removed. With only a base target,
velocity credit retains the legacy semantics.

Progress adds `0.1 * (potential_before - potential_after)` per tick, where
`potential = log1p(position_error/position_tolerance)
+ log1p(rotation_error/rotation_tolerance)`. Both measured poses are compared
against the **same post-step active goal**. Reference movement without measured
robot movement earns zero progress. A newly activated hand mask earns no progress
from its preceding inactive interval. This is an explicit shaping term, not a
claim of policy-invariant potential shaping under discounted, moving goals.

For a perfectly placed base, zero rotation error and matched zero velocities,
the stationary tracking density as a fraction of each profile's own maximum is:

| Hand position error | Legacy | Experimental |
| --- | ---: | ---: |
| 0 mm | 100% | 100% |
| 3 mm | 91.4% | 55.5% |
| 5 mm | 86.9% | 42.2% |
| 10 mm | 78.9% | 29.2% |
| 20 mm | 71.2% | 22.9% |

This table uses the 3 mm / 0.02 rad fine contract and excludes progress,
stability and contact costs. It describes reward shape, not measured success.

`stability_scale` permits a curriculum from 0 to 1 for jerk/effort/command changes.
The supplied profile starts at **1**, preserving those penalties. Force costs,
guard penalties and physical safety limits are separate; the stability curriculum
does not weaken them. The profile changes tracking reward units (maximum density
6 instead of 3), so do not reuse a critic as though the objective were unchanged.

## Validation and next comparison

Tests cover configuration round trips and rejection, legacy hand formula,
3/5/10/20 mm precision sensitivity, rotation and twist sensitivity, progress
under goal jumps and measured toward/away motion, mask activation, force guards
at zero stability scale, wrapper propagation, serialized training provenance,
and actual spawned TAMP execution with bundled physical-tick rewards.

The [paired pilot](wrapper_reward_training_20261116.md) subsequently trains
this profile and a matched legacy control for 16,384 decisions each. New-case
success is 18/24 and 19/24 versus retained 20/24; neither is promoted.
The initial implementation validation below predates that completed run. Historical 1,538-test source
receipts describe the preceding runtime; they do not certify these source edits.
That paired experiment uses independent reachable coupled/mask goals, separate
audit seeds and the 50/50 general/TAMP physical-tick mixture. Its completed
readout reports hand accuracy, quiet drift, transient error, jerk, forces, wheel
lift and confirmed stops separately. The new profile does not improve the
retained actor; the next comparison should isolate reward components and
critic/exploration effects. Improved reward values alone are not improved policy
performance.

The motivating measurements are in
[the hand accuracy diagnosis](hand_tracking_accuracy_20261116.md).

Validation completed on one CPU worker with numerical threads limited to one:
**166 targeted tests pass, zero failures/errors/skips**, covering the seven
wrapper/training/discount/provenance/snapshot test modules. The final JUnit is
`logs/eval/wrapper-reward-v1/verified-targeted.xml`
(SHA-256 `2ff6f40aedff53bbe0fcdd59f494476af1665c031662f0195da015e9c3f5a63f`).
A separate frozen-source comparison against `1014a9f7` reproduces bitwise-identical
three-group rewards on **64 native ticks**, 32 each on Mars and Stretch, using
the same measured states and teacher actions. The legacy reward module SHA-256
is `2d59e9a59c63388689be7f5b4e38de702a242582d86405e314897138dbf40b90`; the readout is
`logs/eval/wrapper-reward-v1/legacy-parity.json`. This is targeted validation,
not a fresh complete regression receipt or a policy-performance experiment.
