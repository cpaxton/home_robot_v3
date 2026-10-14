# Stretch contact baseline, October 13

**6/6 scripted native-actuator transfers passed on main `fd29b783`: three fresh
runs each for the packaged blue block and red cylinder.** These are deterministic
repeats of two fixtures, not six distinct tasks or a perturbation sweep. No task
or simulator implementation changed for this evaluation.

The existing `scripts/render_stretch_pick_place.py` drives Stretch arm/gripper
actuators and steps MuJoCo physics. Fixture initialization positions objects and
initial joints before execution. There are no runtime object pose writes,
attachments or latch assistance. Targets/IK and scoring use privileged simulator
state; this is a contact execution diagnostic, not a real-robot sensor pipeline.

| Object | Passed | Minimum scored held lift | Maximum scored translational slip | Final XY error |
| --- | --- | --- | --- | --- |
| Blue block | 3/3 | 126.1 mm | 9.6 mm | 16.7 mm |
| Red cylinder | 3/3 | 138.2 mm | 1.5 mm | 13.5 mm |

Each run met the existing criteria: at least 80 mm lift; bilateral fingertip
contact and no table contact in the final one-second windows of both lift and
transfer; held translation within 25 mm; final XY/Z errors within 40/15 mm; and
two seconds of table support without fingertip contact at speed below 10 mm/s.
Both objects have bilateral contact in all 24 scored held samples per run.
The maximum final release speed was below 0.3 mm/s. Sampling is at 12 Hz; this
is not continuous collision certification or a force/rotational-slip audit.

![Measured height and bilateral contact](media/contact_20261013/contact_evidence.png)

[PDF figure](media/contact_20261013/contact_evidence.pdf) ·
[Versioned JSON metrics](media/contact_20261013/contact_metrics.json)

## Reproduction and evidence

Managed CPU-only job: `20261013_022329_440d0d`. Two workers, no rendering or large
models, fresh simulator state per run, OMP/OpenBLAS/MKL threads set to 1.
Artifact root: `/home/cpaxton/runs/emet/tamp-contact-20261013/`.
It contains the source manifest, exact commands, six result JSON files, sampled
contact/pose traces, process logs, summary and figure. No run was excluded.

The six commands use the existing entry point with `--object object1` or
`--object object2`, each repeated three times into a fresh `--output` directory.
The CLI was checked before launching. The 47 targeted tests in
`test_contact_tracking.py`, `test_physical_tamp.py`, and
`test_eval_physical_tamp.py` also passed (faulthandler watchdog disabled).

## What this establishes and what remains

Stretch's packaged gripper/contact geometry can lift, carry and release both
fixtures through actuator control on current main. Success does not establish
collision-certified arm paths, swept base transport, obstacle recovery, learned
agent success, hardware performance or broad contact robustness. The script
uses coupled-joint pose IK and a fixed short transfer without base navigation.
The native execution adapter for public `run agent` TAMP tools remains unwired.
Sourccey was not tested by this Stretch-specific driver.

Next gates: connect this execution/evidence path to native-debug public TAMP
planning, then run the same fixtures through plan/execute with path collision
checks; add bounded pose/geometry variation and failed-grasp/release negatives;
then establish a Sourccey contact fixture with its own actuator and gripper
bindings. Observed execution must not obtain contacts/object poses from the
simulator: these privileged measurements belong to debugging and independent
acceptance scoring.
