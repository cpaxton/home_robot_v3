# PR #203 oracle coordinate follow-up

The original matched run passed the RBY1 assisted task on both main and #203,
but the candidate rejected saved Stretch fixtures. A fresh candidate admission
also failed the kettle teleport grasp. These results are retained in
`tamp_matched_regressions_20261012.md`.

The oracle used `sim_object_placements[*].pos` (world AABB center) as the argument
to `sim_set_body_pose` (named-body world origin). Offset meshes and welded
children therefore acquired a translation error. Stretch then temporarily wrote
the requested origin into the center field, hiding the disagreement until the
new dynamic tracker published measured geometry.

PR #203 now publishes separate `body_pos` and `body_quat` fields for entries
that resolve to actual named bodies. Center-targeted oracle translations convert
through the measured origin-to-center offset and pin the measured orientation.
Missing measured body geometry fails explicitly. Aggregate semantic groups that
do not resolve to bodies are not given invented origins. Stretch no longer
patches object metadata from requested command targets.

New clutter certificates use schema 2. Admission records the measured center,
named-body origin and orientation after settling. Replay commands the saved
body pose, then independently checks measured center/orientation and fixture
geometry. Schema-1 certificates are rejected instead of being reinterpreted;
historical results are not rescored, and new admission/replay outcomes have
separate denominators. This remains a privileged teleport oracle, not native
contact or an observed robot sensor.

Validation: the geometry/manipulation/fixture group passed 46 checks before the
final certificate validation test; the final manipulation/fixture group passed
24 checks. The integrated TAMP, payload/contact, Stretch endpoint/telemetry and
placement metadata group passed 154 checks. Counts overlap and are not summed.
Ruff and whitespace checks pass. A real MuJoCo regression exercises an asymmetric
rotated welded child, center-targeted lift and exact replay after an intervening
rotation, including JSON metadata round-trip and missing-origin rejection.

Live source `23fc453e`: job `20261012_212116_b666b0`, artifacts
`/home/cpaxton/runs/emet/tamp-oracle-origins-20261012/`. Two independent workers
run Stretch cleanup/navigation reference admission, then fresh replay of each
admitted schema-2 certificate. Cleanup admitted and replayed successfully;
navigation rejected scatter until the independent-instance fix below.

## Independent-object scatter

The first coordinate-fixed navigation admission still rejected its scatter.
Diagnostic repeat `20261012_212524_ad151b` reproduced the failure: the two kettle
parts `..._1_1_0` and `..._1_2_0` were requested at different ring positions.
Moving the second part moved the whole assembly, displacing the first by 0.407 m.
This is not an IK or motion-planning failure. The cleanup control again admitted
and replayed successfully.

The generator now preserves metadata instance identity and selects one body per
instance, excluding the entire receptacle instance. This is topology-based, not
a kettle/category special case. It retains multiple independent instances of
the same category and still requests the original number of objects. The new
instance-selection and fixture suite passes 18 checks. Final live source
`8870180a`, job `20261012_212909_6d7c09`, rebuilds both fixtures before replay.

## Final acceptance

Job `20261012_212909_6d7c09` completed on source `8870180a`:

| Case | Reference admission | Fresh MCTS replay | Replay wall time |
| --- | --- | --- | --- |
| Stretch cleanup | admitted | pass, 3/3 relocated | 46.5 s |
| Stretch navigation | admitted | pass, 8/8 relocated and goal reached | 55.8 s |

Thus 2/2 requested construction cases admitted and 2/2 admitted fixtures passed
fresh replay. All eight navigation objects reproduced requested scatter XY
within 0.2 mm. The final generator can select different independent objects
under its existing layout/size policies; this is new admission evidence, not a
rescore of old frozen fixtures or a reproduction of the historical 4/4.

Artifacts: `/home/cpaxton/runs/emet/tamp-oracle-origins-r3-20261012/`, including
exact source hash, commands, measurements, schema-2 certificates and results.
Earlier rejected runs remain in the adjacent unsuffixed and `r2` directories.
The oracle compatibility gate for #203 is satisfied. Baseline Nori self-contact
failures and native observed/contact execution remain separate unresolved work.
