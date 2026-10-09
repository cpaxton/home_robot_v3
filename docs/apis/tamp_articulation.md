# Assisted receptacle open/close

`set_receptacle_state` is a CHAT tool for the in-process MuJoCo simulator.
It uses the same schema-v1 JSON envelope as the other [TAMP tools](tamp.md).
It teleports a fixture joint: this is **simulator assistance**, not learned or
physical opening, and does not certify a collision-free door sweep or access.

```json
{"name":"set_receptacle_state","arguments":{"task_ref":"task:...","state":"open"}}
```

`state` is `open` or `closed`. The task handle comes from the current
`scene_tasks` response and selects that task's receptacle. Raw body and joint
names stay private. Use discovery → open → plan → execute → close, checking
`status` after every action. Open before pickup; close only when the robot and
payload are clear. There is no automatic close after failure or automatic retry. This is a separate
agent-callable action, not yet an operator selected by the internal MCTS search.

An example successful result:

```json
{"schema_version":1,"tool":"set_receptacle_state","status":"ok","code":"ok","message":"ok","data":{"assistance":["joint_teleport"],"collision_scope":"none","requested_state":"open","verified":true,"observed_state":"open","convention":"default_limit_is_closed"},"recovery":"none"}
```

Success requires a matching per-request simulator result and current measured
joint state. A transport receipt alone is insufficient. The verification deadline
is 10 wall seconds after sending, in addition to the transport acknowledgement
budget. Timeout/command errors after attempting a command return `partial` and
require inspection. Unsupported/hardware/stale-handle requests fail before sending.
All attempted actions invalidate cached plans, including uncertain outcomes.
Server articulation revisions also invalidate plans in other tool contexts.

## Supported geometry and conservative preflight

The simulator groups scene bodies by top-level fixture, excluding the robot.
A fixture is actionable only when it has exactly one bounded hinge or slide,
with its model default position at one endpoint. That endpoint is called closed;
the opposite endpoint is called open. This convention is an explicit approximation,
not semantic proof that a real mechanism opens that way. State tolerance is 2%
of joint range (minimum 1e-5 in joint units).

Multi-joint fixtures, free objects with articulated children, unbounded joints,
and interior default positions are unsupported. No category-specific exceptions
exist. `scene_tasks.data.tasks[].access` reports `state` and `required_action`.
For providers advertising `sim_articulation_state`, planning and stored-plan
execution reject closed/partial articulated receptacles with
`receptacle_requires_open`, or unsupported ones with `articulation_unsupported`,
before pickup. Unknown metadata returns `articulation_state_unavailable`.
This conservatively gates the entire fixture, including its exterior support tops.
It does not establish access to the object being picked up.

Joint-state publication refreshes affected scene geometry using private MuJoCo
data; live contact/acceleration buffers are not overwritten by metadata refresh.
The existing joint teleport itself still calls MuJoCo forward dynamics. Door
motion can cause contacts or subsequent settling: no swept collision protection,
attachment verification, containment guarantee or stable-open latch is claimed.

Only the in-process server advertises the new state capability. Older providers
and the Stretch subprocess report unknown access and reject this action; they
cannot automatically filter articulated targets through this metadata. Physical
execution mode rejects joint teleport using the existing actuation audit.

## Validation and metric separation

Run a separately scored live open/close cycle through CHAT with:

```bash
PYTHONPATH=src .venv/bin/python scripts/scripted_sim_pick_place.py \
  --start-sim --sim configs/sim/molmospaces_ithor_train_0.yaml \
  --manip-mode kinematic --object bowl --cpu-only --articulation-cycle
```

Use the managed GPU job runner for simulator startup. The cycle discovers the
first task, opens and closes its receptacle, and emits a JSON
`assisted_articulation_cycle` result with `placement_tested:false`. It cannot
be combined with arbitrary `--tool-calls-json`; ordinary pick/place verification
remains unchanged. Unsupported first tasks fail rather than being skipped.

A composed open → pick/place → close run is a new assisted track. Retain the
original closed-goal failures, record joint teleport assistance, and do not pool
this score with physical manipulation or the prior no-opening baseline.
