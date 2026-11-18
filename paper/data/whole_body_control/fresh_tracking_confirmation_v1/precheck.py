"""Precheck nominal two-stage C2 geometry in scratch state without running actors."""
import copy,json
from dataclasses import replace
from pathlib import Path
import numpy as np
from emet.rl.manipulation.coupled_tracking import CoupledTrackingEnv,CoupledReference,admit_reference
ROOT=Path.cwd();rec=ROOT/'logs/recovery/fresh-tracking-confirmation-v1';cases=json.loads((rec/'cases.json').read_text());protocol=json.loads((rec/'protocol.json').read_text());results=[]
for case in cases:
 setting=protocol['settings'][case['id']]
 if 'second_stage' not in setting:continue
 env=CoupledTrackingEnv('innate_mars',[case],suite_path=rec/'suite.json');e=env.unwrapped;e.config=replace(e.config,wheel_feedback_base_only_gains=(0.,.5),initial_heading_range=setting['initial_heading_range_rad'])
 try:
  _,info=env.reset();first=e.control.task_trajectory;before=e.data.qpos.copy();base,joints=first.coordinates(first.end);first.fk(base,joints);proxy=copy.copy(e.control);proxy.data=first.fk.data;s=setting['second_stage'];second=CoupledReference(proxy,dict(case,travel_m=s['travel_m'],turn_rad=s['turn_rad'],arm_delta=s['arm_delta']));admission=admit_reference(e,second)
  for a,b in zip(first.sample(first.end),second.sample(0),strict=True):np.testing.assert_allclose(a,b,atol=1e-9,rtol=0)
  for a,b in zip(first.path.sample(first.end)[1:3],second.path.sample(0)[1:3],strict=True):np.testing.assert_allclose(a,b,atol=1e-8,rtol=0)
  np.testing.assert_array_equal(before,e.data.qpos.copy());results.append({'id':case['id'],'first_admission':info['coupled_admission'],'second_admission':admission,'C2_join_verified':True,'live_qpos_unchanged':True})
 finally:e.close()
(rec/'precheck.json').write_text(json.dumps({'cases':results,'scope':'scratch construction only, all geometry outcomes retained; no controller evaluation'},indent=2)+'\n');print(json.dumps(results,indent=2))
