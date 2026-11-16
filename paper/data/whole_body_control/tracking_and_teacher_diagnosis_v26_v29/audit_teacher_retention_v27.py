"""Retiming diagnostic for teacher retention failures; original baseline unchanged."""
import gzip,hashlib,json
from pathlib import Path
from dataclasses import asdict
import numpy as np,torch
from emet.rl.manipulation.env import ManipulationConfig,ManipulationEnv
from emet.rl.manipulation.dynamic_trajectory import ArmTrajectory,Segment
from emet.rl.manipulation.carry_benchmark import source_hashes
from emet.rl.regression_gate import verify_regressions

torch.set_num_threads(1);verify_regressions(Path('logs/recovery/20261028/failure-replay-regressions-v2.receipt.json'))
root=Path('logs/eval/failure-replay-development-v1-20261028');original=json.load(open(root/'teacher/report.json'));cfg=ManipulationConfig(**original['config']);out=Path('logs/eval/teacher-retention-v27-20261028');out.mkdir(exist_ok=False,parents=True)
r={'scope':__doc__,'source_hashes':source_hashes(),'config':asdict(cfg),'cases':[],'complete':False,'hardware_used':False,'safety_promoted':False,'driver_source':Path(__file__).read_text()}
def save():
 raw=(json.dumps(r,allow_nan=False)+'\n').encode();(out/'report.json.gz').write_bytes(gzip.compress(raw,mtime=0));(out/'summary.json').write_text(json.dumps({'complete':r['complete'],'scope':__doc__,'uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'cases':[{k:c.get(k) for k in ['seed','time_scale','code','success','retention_lost','wheel_lift','position_error_m','orientation_error_rad','duration_s','initial_state_matches_original']} for c in r['cases']]},indent=2)+'\n')
for parent in [c for c in original['cases'] if not c['initialization_failed'] and not c['final_info']['is_success']]:
 for scale in [1.,1.25,1.5,2.]:
  env=ManipulationEnv(cfg)
  try:
   obs,info=env.reset(seed=parent['seed']);assert not info['initialization_failed'];assert env.grasp_lift_admission['initial_state']==parent['admission']['initial_state'];path=env.arm_trajectory
   case={'seed':parent['seed'],'time_scale':scale,'original_failure':parent['final_info']['code'],'initial_state_matches_original':True,'admission':env.grasp_lift_admission,'duration_s':path.end*scale,'trace':[]};r['cases'].append(case)
   if path.end*scale+2>cfg.horizon:
    case.update(code='retimed_reference_exceeds_original_horizon',success=False);save();continue
   env.arm_trajectory=ArmTrajectory([Segment(p.start*scale,p.duration*scale,p.coefficients/(scale**np.arange(6))[:,None]) for p in path.segments])
   assert np.allclose(env.arm_trajectory.sample(env.arm_trajectory.end)[0],np.asarray(parent['admission']['goal']),atol=1e-10)
   for _ in range(round(cfg.horizon/env.profile.dt)+1):
    obs,reward,done,trunc,info=env.step(env.teacher_action());case['trace'].append({'time':info['time'],'code':info['code'],'hand':env.control.ee_pose().tolist(),'base':env.control.base_pose().tolist(),'position_error_m':info['ee_position_error'],'orientation_error_rad':info['ee_orientation_error'],'payload_rotation_rad':info['current_payload_rotation'],'payload_translation_m':info['current_payload_translation'],'peak_contact_force_N':info['peak_contact_force'],'lift_m':float(env.data.xpos[env.payload,2]-env.initial_object_height)})
    if done or trunc:break
   case.update(code=info['code'],success=bool(info['is_success']),retention_lost=bool(info['retention_lost']),wheel_lift=bool(info['wheel_lift']),position_error_m=info['ee_position_error'],orientation_error_rad=info['ee_orientation_error'],final_info=info);save();print('TEACHER',parent['seed'],scale,case['code'],flush=True)
  finally:env.close()
assert r['source_hashes']==source_hashes();r['complete']=True;save();print('TEACHER_RETENTION_COMPLETE',flush=True)
