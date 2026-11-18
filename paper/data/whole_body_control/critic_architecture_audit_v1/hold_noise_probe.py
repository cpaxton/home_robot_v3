"""Frozen-actor probe of observation-conditioned exploration, not action smoothing."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.coupled_tracking import CoupledTrackingEnv,COARSE_SUITE
from emet.rl.manipulation.observation_layout import preview_layout
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.regression_gate import ROOT,source_state,verify_regressions
OUT=ROOT/'logs/eval/hold-exploration-probe-v1'
def main():
 torch.set_num_threads(1);verify_regressions(ROOT/'logs/eval/finite-critic-regressions-v2/full.receipt.json');state=source_state(ROOT);OUT.mkdir(parents=True,exist_ok=False)
 selected=json.loads((ROOT/'logs/eval/critic-architecture-audit-v1/protocol.json').read_text())['cases'];bundle=WholeBodyBundle(ROOT/'logs/eval/scoped-wheel-feedback-v2/candidate/actor');before={k:v.clone() for k,v in bundle.actor.state_dict().items()};fields,_=preview_layout()
 report={'format':'hold-exploration-probe-v1','source_state_digest':digest(state),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'actor_weights_sha256':bundle.manifest['weights_sha256'],'noise_seed':108,'hold_scale':.125,'scope':'same frozen actor, same noise stream, original Gaussian std .02; only a constant active-target preview with zero announced twist lowers motion-action exploration; mean authority and native guards remain unchanged','actor_updated':False,'results':[],'complete':False}
 for mode in ('original_noise','quiet_reference_noise'):
  for case in selected:
   env=CoupledTrackingEnv('innate_mars',[case],suite_path=COARSE_SUITE);e=env.unwrapped
   from dataclasses import replace
   e.config=replace(e.config,wheel_feedback_base_only_gains=(0.,.5));rng=np.random.default_rng(108);quiet_ticks=0;states=hashlib.sha256()
   try:
    obs,_=env.reset()
    for tick in range(500):
     sensor={k:torch.from_numpy(v)[None] for k,v in obs.items() if k in ('history','preview','motion')}
     with torch.no_grad():mean=bundle.actor(sensor)[0].numpy().copy();mask=bundle.actor.action_mask(sensor)[0].numpy()
     previews=obs['preview'][1:];names=('base_error','base_mask','hand_error','hand_mask');constant=all(np.array_equal(previews[:,fields[k]],np.broadcast_to(previews[:1,fields[k]],previews[:,fields[k]].shape)) for k in names);quiet_reference=constant and bool(np.all(obs['motion']==0));scale=.125 if mode=='quiet_reference_noise' and quiet_reference else 1.
     quiet_ticks+=int(scale<1);action=(mean+rng.normal(0,.02,len(mean))*scale)*mask
     obs,_,done,truncated,info=env.step(action);states.update(np.r_[e.data.qpos.copy(),e.data.qvel.copy()].tobytes())
     if done or truncated:break
    m=e._info();e.control.stop();start=float(e.data.time)
    while not e._physics_invalid and not whole_body_stopped(e.control) and float(e.data.time)-start<.75-1e-9:e._tick(np.zeros(e.profile.action_size),score=False)
    post=e._info();row={'mode':mode,'case':case,'success':bool(info.get('is_success')),'code':info['code'],'ticks':tick+1,'reduced_noise_ticks':quiet_ticks,'state_trace_sha256':states.hexdigest(),'stop_confirmed':bool(whole_body_stopped(e.control) and not e._physics_invalid),'safety_flags':{k:bool(post[k]) for k in ('wheel_lift','unsafe_contact','retention_lost')}|{'invalid_physics':bool(e._physics_invalid)},'metrics':{k:m[k] for k in ('ee_position_error','ee_orientation_error','settled_seconds','time_to_first_settle','peak_hold_base_drift_m','peak_base_linear_jerk_m_s3','peak_base_yaw_jerk_rad_s3','peak_contact_force')}};report['results'].append(row);(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('NOISE',mode,case['id'],row['success'],row['code'],'settle',m['settled_seconds'],flush=True)
   finally:env.close()
 for k,v in bundle.actor.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
 if digest(source_state(ROOT))!=digest(state):raise RuntimeError('source changed')
 report['complete']=True;(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
