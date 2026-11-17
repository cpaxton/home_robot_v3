"""Frozen gain-field diagnostic: base acceptance region in feedback, not context."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.coupled_tracking import load_suite,COARSE_SUITE
from emet.rl.manipulation.whole_body_bundle import WholeBodyBundle
from emet.rl.regression_gate import ROOT,source_state,verify_regressions
import importlib.util
REC=ROOT/'logs/recovery/critic-architecture-audit-v1';OUT=ROOT/'logs/eval/critic-architecture-region-probe-v1'
spec=importlib.util.spec_from_file_location('pilot',ROOT/'logs/recovery/wrapper-reward-training-v2/run.py');pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
class Region:
 def __init__(self,actor):self.actor=actor;self.observation=None;self.handle=actor.gain.register_forward_hook(self.hook)
 def hook(self,module,args,value):
  obs=self.observation;f,pf=self.actor.encoder.fields,self.actor.encoder.preview_fields
  active=obs['history'][:,-1,f['hand_mask']].squeeze(-1)>.5
  gain=value.reshape(-1,self.actor.encoder.profile.action_size-1,self.actor.feedback_width).clone()
  for row,start in ((1,0),(4,12)):
   raw=obs['preview'][:,row,pf['base_error']];norm=raw[:,:2].norm(dim=-1)
   xy=(1-.05/norm.clamp_min(1e-12)).clamp_min(0);yaw=(1-np.deg2rad(5)/raw[:,2].abs().clamp_min(1e-12)).clamp_min(0)
   scale=torch.stack([xy,xy,yaw],-1);scale=torch.where(active[:,None],scale,torch.ones_like(scale));gain[:,:,start:start+3]*=scale[:,None]
  return gain.reshape_as(value)
 def __call__(self,obs):self.observation=obs;return self.actor(obs)
def main():
 torch.set_num_threads(1);verify_regressions(ROOT/'logs/eval/wrapper-reward-regressions-v1/full.receipt.json');state=source_state(ROOT);OUT.mkdir(parents=True,exist_ok=False)
 all_cases=json.loads((ROOT/'logs/recovery/wrapper-reward-training-v2/cases.json').read_text());cases=[c for c in all_cases if c['id'].endswith((':train:hand:3',':audit:hand:1',':audit:hand:2'))]+[c for c in load_suite(COARSE_SUITE)['robots']['innate_mars'] if c['id'].endswith((':3',':4',':7'))]
 bundle=WholeBodyBundle(ROOT/'logs/eval/scoped-wheel-feedback-v2/candidate/actor');before={k:v.clone() for k,v in bundle.actor.state_dict().items()};report={'format':'base-region-feedback-diagnostic-v1','source_state_digest':digest(state),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'weights_sha256':bundle.manifest['weights_sha256'],'cases':cases,'results':[],'complete':False,'actor_updated':False,'exported':False,'scope':'same raw observations and unchanged context; only hand-active base feedback uses distance outside 5cm/5deg acceptance region; not a supported runtime or safety-qualified controller'}
 for mode in ('baseline','region_feedback'):
  actor=bundle.actor if mode=='baseline' else Region(bundle.actor);env=pilot.make_general(cases,None)
  try:
   for case in cases:
    row=pilot.episode(env,case,actor);report['results'].append({'mode':mode,**row});(OUT/'results.json').write_text(json.dumps(report,indent=2,default=pilot.json_value)+'\n');print('REGION',mode,case['id'],row['success'],row['code'],'mm',1000*row['metrics']['ee_position_error'],flush=True)
  finally:
   env.close()
   if mode!='baseline':actor.handle.remove()
 for k,v in bundle.actor.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
 if digest(source_state(ROOT))!=digest(state):raise RuntimeError('source changed')
 report['complete']=True;(OUT/'results.json').write_text(json.dumps(report,indent=2,default=pilot.json_value)+'\n')
if __name__=='__main__':main()
