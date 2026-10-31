"""Read-only signed-error / learned yaw-head decomposition on existing cases."""
import json, math
from pathlib import Path
import numpy as np
import torch
from emet.rl.manipulation.mastery import MasteryEnv,load_suite
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.observation_layout import preview_layout

torch.set_num_threads(1)
root=Path.cwd();suite=load_suite(root/'configs/benchmarks/mars_tracking_mastery_crossed_v1.json')
policy=ManipulationPolicy(root/'logs/manipulation/mars-mastery-crossed-v1-seed90/last')
env=MasteryEnv(suite,stages=['base'],split='train',population='training')
fields,_=preview_layout();result=[]
try:
 for posture in ['hold','forward','reverse','yaw_left','yaw_right']:
  for primitive in ['yaw_left','yaw_right']:
   cid='train:base:'+primitive+('' if primitive==posture else '@posture:'+posture)
   obs,info=env.reset(options={'case_id':cid});assert not info['initialization_failed']
   rows=[]
   for step in range(240):
    values={k:np.clip((obs[k]-m)/np.sqrt(v+policy.manifest['epsilon']),-policy.manifest['clip'],policy.manifest['clip']).astype('float32') for k,(m,v) in policy.normalization.items()}
    with torch.no_grad():
     tensors,_=policy.model.policy.obs_to_tensor(values)
     features=policy.model.policy.extract_features(tensors)
     latent=policy.model.policy.mlp_extractor.forward_actor(features)
     head=policy.model.policy.action_net
     ctx,signal=latent[:,:-head.feedback_width],latent[:,-head.feedback_width:]
     gain=head.gain(ctx).reshape(-1,head.motion_size,head.feedback_width)
     terms=(gain[0,1]*signal[0]/math.sqrt(head.feedback_width)).numpy()
     action=head(latent)[0].numpy()
    expected=policy(obs);np.testing.assert_allclose(action,expected,atol=1e-8)
    target=env.control.preview_sampler(float(env.data.time))[0]
    error=float(np.arctan2(np.sin(target[2]-env.control.base_pose()[2]),np.cos(target[2]-env.control.base_pose()[2])))
    observed=float(obs['preview'][1,fields['base_error']][2]);assert abs(error-observed)<1e-7
    if step%10==0 or step==239:
     rows.append({'step':step,'time':env.control.trajectory_elapsed(),'world_yaw_error':error,'preview_yaw_error':observed,'normalized_yaw_error':float(values['preview'][1,3]),'yaw_action':float(action[1]),'signal':signal[0].numpy().tolist(),'gain_yaw_row':gain[0,1].numpy().tolist(),'yaw_terms':terms.tolist(),'group_terms':{name:float(terms[a:b].sum()) for name,a,b in [('current_base',0,3),('current_hand',3,12),('future_base',12,15),('future_hand',15,24),('arm_velocity',24,29),('base_velocity',29,35)]}})
    obs,_,done,truncated,info=env.step(action)
    if done or truncated:break
   final=rows[-1];print(cid,'error',round(final['world_yaw_error'],5),'action',round(final['yaw_action'],6),'groups',final['group_terms'],flush=True)
   result.append({'id':cid,'samples':rows,'final_code':info['code']})
finally:env.close()
(root/'logs/recovery/yaw-sign-20261030/head-decomposition.json').write_text(json.dumps({'cases':result,'weights_sha256':policy.manifest['weights_sha256'],'hardware_used':False},indent=2)+'\n')
