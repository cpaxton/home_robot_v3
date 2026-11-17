"""Case-held-out finite-return fitting, frozen actor, no policy update."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path.cwd();OUT=ROOT/'logs/eval/critic-architecture-audit-v1'
def stats(y,p):
 e=p-y;variance=y.var(0);return {'rmse':np.sqrt((e**2).mean(0)).tolist(),'bias':e.mean(0).tolist(),'explained_variance':[float(1-e[:,i].var()/v) if v>1e-12 else None for i,v in enumerate(variance)]}
def save(r):(OUT/'fit.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def main():
 torch.set_num_threads(1);assert json.loads((OUT/'complete.json').read_text())['complete'];episodes=json.loads((OUT/'episodes.json').read_text());arrays=[]
 for i,episode in enumerate(episodes):
  with np.load(OUT/f'episode-{i:02d}.npz',allow_pickle=False) as a:arrays.append({k:a[k].copy() for k in a.files})
 train=np.concatenate([np.full(e['ticks'],e['case']['split']=='train') for e in episodes]);features=np.concatenate([a['features'] for a in arrays]);history=np.concatenate([a['history'].reshape(len(a['features']),-1) for a in arrays]);state=np.concatenate([a['reward_state'] for a in arrays])
 report={'format':'critic-finite-return-fit-audit-v1','actor_updated':False,'scope':'supervised finite-return fit on fixed retained-actor stochastic traces; same traces for all reward profiles; critic comparisons use fixed held-out cases, not PPO performance','samples':len(train),'train_samples':int(train.sum()),'audit_samples':int((~train).sum()),'updates_per_model':500,'batch_size':128,'training_seed':311,'target':'full finite-episode discounted return; single sampled future, not expected state value','normalization':'training-only feature/target statistics; feature z scores clipped to +/-20','architecture_confounds':'history/reward-state models have more input parameters; improved fit alone would not isolate capacity from observability','results':[]};save(report)
 for profile in ('legacy','contract'):
  y=np.concatenate([a['returns_'+profile] for a in arrays]);ym=y[train].mean(0);ys=y[train].std(0).clip(1e-3)
  for name in ('original_raw','normalized_input','normalized_input_target','history','reward_state'):
   x=features if name in ('original_raw','normalized_input','normalized_input_target') else np.concatenate([features,history if name=='history' else state],axis=1)
   if name!='original_raw':x=((x-x[train].mean(0))/x[train].std(0).clip(1e-4)).clip(-20,20)
   normalized=name in ('normalized_input_target','history','reward_state');offset=ym if normalized else np.zeros(3);scale=ys if normalized else np.ones(3);target=(y-offset)/scale
   torch.manual_seed(311);model=nn.ModuleList([nn.Sequential(nn.Linear(x.shape[1],128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,1)) for _ in range(3)]);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
   xt=torch.from_numpy(x.astype(np.float32));yt=torch.from_numpy(target.astype(np.float32));indices=torch.from_numpy(np.flatnonzero(train));rng=torch.Generator().manual_seed(411)
   for _ in range(500):
    ids=indices[torch.randint(len(indices),(128,),generator=rng)];prediction=torch.cat([m(xt[ids]) for m in model],-1);loss=.5*((prediction-yt[ids])**2).mean(0).sum();opt.zero_grad();loss.backward()
    for m in model:torch.nn.utils.clip_grad_norm_(m.parameters(),.5)
    opt.step()
   with torch.no_grad():p=torch.cat([m(xt) for m in model],-1).numpy()*scale+offset
   row={'profile':profile,'features':name,'input_width':x.shape[1],'train':stats(y[train],p[train]),'heldout_cases':stats(y[~train],p[~train]),'late_hold_heldout':stats(y[(~train)&(state[:,-1]>.6)],p[(~train)&(state[:,-1]>.6)]),'training_contact_return_nonzero_samples':int(np.count_nonzero(y[train,2]))}
   report['results'].append(row);save(report);print('FIT',profile,name,'train RMSE',row['train']['rmse'],'heldout',row['heldout_cases']['rmse'],flush=True)
 report['complete']=True;save(report)
if __name__=='__main__':main()
