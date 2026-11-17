"""Describe stochastic pose/quiet occupancy; do not infer a safety certificate."""
import json
from pathlib import Path
import numpy as np
from emet.rl.manipulation.observation_layout import preview_layout
ROOT=Path.cwd();OUT=ROOT/'logs/eval/critic-architecture-audit-v1'
def main():
 episodes=json.loads((OUT/'episodes.json').read_text());rows=[];fields,_=preview_layout()
 for i,e in enumerate(episodes):
  if not e['truncated']:continue
  with np.load(OUT/f'episode-{i:02d}.npz',allow_pickle=False) as a:
   h=a['history'];features=a['features'];state=a['reward_state'];d=a['diagnostics'];preview=features[:,h.shape[-1]:h.shape[-1]+126].reshape(-1,7,18)
   late=state[:,-1]>=.6;active=preview[:,1,fields['hand_mask']].reshape(-1)>.5;base_error=preview[:,1,fields['base_error']]*[.1,.1,.5]
   hand_pose=(~active)|((d[:,1]<=.003)&(d[:,2]<=.02));base_pose=(np.linalg.norm(base_error[:,:2],axis=-1)<=.05)&(abs(base_error[:,2])<=np.deg2rad(5));pose=hand_pose&base_pose
   velocity=h[:,-1,-9:-3]*[.12,.12,.12,.5,.5,.5]
   quiet=(d[:,4]<.03)&(np.linalg.norm(velocity[:,:3],axis=-1)<.005)&(np.linalg.norm(velocity[:,3:],axis=-1)<.02)
   both=pose&quiet&(state[:,-2]>.5)
   count=best=0
   for valid in both[late]:count=count+1 if valid else 0;best=max(best,count)
   rows.append({'case_id':e['case']['id'],'noise_seed':e['noise_seed'],'split':e['case']['split'],'hand_active':bool(active.any()),'late_samples':int(late.sum()),'late_pose_fraction':float(pose[late].mean()),'late_quiet_fraction':float(quiet[late].mean()),'late_pose_and_quiet_fraction':float(both[late].mean()),'max_native_settle_seconds':float(d[:,3].max()),'longest_shadow_late_hold_seconds':float(best*.05),'final_hand_position_error_m':float(d[-1,1])})
 report={'format':'stochastic-settling-audit-v1','scope':'shadow occupancy from saved pre-action samples; mean-policy baseline remains a separate native readout; no policy update','action_std':.02,'late_window':'final 40% of fixed 25s contract','required_hold_seconds':2,'episodes':rows,'summary':{'deadline_episodes':len(rows),'ever_native_held_2s':sum(r['max_native_settle_seconds']>=2 for r in rows),'majority_late_pose_in_contract':sum(r['late_pose_fraction']>.5 for r in rows),'majority_late_quiet':sum(r['late_quiet_fraction']>.5 for r in rows)}}
 (OUT/'settling.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
