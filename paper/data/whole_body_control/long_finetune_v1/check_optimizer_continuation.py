"""Verify actual saved optimizer clocks include source history plus new updates."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path.cwd();base=ROOT/'logs/manipulation/quiet-reference-pilot-v1/ordinary_noise/training.pt';current=ROOT/'logs/manipulation/long-finetune-v1/continued/training.pt';a=torch.load(base,map_location='cpu',weights_only=False);b=torch.load(current,map_location='cpu',weights_only=False)
rows=[json.loads(x) for x in (current.parent/'updates.jsonl').read_text().splitlines() if json.loads(x)['decisions']<=b['decisions']];counts=[sum(x['optimizer_steps'] for x in rows),sum(x['critic_optimizer_steps'] for x in rows)]
if len(a['optimizer']['param_groups'])!=2 or len(b['optimizer']['param_groups'])!=2:raise RuntimeError('optimizer groups differ')
checked=0;source_steps=[]
for index,(old,new) in enumerate(zip(a['optimizer']['param_groups'],b['optimizer']['param_groups'],strict=True)):
 if old['params']!=new['params']:raise RuntimeError('optimizer parameter order differs')
 for parameter in old['params']:
  before=a['optimizer']['state'].get(parameter);after=b['optimizer']['state'].get(parameter)
  if before is None or after is None:raise RuntimeError('optimizer state missing')
  source_step=float(before['step']);actual=float(after['step']);expected=source_step+counts[index]
  if actual!=expected:raise RuntimeError(f'optimizer clock reset or skipped parameter: {actual} != {expected}')
  checked+=1;source_steps.append(source_step)
report={'format':'optimizer-continuation-check-v1','new_decisions':b['decisions'],'source_decisions':a['decisions'],'new_actor_steps':counts[0],'new_critic_steps':counts[1],'optimizer_states_checked':checked,'all_clocks_include_source_history':True,'source_step_range':[min(source_steps),max(source_steps)],'source_checkpoint_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'scope':'actual saved optimizer clocks, not exact simulator/RNG continuation'}
(ROOT/'logs/recovery/long-finetune-v1/optimizer-continuation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
