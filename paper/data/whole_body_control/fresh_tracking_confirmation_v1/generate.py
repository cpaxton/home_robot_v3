"""Freeze independent reachable-reference draws before the new actor is evaluated."""
import json,hashlib
from pathlib import Path
import numpy as np
from emet.rl.manipulation.coupled_tracking import load_suite,COARSE_SUITE
ROOT=Path.cwd();OUT=ROOT/'logs/recovery/fresh-tracking-confirmation-v1';rng=np.random.default_rng(2026111707);original=load_suite(COARSE_SUITE);cases=[];settings={}
for i in range(24):
 mode='coupled' if i<8 or i>=20 else 'hand_free_base' if i<16 else 'base_only'
 sign=1 if i%2==0 else -1;travel=sign*float(rng.uniform(.12,.18)) if mode!='hand_free_base' else 0.;turn=(-1 if (i//2)%2 else 1)*float(rng.uniform(.18,.42)) if mode!='hand_free_base' else 0.
 delta=rng.uniform(-.04,.04,5).tolist() if mode!='base_only' else [0.]*5
 case={'id':f'innate_mars:confirmation-v1:audit:{mode}:{i}','seed':410300000+i,'split':'audit','mode':mode,'travel_m':travel,'turn_rad':turn,'arm_delta':delta,'base_mask':[False]*3 if mode=='hand_free_base' else [True,True,not(i<8 and i%4>=2)],'purpose':'tracking'}
 cases.append(case);setting={'initial_heading_range_rad':0. if i%2==0 else float(np.pi) if i in (7,19) else 1.}
 if i>=20:setting['second_stage']={'travel_m':-travel,'turn_rad':-turn,'arm_delta':(-np.asarray(delta)).tolist(),'after_nominal_end_s':.25}
 settings[case['id']]=setting
# Validate against the established loader, retaining original train identities.
suite=json.loads(Path(COARSE_SUITE).read_text());suite['robots']['innate_mars']=[c for c in original['robots']['innate_mars'] if c['split']=='train']+cases
(OUT/'suite.json').write_text(json.dumps(suite,indent=2)+'\n');load_suite(OUT/'suite.json')
prior=json.loads((ROOT/'logs/recovery/quiet-reference-pilot-v1/cases.json').read_text());assert not ({c['id'] for c in cases}&{c['id'] for c in prior});assert not ({c['seed'] for c in cases}&{c['seed'] for c in prior});assert len({c['seed'] for c in cases})==24
(OUT/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
protocol={'format':'fresh-tracking-confirmation-v1','runtime_revision':'9d0620f9c3d3e198494b0a02ca3ce78513b4d04c','generator_seed':2026111707,'cases':24,'source_training_namespace_disjoint':True,'settings':settings,'case_sha256':hashlib.sha256((OUT/'cases.json').read_bytes()).hexdigest(),'selection':'all generated cases retained, including geometry rejections and teacher failures; never enter PPO','scope':'nearby FK references with existing bounded posture resets, varied initial headings, full/XY-only/free-base masks and four nominally anchored two-stage goal changes; unloaded','two_stage_contract':'second reference compiled in reset-only scratch data at first nominal endpoint; pose/velocity/acceleration continuous at announced switch; targets independent of controller state','new_actor_evaluation':'fixed final continued checkpoint only, after training completes; no interim checkpoint selection','hardware_used':False,'safety_promoted':False}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print('24 disjoint cases validated; four controller-independent two-stage plans')
