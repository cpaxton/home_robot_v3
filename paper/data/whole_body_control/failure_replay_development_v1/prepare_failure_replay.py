"""New development families; old assessment seeds are never training parents."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,torch
from emet.rl.manipulation.env import ManipulationConfig
from emet.rl.manipulation.policy import ManipulationPolicy
from emet.rl.manipulation.grasp_lift import export_cases
from emet.rl.manipulation.failure_replay import mine_failures,validate_corpus

torch.set_num_threads(1)
root=Path('logs/eval/failure-replay-development-v1-20261028');root.mkdir(exist_ok=False,parents=True)
spec=json.load(open('configs/benchmarks/mars_jerk_tracking_curriculum_v1.json'))
cfg=replace(ManipulationConfig(**spec['environment']),physical_wheel_slew=True,hand_gated_base_reward=False,tracking_precision=4,motion_penalty_scale=1,smoothness_curriculum=False,scenario_weights=(.70,.04,.20,.04,.02),grasp_lift_phase_weights=None)
(root/'config.json').write_text(json.dumps({'environment':cfg.__dict__},indent=2)+'\n')
bundle=Path('logs/manipulation/mars-jerk-tracking-v20/control/seed-86/last');actor=ManipulationPolicy(bundle)
report=export_cases(cfg,root/'policy',samples=20,seed=33_000_000,policy=actor)
teacher=export_cases(cfg,root/'teacher',samples=20,seed=33_000_000,teacher=True)
corpus=mine_failures(report,teacher,goal_jitter_rad=.005);validate_corpus(corpus,cfg)
(root/'corpus.json').write_text(json.dumps(corpus,indent=2)+'\n')
summary={'scope':__doc__,'complete':True,'parent_policy':str(bundle),'parent_weights_sha256':actor.manifest['weights_sha256'],'corpus_sha256':hashlib.sha256((root/'corpus.json').read_bytes()).hexdigest(),'families':corpus['families'],'teacher_unverified_parent_seeds':corpus['teacher_unverified_parent_seeds'],'rejected_parent_seeds':corpus['rejected_parent_seeds'],'training_started':False,'safety_promoted':False}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('CORPUS_COMPLETE',json.dumps(summary),flush=True)
