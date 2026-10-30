"""Freeze baseline/intervention evidence and launch settings, never weights."""
from pathlib import Path
import hashlib,json,shutil
from emet.rl.manipulation.mastery import load_suite,read_report,select_cases,validate_preflight
from emet.rl.regression_gate import verify_regressions
root=Path.cwd();rec=root/'logs/recovery/restoring-yaw-v1';run=root/'logs/manipulation/mars-mastery-restoring-yaw-v1-seed90'
out=root/'paper/data/whole_body_control/restoring_yaw_v1_setup';out.mkdir(exist_ok=True,parents=True)
suite=load_suite(root/'configs/benchmarks/mars_tracking_mastery_restoring_yaw_v1.json')
teacher=root/'logs/eval/mars-restoring-yaw-v1-teacher';validate_preflight(read_report(teacher/'report.json.gz'),suite,['base'])
receipt=verify_regressions(rec/'regressions.receipt.json')
for source,prefix in [(teacher,'teacher'),(root/'logs/eval/mars-restoring-yaw-v1-baseline','baseline')]:
 for name in ['report.json.gz','summary.json']:shutil.copy2(source/name,out/(prefix+'-'+name))
for name in ['manifest.json','evaluation.json.gz','summary.json','admission-evaluation.json.gz','admission-summary.json']:
 shutil.copy2(run/'step-0'/name,out/('initial-'+name))
shutil.copy2(run/'run.json',out/'run.json')
for name in ['regressions.xml','regressions.receipt.json','launch.py']:shutil.copy2(rec/name,out/name)
base=read_report(out/'baseline-report.json.gz');new=read_report(out/'initial-admission-evaluation.json.gz');canonical=read_report(out/'initial-evaluation.json.gz')
for a,b in zip(base['cases'],new['cases'],strict=True):
 assert a['id']==b['id']
 for key in ['initial_state_sha256','reference_sha256']:assert a['admission'][key]==b['admission'][key],a['id']
summary={'controller_tests':len(receipt['test_ids']),'cases':len(new['cases']),'teacher_passed':50,'training_cases':len(select_cases(suite,['base'],'train',population='training')),'initial_structural_intervention_not_learning':True,'matched_initial_states_and_references':True,'baseline_full_mastery':sum(c['mastered'] for c in base['cases']),'baseline_tracking_and_holding':sum(c['tracking_and_settling_pass'] for c in base['cases']),'initial_full_mastery':sum(c['mastered'] for c in new['cases']),'initial_tracking_and_holding':sum(c['tracking_and_settling_pass'] for c in new['cases']),'canonical_initial_mastery':sum(c['mastered'] for c in canonical['cases']),'canonical_initial_tracking':sum(c['tracking_and_settling_pass'] for c in canonical['cases']),'checkpoint_safety_flags':{key:sum(bool(c['final_info'].get(key)) for c in new['cases']) for key in ['wheel_lift','unsafe_contact','invalid_physics']},'steps_planned':65536,'seed':90,'training_started':True,'hardware_used':False,'safety_promoted':False}
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
shutil.copy2(__file__,out/'archive_setup.py')
(out/'files-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name!='files-sha256.json'},indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
