"""Archive signed-error evidence, guard comparison and fixed training distribution."""
import gzip, hashlib, json, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emet.rl.manipulation.mastery import read_report,load_suite,select_cases,validate_preflight
from emet.rl.regression_gate import verify_regressions

root=Path.cwd();recovery=root/'logs/recovery/yaw-sign-20261030';source=root/'logs/eval/mars-yaw-sign-20261030-v2'
out=root/'paper/data/whole_body_control/yaw_sign_guard_v1';out.mkdir(exist_ok=True,parents=True)
receipt=verify_regressions(recovery/'regressions-v2.receipt.json')
summary={'controller_tests':len(receipt['test_ids']),'hardware_used':False,'training_started':False,'safety_promoted':False,'variants':{}}
reports={}
for variant in ['unguarded','guarded']:
    for name in ['suite.json','teacher.json.gz','policy.json.gz','feedback.json.gz','summary.json']:
        shutil.copy2(source/variant/name,out/(variant+'-'+name))
    report=read_report(source/variant/'policy.json.gz');reports[variant]=report
    teacher=read_report(source/variant/'teacher.json.gz')
    summary['variants'][variant]={'teacher_passed':sum(c['mastered'] for c in teacher['cases']),'cases':len(report['cases']),'mastered':sum(c['mastered'] for c in report['cases']),'tracking_and_holding':sum(c['tracking_and_settling_pass'] for c in report['cases']),'codes':{},'faults':[]}
    for case in report['cases']:
        codes=summary['variants'][variant]['codes'];codes[case['code']]=codes.get(case['code'],0)+1
        if case['final_info'].get('motion_fault'):
            summary['variants'][variant]['faults'].append({'id':case['id'],'time':case['trace'][-1]['time'],'evidence':case['final_info']['turn_guard_evidence']})
for a,b in zip(reports['unguarded']['cases'],reports['guarded']['cases'],strict=True):
    assert a['id']==b['id']
    for key in ['initial_state_sha256','reference_sha256']:assert a['admission'][key]==b['admission'][key],a['id']
summary['matched_initial_states_and_references']=True
suite=load_suite(root/'configs/benchmarks/mars_tracking_mastery_yaw_curriculum_v1.json')
preflight=read_report(source/'teacher-admission/report.json.gz');validate_preflight(preflight,suite,['base'])
summary['training_cases']=len(select_cases(suite,['base'],'train',population='training'));summary['canonical_evaluation_cases']=len(select_cases(suite,['base'],'all'));summary['teacher_admission_passed']=sum(c['mastered'] for c in preflight['cases'])
shutil.copy2(source/'teacher-admission/report.json.gz',out/'teacher-admission.json.gz')
for name in ['report.json.gz','summary.json']:shutil.copy2(source/'guarded-canonical'/name,out/('guarded-canonical-'+name))
canonical=read_report(source/'guarded-canonical/report.json.gz')
summary['guarded_canonical']={'cases':len(canonical['cases']),'mastered':sum(c['mastered'] for c in canonical['cases']),'tracking_and_holding':sum(c['tracking_and_settling_pass'] for c in canonical['cases'])}
for name in ['regressions-v2.xml','regressions-v2.receipt.json','head-decomposition.json','probe.py','sweep-v2.py']:shutil.copy2(recovery/name,out/name)
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
postures=['hold','forward','reverse','yaw_left','yaw_right'];turns=[.02,-.02,.04,-.04,.06,-.06]
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,variant in zip(axes,['unguarded','guarded']):
    matrix=np.zeros((5,6));labels={}
    for c in reports[variant]['cases']:
        i=postures.index(c['id'].split('@posture:')[1]);j=turns.index(c['turn_rad'])
        matrix[i,j]=1000*c['metrics']['base_heading_error_rad']
        label='P' if c['mastered'] else 'T' if c['tracking_and_settling_pass'] else 'STOP' if c['final_info'].get('motion_fault') else 'F'
        labels[i,j]=f'{matrix[i,j]:.1f}\n{label}'
    im=ax.imshow(matrix,vmin=0,vmax=80,cmap='YlOrRd')
    for (i,j),label in labels.items():ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>45 else 'black')
    ax.set_xticks(range(6),[f'{x:+.2f}' for x in turns]);ax.set_yticks(range(5),postures);ax.set_xlabel('Signed yaw target (rad)');ax.set_ylabel('Training reset posture');ax.set_title(variant)
fig.colorbar(im,ax=axes,label='Final absolute yaw error (mrad); tracking tolerance = 10 mrad')
fig.suptitle('P: full mastery; T: tracking/hold only; F: failure; STOP: guard fault\nGuard faults remain failures; stopping early is not target achievement',fontsize=10)
fig.savefig(out/'yaw-grid.svg');plt.close(fig)
shutil.copy2(__file__,out/'archive.py')
checks={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name!='files-sha256.json'}
(out/'files-sha256.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
