"""Archive final negative PPO results and independent diagnostic interventions."""
import gzip,hashlib,json,shutil
from pathlib import Path
out=Path('paper/data/whole_body_control/failure_replay_results_v25');out.mkdir(exist_ok=True)
root=Path('logs/eval/failure-replay-v25-20261028');raw=gzip.decompress((root/'report.json.gz').read_bytes());r=json.loads(raw);assert r['complete']
run=Path('logs/manipulation/mars-failure-replay-v25/tracking_replay/seed-87')
portable={'scope':__doc__,'complete':True,'source_hashes':r['source_hashes'],'checkpoint':r['checkpoint'],'cases':r['cases'],'public_program_summaries':json.load(open(root/'summary.json'))['programs'],'training_run':json.load(open(run/'run.json')),'completed_training':json.load(open(run/'complete.json')),'final_manifest':json.load(open(run/'last/manifest.json')),'scalar_metrics':[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()],'safety_promoted':False,'hardware_used':False}
b=(json.dumps(portable,allow_nan=False)+'\n').encode();(out/'results.json.gz').write_bytes(gzip.compress(b,mtime=0));shutil.copy2(root/'summary.json',out/'summary.json')
h=hashlib.sha256()
with (root/'report.json.gz').open('rb') as f:
 for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
(out/'provenance.json').write_text(json.dumps({'scope':__doc__,'source_revision':'49a11690','portable_uncompressed_sha256':hashlib.sha256(b).hexdigest(),'local_full_report':str(root/'report.json.gz'),'local_full_report_compressed_sha256':h.hexdigest(),'local_full_report_uncompressed_sha256':hashlib.sha256(raw).hexdigest(),'public_observation_traces_local_only':True,'safety_promoted':False},indent=2)+'\n')
diagnostics=Path('paper/data/whole_body_control/tracking_and_teacher_diagnosis_v26_v29');diagnostics.mkdir(exist_ok=True)
for name,folder in [('base_v26','base-isolation-v26-20261028'),('retiming_bounds_v27','teacher-retention-v27-20261028'),('grip_and_retime_v28','teacher-grip-v28-20261028'),('uniform_retime_v29','teacher-uniform-retime-v29-20261028')]:
 src=Path('logs/eval')/folder;summary=json.load(open(src/'summary.json'));assert summary['complete']
 for file in ['summary.json','report.json.gz']:shutil.copy2(src/file,diagnostics/(name+'-'+file))
for name in ['isolate_base_v26.py','audit_teacher_retention_v27.py','audit_teacher_grip_v28.py','audit_teacher_uniform_v29.py']:
 shutil.copy2(Path('logs/recovery/20261028')/name,diagnostics/name)
(diagnostics/'provenance.json').write_text(json.dumps({'scope':'All repeated diagnostic populations remain development, not new qualification; uniform retiming changes reference timing, not original baseline evidence.','hardware_used':False,'safety_promoted':False,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in diagnostics.iterdir() if p.is_file() and p.name!='provenance.json'}},indent=2)+'\n')
print({'v25_portable_bytes':(out/'results.json.gz').stat().st_size,'diagnostic_bytes':sum(p.stat().st_size for p in diagnostics.iterdir())})
