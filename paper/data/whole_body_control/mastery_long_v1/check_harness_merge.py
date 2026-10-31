"""Run every Python test added/changed by the merged harness update on CPU."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import xml.etree.ElementTree as ET
root=Path.cwd();out=root/'logs/recovery/mastery-long-v1';revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
paths=subprocess.check_output(['git','diff','--name-only','61da55cb..cd169673'],text=True).splitlines()
tests=[p for p in paths if p.startswith('src/test/') and Path(p).name.startswith('test_') and p.endswith('.py')]
def fingerprints():
 return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() if (root/p).is_file() else None for p in paths}
before=fingerprints();start=time.time();xml=out/'harness-contracts.xml'
command=[sys.executable,'-m','pytest','-q','--tb=short',*tests,'--junitxml='+str(xml)]
with (out/'harness-contracts.log').open('x') as log:
 result=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=str(root/'src')),stdout=log,stderr=subprocess.STDOUT,check=False)
status={'source_revision':revision,'merged_main':'cd169673','test_paths':tests,'source_hashes':before,'command':command,'returncode':result.returncode,'started_unix':start,'finished_unix':time.time(),'complete':False,'all_pass':False}
if xml.is_file():
 cases=list(ET.parse(xml).getroot().iter('testcase'));status['counts']={key:sum(c.find(key) is not None for c in cases) for key in ['failure','error','skipped']};status['counts']['tests']=len(cases)
 status['all_pass']=result.returncode==0 and bool(cases) and not any(status['counts'][k] for k in ['failure','error','skipped']) and before==fingerprints()
status['complete']=True
(out/'harness-contracts.json').write_text(json.dumps(status,indent=2)+'\n');print(status.get('counts'),status['all_pass'],flush=True)
if not status['all_pass']:raise RuntimeError('merged harness contract gate failed; inspect log')
