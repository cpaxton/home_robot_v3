"""Copy only audited text sources into local ignored storage; remote is read-only."""
import hashlib,json,shlex,subprocess
from pathlib import Path
paths=[n for n in json.loads(Path('docs/patches/innate_os_feedback_units.meta.json').read_text())['before_sha256'] if not n.endswith('/arm.cpp')]
remote='from pathlib import Path; import json; root=Path.home()/"innate-os"; names='+repr(paths)+'; print(json.dumps({n:(root/n).read_text() for n in names}))'
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=yes','jetson1@192.168.1.43','python3 -c '+shlex.quote(remote)],capture_output=True,text=True,timeout=12)
if r.returncode:raise RuntimeError(r.stderr[:1500])
files=json.loads(r.stdout);out=Path('logs/recovery/herman-passive-20261117/source');manifest={}
for name,data in files.items():
 p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data);manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
Path('logs/recovery/herman-passive-20261117/source-hashes.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Copied',len(files),'audited source files locally; no robot writes')
