"""Archive training-population and current teacher evidence; never policy weights."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

from emet.rl.manipulation.carry_benchmark import digest
from emet.rl.manipulation.mastery import load_suite, read_report, select_cases, validate_preflight

root = Path.cwd()
recovery = root / 'logs/recovery/mastery-crossed-v1'
teacher_dir = root / 'logs/eval/mars-mastery-crossed-v1-teacher'
suite = load_suite(root / 'configs/benchmarks/mars_tracking_mastery_crossed_v1.json')
teacher = read_report(teacher_dir / 'report.json.gz')
validate_preflight(teacher, suite, ['base'])
out = root / 'paper/data/whole_body_control/crossed_posture_v1_setup'
out.mkdir(exist_ok=True, parents=True)
for name in ['report.json.gz', 'summary.json', 'targets.svg']:
    shutil.copy2(teacher_dir / name, out / ('teacher-' + name))
for name in ['regressions.xml', 'regressions.receipt.json', 'launch_when_ready.py']:
    if (recovery / name).exists():
        shutil.copy2(recovery / name, out / name)
manifest = {
    'scope': 'Finite base training-population expansion; unchanged canonical evaluation and gates',
    'suite_sha256': digest(suite), 'suite': suite,
    'training_cases': select_cases(suite, ['base'], 'train', population='training'),
    'canonical_evaluation_cases': select_cases(suite, ['base'], 'all'),
    'teacher_admission_cases': len(teacher['cases']),
    'teacher_passed': sum(c['mastered'] for c in teacher['cases']),
    'source_hashes': teacher['source_hashes'],
    'initialization_weights_sha256': '8a81fb241e4cfb59e310bc50762d8bedaec96b66ac5f6259e494d7586bab966d',
    'planned_steps': 65536, 'seed': 90, 'learning_rate': 0.00005,
    'base_std': 0.005, 'arm_std': 0.01, 'command_smoothness_warmup_steps': 0,
    'canonical_checkpoint_every': 16384, 'hardware_used': False, 'safety_promoted': False,
}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
shutil.copy2(__file__, out / 'archive_setup.py')
checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.name != 'files-sha256.json'}
(out / 'files-sha256.json').write_text(json.dumps(checksums, indent=2) + '\n')
print(out, flush=True)
