import argparse,json
from pathlib import Path
from emet.rl.manipulation.control import ManipulationControl
from emet.rl.manipulation.integrated import run_case
p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline','bias','trim','both']);args=p.parse_args()
original=ManipulationControl.rearm
def rearm(self):
    bias=self.base_bias_compensation_enabled
    trim=self.tracking_integral.copy()
    original(self)
    if args.mode in ('bias','both'):
        self.set_base_bias_compensation(bias)
    if args.mode in ('trim','both'):
        self.tracking_integral[:]=trim
ManipulationControl.rearm=rearm
rows=[]
for seed,mass in [(1860001,.02),(1860007,.05),(1860010,.05)]:
 r=run_case(robot='innate_mars',seed=seed,payload_object='object1',payload_mass_kg=mass,action_mode='direct_velocity',contact_cone='elliptic',contact_impratio=3,balanced_grasp_setup=True)
 rows.append(r);print(args.mode,seed,mass,r['success'],r.get('failure'),flush=True)
Path('/tmp/mars-stage-ablation-'+args.mode+'.json').write_text(json.dumps(rows,indent=2))
