"""Run on a GPU node: python energy_convergence_smoke.py executable fresh_results."""
import json
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parents[3]
exe=Path(sys.argv[1]).resolve(); work=Path(sys.argv[2]).resolve(); work.mkdir(parents=True,exist_ok=False)
example=root/'examples/gpucfneb/diamond_vacancy'
report=[]
def run(name, option='', expected=8, energy=False, force='1e-8', fail=False, ina=False):
    directory=work/name;directory.mkdir()
    text=f'potential {example / "nep.txt"}\nneb_set is_name {example / "is.xyz"} fs_name {example / "fs.xyz"} interpolate 3 p 30\n'
    if not ina: text+='neb_set no_ina\n'
    text+=option+f'\nneb_run fire {force} 8\n'
    (directory/'run.in').write_text(text)
    p=subprocess.run([str(exe)],cwd=directory,capture_output=True,text=True,timeout=180)
    log=p.stdout+p.stderr;(directory/'log').write_text(log)
    assert p.returncode==(1 if fail else 0),(name,p.returncode,log[-3000:])
    if not fail:
        m=re.search(r'NEB step summary: regular=(\d+), INA local=(\d+), total=(\d+)',log);assert m,log
        steps,local,total=map(int,m.groups())
        if expected is not None: assert steps==expected,(name,steps)
        assert total==steps+local
        assert ('NEB converged by peak-energy stability:' in log)==energy,(name,log[-3000:])
        assert (directory/'final_traj.xyz').exists() and (directory/'neb_energies.out').exists()
        if energy:
            found=re.search(r'window=(\d+), range=([^ ]+) eV <= ([^ ]+) eV',log)
            assert found and float(found[2])<=float(found[3])
        if ina:
            assert 'add an image:' in log and local>0,log
        report.append({'case':name,'regular':steps,'local':local,'energy_stop':energy})
    else: report.append({'case':name,'returncode':p.returncode})
run('default')
run('enabled','neb_set energy_convergence 3 1e6',3,True)
run('strict','neb_set energy_convergence 4 0')
run('endpoint_peak','neb_set energy_convergence 3 0',8,True)
run('not_full','neb_set energy_convergence 20 1e6')
run('disabled','neb_set energy_convergence 3 1e6\nneb_set energy_convergence 0 0')
run('force_first','neb_set energy_convergence 20 0',1,False,force='1e9')
run('ina_reset','neb_set energy_convergence 3 1e6\nneb_set dist_range 0.001 0.05 ina_interval 1 ina_local_relax_steps 2\nneb_set ina_force_tol 1 1000000',ina=True)
for i,opt in enumerate(['1 0.01','-1 0.01','3 -1','3 nan','3 inf','3','x 1']):
    run('invalid_'+str(i),'neb_set energy_convergence '+opt,fail=True)
(work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('Passed',len(report),'energy convergence cases')
