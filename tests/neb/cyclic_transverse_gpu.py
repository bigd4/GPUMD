#!/usr/bin/env python3
"""GPU smoke tests. Pass a new shared-filesystem directory via --work-dir."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', required=True, type=Path)
parser.add_argument('--work-dir', required=True, type=Path)
args = parser.parse_args()
binary = args.binary.resolve()
root = args.work_dir.resolve()
root.mkdir(parents=True, exist_ok=False)
fixture = Path(__file__).resolve().parents[2] / 'examples/gpucfneb/diamond_vacancy'
base = '''potential nep.txt
neb_set is_name is.xyz fs_name fs.xyz interpolate 5
neb_set p 30 tangent modified no_ina diagnostic_interval 1
'''
fire = 'dt_0 0.1 dt_max 1 dt_min 0.005 max_move 0.1 imagewise alpha_min 0.01'


def run(name, options, tolerance, steps, success=True):
    directory = root / name
    directory.mkdir()
    for filename in ['is.xyz', 'fs.xyz', 'nep.txt']:
        (directory / filename).symlink_to(fixture / filename)
    (directory / 'model.xyz').symlink_to('is.xyz')
    (directory / 'run.in').write_text(base + options + f'neb_run fire {tolerance} {steps} {fire}\n')
    with (directory / 'log').open('w') as log:
        result = subprocess.run([str(binary)], cwd=directory, stdout=log, stderr=subprocess.STDOUT)
    assert (result.returncode == 0) == success, (name, result.returncode)
    return directory


def summary(directory):
    return dict(line.split() for line in (directory / 'neb_cyclic_transverse_summary.out').read_text().splitlines())


for name, options in [('default', ''), ('off', 'neb_set cyclic_transverse off\n')]:
    d = run(name, options, '1e-12', 6)
    assert not (d / 'neb_cyclic_transverse_summary.out').exists()
# A huge target deliberately isolates the controller from physical convergence.
# Even with zero/low spring force, a one-evaluation parallel budget must fail.
d = run('parallel_budget', 'neb_set cyclic_transverse on\n', '1e6', 1)
s = summary(d)
assert s['accepted'] == '0' and s['regular_evaluations'] == '1' and s['cycles'] == '0'
d = run('cycles', 'neb_set cyclic_transverse 1 cyclic_parallel_steps 2 cyclic_perpendicular_steps 3 cyclic_settling_steps 1 cyclic_required_cycles 2\n', '1e6', 20)
s = summary(d)
assert s['accepted'] == '1' and s['cycles'] == '2' and s['regular_evaluations'] == '6'
for name, options in [('bad_switch', 'cyclic_transverse yes'), ('bad_count', 'cyclic_parallel_steps 0'), ('dyneb', 'cyclic_transverse on dyneb')]:
    run(name, 'neb_set ' + options + '\n', '.001', 6, success=False)
print('Cyclic transverse GPU smoke tests passed; synthetic high-tolerance test is not a physical convergence result.')
