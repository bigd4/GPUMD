"""GPU regression: auto/manual INA counts, endpoint metrics, and parser errors.

Run on a compute node: python test_dist_ncount.py --binary /path/gpumd --output NEW_DIR
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).resolve().parents[2] / 'examples/gpucfneb/diamond_vacancy'
    start = read(fixture / 'is.xyz')
    finish = start.copy()
    finish.positions[:3, 0] += [.2, .25, .4]
    summaries = {}
    paths = {}

    def run(name, options='', count=3, active=3, variable=False, affine=False, mic=False,
            expect_insertion=False):
        directory = out / name
        directory.mkdir()
        end = finish.copy()
        if affine:
            end = start.copy()
            end.set_cell(start.cell.array * 1.02, scale_atoms=True)
        if mic:
            end.positions[4] += end.cell.array[0]
        write(directory / 'is.xyz', start, format='extxyz')
        write(directory / 'fs.xyz', end, format='extxyz')
        shutil.copyfile(directory / 'is.xyz', directory / 'model.xyz')
        (directory / 'nep.txt').symlink_to((fixture / 'nep.txt').resolve())
        text = ('potential nep.txt\n'
                'neb_set is_name is.xyz fs_name fs.xyz interpolate 1\n'
                'neb_set p 0 cell_factor 10 no_remove_translation\n'
                'neb_set tangent modified dist_range 0.001 100 ina_interval 1\n'
                'neb_set print_interval 1 diagnostic_interval 1\n')
        if not variable:
            text += 'neb_set no_vc\n'
        if mic:
            text += 'neb_set find_mic\n'
        if options:
            text += 'neb_set ' + options + '\n'
        text += 'neb_run fire 1e-10 3 dt_0 0.01 dt_max 0.1 max_move 0.01\n'
        (directory / 'run.in').write_text(text)
        with (directory / 'log').open('w') as log:
            result = subprocess.run([str(binary)], cwd=directory, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=120)
        log = (directory / 'log').read_text()
        if count is None:
            assert result.returncode != 0, name
            assert 'dist_ncount should be auto or a positive integer.' in log, name
            summaries[name] = {'invalid_input_rejected': True}
            return
        assert result.returncode == 0, (name, log[-2000:])
        found = re.search(r'^\s*dist_ncount\s*=\s*(\d+)', log, re.M)
        assert found and int(found[1]) == count, (name, found)
        mode = re.search(r'^\s*dist_ncount_auto\s*=\s*(\w+)', log, re.M)
        assert mode and mode[1] == ('true' if active is not None else 'false'), name
        if active is not None:
            found = re.search(r'^\s*dist_ncount_active_atoms\s*=\s*(\d+)', log, re.M)
            assert found and int(found[1]) == active, name
        frames = read(directory / 'final_traj.xyz', ':')
        assert (len(frames) > 3 if expect_insertion else len(frames) == 3), name
        assert all(np.isfinite(f.positions).all() for f in frames), name
        paths[name] = np.concatenate([np.r_[f.positions.ravel(), f.cell.array.ravel()] for f in frames])
        summaries[name] = {'count': count, 'active_atoms': active, 'auto': active is not None,
                           'final_images': len(frames)}

    run('fixed_default')
    run('fixed_auto', 'dist_ncount auto')
    run('fixed_manual', 'dist_ncount 3', active=None)
    run('fixed_override_auto', 'dist_ncount 2 dist_ncount auto')
    run('fixed_override_manual', 'dist_ncount auto dist_ncount 2', count=2, active=None)
    run('fixed_clamp', 'dist_ncount 99999', count=len(start), active=None)
    run('fixed_threshold', 'active_atom_threshold 0.3', count=1, active=1)
    run('fixed_zero_active', 'active_atom_threshold 1', count=1, active=0)
    run('fixed_mic', mic=True)
    run('variable_default', variable=True)
    run('variable_auto', 'dist_ncount auto', variable=True)
    run('variable_manual', 'dist_ncount 3', active=None, variable=True)
    run('variable_affine', count=1, active=0, variable=True, affine=True)
    insertion = 'dist_range 0.001 0.1 ina_force_tol 1 1e100 ina_force_tol_end '
    run('ina_auto', insertion + 'dist_ncount auto', expect_insertion=True)
    run('ina_manual', insertion + 'dist_ncount 3', active=None, expect_insertion=True)
    for name, value in [('invalid_zero', '0'), ('invalid_negative', '-1'),
                        ('invalid_text', 'automatic'), ('invalid_fraction', '1.5')]:
        run(name, 'dist_ncount ' + value, count=None)
    comparisons = {}
    for a, b in [('fixed_default', 'fixed_auto'), ('fixed_auto', 'fixed_manual'),
                 ('fixed_auto', 'fixed_override_auto'),
                 ('variable_default', 'variable_auto'), ('variable_auto', 'variable_manual'),
                 ('ina_auto', 'ina_manual')]:
        error = float(abs(paths[a] - paths[b]).max())
        assert error < 1e-10, (a, b, error)
        comparisons[a + '__' + b] = error
    (out / 'validation.json').write_text(json.dumps(dict(cases=summaries,
        max_coordinate_cell_differences=comparisons, passed=True), indent=2) + '\n')
    print(f'Passed {len(summaries)} GPU cases and {len(comparisons)} path comparisons.')


if __name__ == '__main__':
    main()
