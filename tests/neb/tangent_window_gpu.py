"""Small GPU integration tests for ODE windows, INA, rollback and continued input.

The loose-target cases exercise acceptance state, not physical convergence.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    fixture = repo / 'examples/gpucfneb/diamond_vacancy'
    sys.path.insert(0, str(repo / 'tools'))
    from binary_xyz.raw import iter_raw
    initial = read(fixture / 'is.xyz')
    final = initial.copy()
    final.positions[:3, 0] += [.2, .25, .4]
    middle = initial.copy()
    middle.positions[:] = (initial.positions + final.positions) / 2
    middle.positions[0, 1] += .02
    report = {}

    def run(name, options='', settings='', steps=12, tol='1e-10', ina=False,
            variable=False, continuation=False, failure=None, restart=None, seed=None):
        directory = output / name
        directory.mkdir()
        write(directory / 'seed.xyz', [initial, middle, final], format='extxyz')
        write(directory / 'model.xyz', initial, format='extxyz')
        (directory / 'nep.txt').symlink_to((fixture / 'nep.txt').resolve())
        text = 'potential nep.txt\noutput_format zbxyz precision lossless\n'
        text += f'neb_set traj_name {seed or "seed.xyz"} interpolate 0\n'
        text += 'neb_set p 0 tangent modified diagnostic_interval 1 no_remove_translation\n'
        if not variable:
            text += 'neb_set no_vc\n'
        if ina:
            text += ('neb_set dist_range 0.001 0.08 ina_interval 1 '
                     'ina_force_tol 1 1e100 ina_force_tol_end\n')
        else:
            text += 'neb_set no_ina\n'
        if settings:
            text += 'neb_set ' + settings + '\n'
        if restart:
            shutil.copyfile(restart, directory / 'restart.txt')
            options += ' window_restart restart.txt'
        command = f'neb_run ode12r {tol} {steps} {options}'
        if continuation:
            text = text.replace('potential nep.txt', 'potential \\ # continuation\n\n# comment\nnep.txt')
            command = command.replace(' ode12r ', ' & # optimizer\n ode12r ').replace(' tangent_window ', ' \\\n tangent_window ')
        text += command + '\n'
        (directory / 'run.in').write_text(text)
        with (directory / 'log').open('w') as log:
            result = subprocess.run([str(binary)], cwd=directory, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=120)
        log = (directory / 'log').read_text()
        if failure:
            assert result.returncode != 0 and failure in log, (name, log[-2000:])
            report[name] = {'rejected_input': True}
            return directory, None
        assert result.returncode == 0, (name, log[-2000:])
        summary = dict(line.split() for line in (directory / 'neb_ode12r_summary.out').read_text().splitlines())
        assert int(summary['evaluations']) <= steps, (name, summary)
        assert float(summary['rollback_force_error']) <= 1e-8 * (1 + float(summary['final_residual']))
        frames = list(iter_raw(directory / 'final_traj.zbxyz'))
        assert len(frames) == int(summary['final_images'])
        assert all(np.isfinite(f.arrays['pos']).all() for f in frames)
        table = np.loadtxt(directory / 'neb_window.out', ndmin=2)
        assert np.all(table[:, 6] == 0)  # No geometry/spacing activation.
        rejected = table[table[:, 2] == 0]
        assert np.array_equal(rejected[:, 3], rejected[:, 4])
        assert len(set(table[-(len(frames)-2):, 15])) == len(frames)-2
        report[name] = summary
        return directory, summary

    default, _ = run('default')
    off, _ = run('off', 'tangent_window off')
    for left, right in zip(iter_raw(default / 'final_traj.zbxyz'), iter_raw(off / 'final_traj.zbxyz')):
        np.testing.assert_array_equal(left.arrays['pos'], right.arrays['pos'])
    on, _ = run('on', 'tangent_window on')
    continued, _ = run('continued', 'tangent_window on', continuation=True)
    for left, right in zip(iter_raw(on / 'final_traj.zbxyz'), iter_raw(continued / 'final_traj.zbxyz')):
        np.testing.assert_array_equal(left.arrays['pos'], right.arrays['pos'])
    wide = 'tangent_window on window_off 100000 window_on 200000 window_hold 2'
    converged, summary = run('hold', wide, tol='1e6', steps=8)
    assert summary['stop_reason'] == 'window_converged' and summary['window_qualified_updates'] == '2'
    _, summary = run('static_budget', wide, tol='1e6', steps=2)
    assert summary['stop_reason'] == 'evaluation_budget' and summary['window_qualified_updates'] == '0'
    _, summary = run('restart', wide, tol='1e6', steps=2,
                     restart=converged / 'window_state.txt', seed=converged / 'final_traj.zbxyz')
    assert summary['window_qualified_updates'] == '0'
    _, summary = run('full_acceptance', 'tangent_window off', tol='1e6', steps=8)
    assert summary['stop_reason'] == 'full_force_converged'
    _, summary = run('ina', 'tangent_window on', ina=True, steps=20)
    assert int(summary['ina_rebuilds']) > 0 and int(summary['final_images']) > 3
    topology = (output / 'ina/neb_window_topology.out').read_text()
    assert ' insert ' in topology
    run('variable', 'tangent_window on', variable=True, steps=6)
    run('energy_spacing', 'tangent_window on', settings='energy_based_spacing energy_spacing_damping 1')
    _, summary = run('rollback', 'tangent_window on h0 100 max_move 10 rtol 1e-10', steps=20)
    assert int(summary['rejected_trials']) > 0
    for name, options, settings, error in [
        ('thresholds', 'tangent_window on window_off .04 window_on .03', '', 'window_off < window_on'),
        ('missing_value', 'tangent_window', '', 'requires a value'),
        ('damped_spacing', '', 'energy_based_spacing', 'energy_spacing_damping 1'),
        ('cyclic', '', 'cyclic_transverse on', 'cannot be combined'),
        ('energy_stop', '', 'energy_convergence 2 100', 'cannot be combined'),
        ('excluded_spring', '', 'spring_parallel length', 'no keyword match'),
    ]:
        run(name, options, settings=settings, failure=error)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(report)} window integration cases passed (loose targets test control flow only).')


if __name__ == '__main__':
    main()
