"""NEB format integration checks; run on a GPU compute node in a fresh directory."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from binary_xyz.raw import iter_raw
from gpu_smoke import text_frames


def encode_frame(arrays, info, compressed=False):
    """Independent v4 fixture, with exact R64 positions and variable strings."""
    n = len(arrays['species'])
    payload = b''
    for symbol, pos in zip(arrays['species'], arrays['pos']):
        label = symbol.encode()
        payload += struct.pack('<I', len(label)) + label + struct.pack('<3d', *pos)
    metadata = ' '.join(f'{k}="{v}"' for k, v in info.items())
    codec = ''
    if compressed:
        import zstandard
        size = len(payload)
        payload = zstandard.ZstdCompressor().compress(payload)
        codec = f' BXYZCodec=zstd BXYZFilter=none BXYZRawBytes={size}'
    header = f'{metadata} Properties=species:S:1:pos:R:3 BXYZVersion={5 if compressed else 4} BXYZStorage=S:R64{codec} BXYZPayloadBytes={len(payload)}'
    digest = hashlib.sha256(f'{n}\n{header}\n'.encode() + payload).hexdigest()
    return f'{n}\n{header} BXYZChecksum=sha256:{digest}\n'.encode() + payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('executable', type=Path)
    p.add_argument('workdir', type=Path)
    args = p.parse_args()
    exe, work = args.executable.resolve(), args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    example = ROOT / 'examples/gpucfneb/diamond_vacancy'
    fixtures = work / 'inputs'; fixtures.mkdir()
    for name in ('is', 'fs'):
        arrays, info = text_frames(example / f'{name}.xyz')[0]
        (fixtures / f'{name}.bxyz').write_bytes(encode_frame(arrays, info))
        (fixtures / f'{name}.zbxyz').write_bytes(encode_frame(arrays, info, True))
    initial, meta = text_frames(example / 'is.xyz')[0]
    final, _ = text_frames(example / 'fs.xyz')[0]
    initial['pos'] = (initial['pos'] + final['pos']) / 2
    (fixtures / 'mid.bxyz').write_bytes(encode_frame(initial, meta))
    report = []
    def run(name, fmt='xyz', options='', endpoints='text', fail=False, existing=None, precision='lossless'):
        d = existing or work / name
        d.mkdir(exist_ok=True)
        suffix = 'xyz' if endpoints == 'text' else 'zbxyz' if endpoints == 'zbxyz' else 'bxyz'
        source = example if endpoints == 'text' else fixtures
        commands = f'potential {example / "nep.txt"}\n'
        commands += f'neb_set is_name {source / ("is." + suffix)} fs_name {source / ("fs." + suffix)}\n'
        commands += 'neb_set interpolate 3 no_ina dump_interval 1 peek_interval 1\n'
        commands += options + '\n'
        commands += f'output_format {fmt}' + ('' if fmt == 'xyz' else f' precision {precision}') + '\n'
        commands += 'neb_run fire 0.00000001 3\n'
        (d / 'run.in').write_text(commands)
        proc = subprocess.run([str(exe)], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
        (d / (name + '.log')).write_text(proc.stdout)
        assert proc.returncode == (1 if fail else 0), (name, proc.returncode, proc.stdout[-4000:])
        if not fail: assert 'NEB step summary:' in proc.stdout
        report.append({'case': name, 'returncode': proc.returncode})
        return d
    def compare(text_dir, binary_dir, fmt, stems):
        for stem in stems:
            text = text_frames(text_dir / (stem + '.xyz'))
            binary = list(iter_raw(binary_dir / (stem + '.' + fmt)))
            assert len(text) == len(binary), (stem, len(text), len(binary))
            for (arrays, meta), frame in zip(text, binary):
                assert set(frame.arrays) == {'species', 'pos'}
                np.testing.assert_array_equal(arrays['species'], frame.arrays['species'])
                np.testing.assert_allclose(arrays['pos'], frame.arrays['pos'], rtol=6e-6, atol=6e-6)
                np.testing.assert_allclose(np.fromstring(meta['Lattice'], sep=' '), np.fromstring(frame.info['Lattice'], sep=' '), rtol=1e-7, atol=1e-7)
                for key in ('energy', 'enthalpy'):
                    np.testing.assert_allclose(float(meta[key]), float(frame.info[key]), rtol=1e-9, atol=1e-6)
                assert 0 <= int(frame.info['image']) < int(frame.info['nimages'])
    stems = ('interpolate', 'dump_traj', 'peek_traj', 'final_traj')
    for mode, options in [('fixed', 'neb_set no_vc match_cell_to_initial'), ('variable', 'neb_set p 30')]:
        baseline = run(mode + '_xyz', options=options)
        for fmt in ('bxyz', 'zbxyz'):
            binary = run(mode + '_' + fmt, fmt, options, endpoints='binary')
            compare(baseline, binary, fmt, stems)
            frames = list(iter_raw(binary / ('final_traj.' + fmt)))
            assert len(frames) == 5
            assert len(list(iter_raw(binary / ('dump_traj.' + fmt)))) == 10
            # Read an entire binary path; no interpolation is requested for this path.
            reload = run(mode + '_reload_' + fmt, fmt, options + f'\nneb_set traj_name {binary / ("final_traj." + fmt)} interpolate 0')
            assert len(list(iter_raw(reload / ('final_traj.' + fmt)))) == 5
    run('compressed_endpoints', 'xyz', 'neb_set no_vc match_cell_to_initial', endpoints='zbxyz')
    auto = run('auto_zbxyz', 'zbxyz', 'neb_set no_vc match_cell_to_initial', endpoints='binary', precision='auto')
    reference = list(iter_raw(work / 'fixed_zbxyz/final_traj.zbxyz'))
    for a, b in zip(reference, iter_raw(auto / 'final_traj.zbxyz')):
        np.testing.assert_allclose(a.arrays['pos'], b.arrays['pos'], rtol=0, atol=1e-6)
    relaxed_text = run('relax_xyz', options='neb_set no_vc match_cell_to_initial need_relax')
    relaxed_binary = run('relax_zbxyz', 'zbxyz', 'neb_set no_vc match_cell_to_initial need_relax')
    compare(relaxed_text, relaxed_binary, 'zbxyz', (*stems, 'relaxed_is_fs'))
    assert len(list(iter_raw(relaxed_binary / 'relaxed_is_fs.zbxyz'))) == 2
    # An explicitly named binary intermediate state uses the same filename reader.
    mid = run('binary_mid', 'bxyz', f'neb_set no_vc match_cell_to_initial mid_name {fixtures / "mid.bxyz"}', endpoints='binary')
    assert len(list(iter_raw(mid / 'final_traj.bxyz'))) >= 3
    baseline = work / 'fixed_bxyz'
    old_dump = (baseline / 'dump_traj.bxyz').read_bytes()
    run('append', 'bxyz', 'neb_set no_vc match_cell_to_initial', endpoints='binary', existing=baseline)
    assert (baseline / 'dump_traj.bxyz').read_bytes().startswith(old_dump)
    assert len(list(iter_raw(baseline / 'dump_traj.bxyz'))) == 20
    assert len(list(iter_raw(baseline / 'peek_traj.bxyz'))) == 5
    # Damage the last frame: first-frame model initialization still succeeds.
    path = work / 'variable_bxyz/final_traj.bxyz'
    data = path.read_bytes()
    for name, broken in [('truncated', data[:-3]), ('checksum', data[:-1] + bytes([data[-1] ^ 1]))]:
        target = fixtures / (name + '.bxyz'); target.write_bytes(broken)
        run(name, 'bxyz', f'neb_set traj_name {target} interpolate 0', fail=True)
    run('one_frame', 'bxyz', f'neb_set traj_name {fixtures / "is.bxyz"} interpolate 0', fail=True)
    empty = fixtures / 'empty.bxyz'; empty.write_bytes(b'')
    run('empty_path', 'bxyz', f'neb_set traj_name {empty} interpolate 0', fail=True)
    # A valid first structure followed by an array-only frame must be rejected.
    payload = struct.pack('<6d', *([0.] * 6))
    header = f'Properties=forces:R:3 BXYZVersion=4 BXYZStorage=R64 BXYZPayloadBytes={len(payload)}'
    digest = hashlib.sha256(f'2\n{header}\n'.encode() + payload).hexdigest()
    array = f'2\n{header} BXYZChecksum=sha256:{digest}\n'.encode() + payload
    target = fixtures / 'array_path.bxyz'
    target.write_bytes((fixtures / 'is.bxyz').read_bytes() + array)
    run('array_path', 'bxyz', f'neb_set traj_name {target} interpolate 0', fail=True)
    arrays, info = text_frames(example / 'is.xyz')[0]
    smaller = {key: value[:-1] for key, value in arrays.items()}
    target = fixtures / 'mismatched.bxyz'
    target.write_bytes((fixtures / 'is.bxyz').read_bytes() + encode_frame(smaller, info))
    run('mismatched_atoms', 'bxyz', f'neb_set traj_name {target} interpolate 0', fail=True)
    corrupt_dump = baseline / 'dump_traj.bxyz'
    corrupt_dump.write_bytes(corrupt_dump.read_bytes()[:-1])
    run('bad_append', 'bxyz', 'neb_set no_vc match_cell_to_initial', endpoints='binary', existing=baseline, fail=True)
    (work / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Passed {len(report)} NEB cases')


if __name__ == '__main__':
    main()
