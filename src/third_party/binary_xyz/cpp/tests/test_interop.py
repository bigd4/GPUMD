"""Run through CTest with BXYZ_CPP_CLI and BXYZ_CPP_EXAMPLE set."""
import hashlib
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from binary_xyz import iread, read, verify, write


CLI = os.environ.get("BXYZ_CPP_CLI")
EXAMPLE = os.environ.get("BXYZ_CPP_EXAMPLE")
pytestmark = pytest.mark.skipif(not CLI or not EXAMPLE, reason="Configure and build C++ targets, then run CTest")


def command(*args, success=True):
    result = subprocess.run([CLI, *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if success:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def rich_frame():
    a = Atoms("XHTiOg", positions=np.arange(12).reshape(4, 3) / 7,
              cell=[[5, .1, .2], [.3, 6, .4], [.5, .6, 7]], pbc=[True, False, True])
    a.new_array("index", np.array([-(2**63), 2**63-1, 2**31, -2**31], dtype=np.int64))
    a.new_array("small", np.array([-(2**31), 2**31-1, 0, 42], dtype=np.int64))
    a.new_array("mask", np.array([[True, False]] * 4))
    a.new_array("labels", np.array([["", "钛"], ["a b", "氧气"], ['"quote', "\\end"], ["a\n", "label"]]))
    a.new_array("bits", np.array([0x8000000000000000, 1, 0x7FF0000000000000,
                                  0x7FF8000000000001], dtype=np.uint64).view(np.float64))
    a.info.update(label="1", path="C:\\tmp\\file", nested={"a": [1, 2]},
                  matrix=np.arange(6).reshape(2, 3), note="中文\nsecond line", integer=2**60+1)
    a.calc = SinglePointCalculator(a, energy=-1.2, forces=np.arange(12).reshape(4, 3) / 17,
                                  stress=np.arange(6) / 13)
    return a


def equal(a, b):
    np.testing.assert_array_equal(a.numbers, b.numbers)
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.cell, b.cell)
    np.testing.assert_array_equal(a.pbc, b.pbc)
    assert a.arrays.keys() == b.arrays.keys()
    for key in a.arrays:
        np.testing.assert_array_equal(a.arrays[key], b.arrays[key])
    assert a.info.keys() == b.info.keys()
    for key in a.info:
        if isinstance(a.info[key], dict):
            assert a.info[key] == b.info[key]
        else:
            np.testing.assert_array_equal(a.info[key], b.info[key])
    ar = a.calc.results if a.calc else {}
    br = b.calc.results if b.calc else {}
    assert ar.keys() == br.keys()
    for key in ar:
        np.testing.assert_array_equal(ar[key], br[key])


@pytest.mark.parametrize("source_suffix", [".bxyz", ".zbxyz"])
@pytest.mark.parametrize("output_suffix", [".bxyz", ".zbxyz"])
@pytest.mark.parametrize("mode", ["auto", "variable"])
def test_python_cpp_python(tmp_path, source_suffix, output_suffix, mode):
    source = tmp_path / ("source" + source_suffix)
    output = tmp_path / ("result" + output_suffix)
    empty = Atoms()
    empty.new_array("empty_strings", np.empty((0, 2), dtype=str))
    write(source, [rich_frame(), empty, Atoms("H")], float_rtol=0, string_mode=mode)
    assert "Verified 3 frames" in command("verify", source).stdout
    command("convert", source, output)
    assert verify(output) == 3
    for a, b in zip(iread(source), iread(output)):
        equal(a, b)
        if "bits" in a.arrays:
            np.testing.assert_array_equal(a.arrays["bits"].view(np.uint64), b.arrays["bits"].view(np.uint64))


@pytest.mark.parametrize("suffix", [".bxyz", ".zbxyz"])
def test_md_native_writer(tmp_path, suffix):
    path = tmp_path / ("native" + suffix)
    result = subprocess.run([EXAMPLE, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert verify(path) == 3
    frames = list(iread(path))
    for step, frame in enumerate(frames):
        assert frame.numbers.tolist() == [22, 8]
        np.testing.assert_allclose(frame.positions[0], [.1+.01*step, .2, .3], rtol=1e-7, atol=0)
        assert frame.info["step"] == step
        assert frame.get_potential_energy() == -1.25
        np.testing.assert_allclose(frame.get_forces(), [[.1, 0, 0], [-.1, 0, 0]], rtol=1e-7, atol=0)
    assert "frame=2 atoms=2" in command("inspect", path, "2").stdout


def test_legacy_and_mixed_versions(tmp_path):
    path = tmp_path / "mixed.bxyz"
    v1 = b"1\nProperties=species:S:1:pos:R:3 BXYZVersion=1 BXYZPayloadBytes=29\n" + struct.pack("<I", 1)+b"H"+struct.pack("<3d", 1,2,3)
    v2 = b"1\nProperties=species:S:1:pos:R:3 BXYZVersion=2 BXYZStorage=S:R32 BXYZPayloadBytes=18\n"+struct.pack("<I",1)+b"H"+struct.pack("<3f",1,2,3)+b"\n"
    v3 = b"1\nProperties=Z:I:1:pos:R:3 BXYZVersion=3 BXYZStorage=I32:R32 BXYZPayloadBytes=16\n"+struct.pack("<i3f",8,1,2,3)
    path.write_bytes(v1+v2+v3)
    write(path, Atoms("Ti"), append=True)
    assert "frames=4 atoms=4" in command("inspect", path).stdout
    assert "legacy frame has no checksum" in command("verify", path, success=False).stderr
    output=tmp_path / "upgraded.zbxyz"
    command("convert", path, output)
    assert verify(output) == 4
    assert [a.numbers.tolist() for a in iread(output)] == [[1], [1], [8], [22]]


def test_checksum_and_skip_policy(tmp_path):
    one = tmp_path / "one.zbxyz"
    write(one, Atoms("H"))
    frame = one.read_bytes()
    damaged = frame[:-1] + bytes([frame[-1] ^ 1])
    path = tmp_path / "two.zbxyz"
    path.write_bytes(damaged + frame)
    assert "checksum mismatch" in command("verify", path, success=False).stderr
    assert "frame=1 atoms=1" in command("inspect", path, "1").stdout
    path.write_bytes(frame + damaged)
    assert "frame 2" in command("inspect", path, "1", success=False).stderr
    output=tmp_path / "out.bxyz"
    command("convert", path, output, success=False)
    assert not output.exists()
    assert not list(tmp_path.glob("out.bxyz.tmp.*"))


def test_nul_variable_strings(tmp_path):
    a=Atoms("H")
    a.new_array("label", np.array(["a\0b\0"], dtype=object))
    source,output=tmp_path/"in.zbxyz",tmp_path/"out.zbxyz"
    write(source,a)
    command("convert",source,output)
    assert read(output).arrays["label"][0] == "a\0b\0"


def test_crlf_and_independent_hash(tmp_path):
    payload=b"\x01"+struct.pack("<3f",1,2,3)
    prefix=b"1\r\nProperties=species:S:1:pos:R:3 BXYZVersion=4 BXYZStorage=Z8:R32 BXYZPayloadBytes=13\r\n"
    digest=hashlib.sha256(prefix+payload).hexdigest().encode()
    path=tmp_path/"crlf.bxyz"
    path.write_bytes(prefix[:-2]+b" BXYZChecksum=sha256:"+digest+b"\r\n"+payload)
    assert "Verified 1 frames" in command("verify",path).stdout
