"""CPU integration: native raw codec -> independent Python reader."""
from pathlib import Path
import subprocess
import sys
import hashlib
import struct
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from binary_xyz.raw import iter_raw, read_raw

@pytest.fixture(scope='session')
def fixtures(tmp_path_factory):
    work=tmp_path_factory.mktemp('bxyz_cpu')
    # Tests contain synthetic data only; no private trajectory is copied here.
    subprocess.run(['/usr/bin/g++','-std=c++17','-O2','-DUSE_BXYZ','-I'+str(ROOT/'src'),
                    '-I'+str(ROOT/'src/third_party/binary_xyz/cpp/include'),
                    str(ROOT/'tests/output_format/test_raw.cpp'),str(ROOT/'src/io/output_format.cpp'),
                    str(ROOT/'src/third_party/binary_xyz/cpp/src/binary_xyz.cpp'),'-lzstd','-lcrypto','-o',str(work/'test')],check=True)
    subprocess.run([str(work/'test'),str(work/'frames')],check=True)
    return work/'frames'

@pytest.mark.parametrize('ext',['bxyz','zbxyz'])
def test_arrays(fixtures,ext):
    frames=list(iter_raw(fixtures/('raw.'+ext)))
    assert len(frames)==2
    f=frames[0];assert f.count==2 and set(f.arrays)=={'vel','virial','label'}
    np.testing.assert_array_equal(f.arrays['vel'],[[.1,-0.,2],[3,4,5]])
    assert np.signbit(f.arrays['vel'][0,1])
    np.testing.assert_array_equal(f.arrays['virial'],np.full((2,9),1.234567890123))
    assert f.info['virial']=='1 2 3 4 5 6 7 8 9'
    assert f.arrays['label'].tolist()==['CG_A','粒子']

@pytest.mark.parametrize('ext',['bxyz','zbxyz'])
def test_empty_and_labels(fixtures,ext):
    f=read_raw(fixtures/('empty.'+ext));assert f.count==0 and f.arrays['vel'].shape==(0,3)
    f=read_raw(fixtures/('cg.'+ext));assert f.arrays['species'].tolist()==['CG_A','CG_B']

@pytest.mark.parametrize('ext',['bxyz','zbxyz'])
def test_corruption(fixtures,ext):
    with pytest.raises(ValueError,match='SHA-256'): list(iter_raw(fixtures/('broken.'+ext)))

@pytest.mark.parametrize('cut',[1,2,20,70])
def test_truncation(fixtures,tmp_path,cut):
    data=(fixtures/'raw.bxyz').read_bytes();p=tmp_path/'truncated.bxyz';p.write_bytes(data[:-cut])
    with pytest.raises(ValueError,match='truncated'):list(iter_raw(p))

@pytest.mark.parametrize('version',[1,2,3,4,5])
def test_legacy_structure(tmp_path,version):
    # Independently encode one H atom; retain historical layouts and checksums.
    data=struct.pack('<I',1)+b'H'+struct.pack('<3d',.1,.2,.3)
    if version==2:data+=b'\n'
    header='Properties=species:S:1:pos:R:3 BXYZVersion='+str(version)
    if version>=2:header+=' BXYZStorage=S:R64'
    if version==5:
        import zstandard
        raw_size=len(data);data=zstandard.ZstdCompressor().compress(data)
        header+=f' BXYZCodec=zstd BXYZFilter=none BXYZRawBytes={raw_size}'
    header+=f' BXYZPayloadBytes={len(data)}'
    if version>=4:header+=' BXYZChecksum=sha256:'+hashlib.sha256(('1\n'+header+'\n').encode()+data).hexdigest()
    path=tmp_path/'legacy.bxyz';path.write_bytes(('1\n'+header+'\n').encode()+data)
    frame=read_raw(path);assert frame.arrays['species'].tolist()==['H']
    np.testing.assert_array_equal(frame.arrays['pos'],[[.1,.2,.3]])
