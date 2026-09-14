"""Small GPU integration checks. Run on a GPU node, not the login node."""
import argparse
import json
import hashlib
import struct
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from ase.build import bulk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from binary_xyz.raw import iter_raw, _header


def text_frames(path):
    frames=[]
    with open(path) as f:
        while (first:=f.readline()):
            n=int(first); meta=_header(f.readline()); p=meta.pop('Properties').split(':')
            rows=[f.readline().split() for _ in range(n)]
            arrays={}; offset=0
            for i in range(0,len(p),3):
                key,kind,c=p[i],p[i+1],int(p[i+2])
                data=np.array([r[offset:offset+c] for r in rows],dtype=str if kind=='S' else int if kind=='I' else float).reshape(n,c)
                arrays[key]=data[:,0] if c==1 else data; offset+=c
            frames.append((arrays,meta))
    return frames


def main():
    parser=argparse.ArgumentParser();parser.add_argument('executable',type=Path);parser.add_argument('workdir',type=Path)
    args=parser.parse_args();exe=args.executable.resolve();work=args.workdir.resolve();work.mkdir(parents=True,exist_ok=True)
    if any(work.iterdir()): raise RuntimeError('Use a fresh workdir; append tests must not reuse old outputs')
    atoms=bulk('C','diamond',a=3.57,cubic=True).repeat((3,3,3));n=len(atoms)
    cell=atoms.cell.array.copy();cell[1,0]=.25;atoms.set_cell(cell,scale_atoms=True)
    groups=np.column_stack([2*(np.arange(n)%2),np.arange(n)//2])
    velocity=np.sin(np.arange(n*3).reshape(n,3))*.001
    header='Lattice="'+' '.join(f'{v:.17g}' for v in cell.ravel())+'" pbc="T F T" Properties=species:S:1:pos:R:3:mass:R:1:charge:R:1:vel:R:3:group:I:2\n'
    model=str(n)+'\n'+header
    for i in range(n):
        vals=[*atoms.positions[i],12.01+(i%2)*.01, .125*(-1)**i,*velocity[i]]
        model+='C '+' '.join(f'{v:.17g}' for v in vals)+f' {groups[i,0]} {groups[i,1]}\n'
    potential=ROOT/'potentials/nep/C_2022_NEP3.txt'
    prefix=f'potential {potential}\ntime_step 0.1\nensemble nve\n'
    report=[]
    def run(name, commands, model_text=model, model_path=None, fail=False):
        directory=work/name;directory.mkdir(exist_ok=True)
        (directory/'model.xyz').write_text(model_text)
        (directory/'bead_name.txt').write_text('CG_bead\n'*n)
        (directory/'run.in').write_text(commands)
        command=[str(exe)] + (['-m',str(model_path)] if model_path else [])
        start=time.perf_counter()
        result=subprocess.run(command,cwd=directory,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
        (directory/'log').write_text(result.stdout)
        report.append(dict(case=name,seconds=time.perf_counter()-start,returncode=result.returncode))
        if fail:
            assert result.returncode==1,(name,result.stdout[-2000:])
        else:
            assert result.returncode==0,(name,result.stdout[-2000:])
        print(name,result.returncode,flush=True)
        return directory
    dumps='''dump_position 1 group 0 2 precision double
 dump_velocity 1 group 0 2
 dump_force 1 group 0 2
 dump_restart 1
 dump_exyz 1 1 1 1 0
 dump_xyz -1 -1 1 full.xyz velocity force potential unwrapped_position mass charge virial group
 dump_xyz 0 2 1 selected.xyz velocity force group
 dump_xyz 0 1 1 empty.xyz velocity force group
 dump_xyz -1 -1 2 snapshot.* velocity
 dump_observer observe 1 1 1 1
 dump_cg 1 1
 dump_thermo 1
'''
    baseline=run('text',prefix+dumps+'run 4\n')
    for mode in ('bxyz','zbxyz'):
        directory=run(mode,prefix+dumps+f'output_format {mode} precision lossless\nrun 4\n')
        for stem in ('movie','restart','dump','full','selected','empty','observer','train'):
            text=text_frames(baseline/(stem+'.xyz'));binary=list(iter_raw(directory/(stem+'.'+mode)))
            assert len(text)==len(binary),(stem,len(text),len(binary))
            for (expected,meta),frame in zip(text,binary):
                assert set(expected)==set(frame.arrays),(stem,expected.keys(),frame.arrays.keys())
                for key,values in expected.items():
                    actual=frame.arrays[key]
                    if values.dtype.kind in 'USO': np.testing.assert_array_equal(actual,values)
                    else: np.testing.assert_allclose(actual,values,rtol=1e-5 if stem in ('movie','restart') else 1e-7,atol=6e-8,err_msg=f'{stem}.{key}')
                for key,value in meta.items():
                    if key in ('Lattice','virial','stress'):
                        np.testing.assert_allclose(np.fromstring(frame.info[key],sep=' '),np.fromstring(value,sep=' '),rtol=1e-5,atol=6e-8,err_msg=f'{stem} metadata {key}')
        for stem,key in [('velocity','vel'),('force','forces')]:
            binary=list(iter_raw(directory/(stem+'.'+mode))); assert len(binary)==4
            expected=np.loadtxt(baseline/(stem+'.out')).reshape(4,n//2,3)
            for frame,values in zip(binary,expected):
                assert set(frame.arrays)=={key};assert frame.info['group_id']=='2'
                np.testing.assert_allclose(frame.arrays[key],values,rtol=6e-6,atol=1e-10)
        assert (directory/('snapshot.2.'+mode)).exists()
        assert len(list(iter_raw(directory/('restart.'+mode))))==1
        # Full binary model and binary restart are accepted as fresh inputs.
        restarted=run(mode+'_restart',prefix+f'output_format {mode} precision lossless\ndump_xyz -1 -1 1 resumed.xyz mass velocity group\nrun 1\n',model_path=directory/('restart.'+mode))
        resume=next(iter_raw(restarted/('resumed.'+mode)))
        np.testing.assert_array_equal(resume.arrays['group'],groups)
        np.testing.assert_array_equal(resume.arrays['mass'],np.array([12.01+(i%2)*.01 for i in range(n)]))
        run(mode+'_model',prefix+'run 1\n',model_path=directory/('full.'+mode))
        run(mode+'_array_rejected',prefix+'run 1\n',model_path=directory/('velocity.'+mode),fail=True)
        # Check format persistence, reset to defaults and switching back to text.
        switch=run(mode+'_switch',prefix+f'output_format {mode} precision lossless\ndump_velocity 1\nrun 1\ndump_velocity 1\nrun 1\noutput_format xyz\ndump_velocity 1\nrun 1\n')
        assert len(list(iter_raw(switch/('velocity.'+mode))))==2
        assert np.loadtxt(switch/'velocity.out').shape==(n,3)
        empty=run(mode+'_empty',prefix+f'output_format {mode}\ndump_velocity 1 group 0 1\ndump_force 1 group 0 1\ndump_position 1 group 0 1\nrun 1\n')
        assert all(next(iter_raw(p)).count==0 for p in empty.glob('*.'+mode))
        beads=run(mode+'_beads',f'potential {potential}\ntime_step 0.1\nensemble pimd 4 300 300 100\noutput_format {mode}\ndump_beads 1 1 1\nrun 2\n')
        assert len(list(beads.glob('beads_dump_*.'+mode)))==4
        for path in beads.glob('beads_dump_*.'+mode):
            f=list(iter_raw(path));assert len(f)==2 and set(f[0].arrays)=={'species','pos','vel','forces'}
        auto=run(mode+'_auto',prefix+f'output_format {mode}\ndump_exyz 1 1 1 1 1\nrun 2\n')
        assert len(list(auto.glob('dump.*.'+mode)))==2
        # Appending refuses an incomplete previous trajectory.
        corrupt=work/(mode+'_corrupt');corrupt.mkdir(exist_ok=True)
        data=(directory/('velocity.'+mode)).read_bytes();(corrupt/('velocity.'+mode)).write_bytes(data[:-1])
        run(mode+'_corrupt',prefix+f'output_format {mode}\ndump_velocity 1\nrun 1\n',fail=True)
    # Multiple observer files use each potential's own result and keep numbering.
    observer=run('observer_multi', f'potential {potential}\npotential {potential}\ntime_step 0.1\nensemble nve\noutput_format zbxyz\ndump_observer observe 1 1 1 1\nrun 2\n')
    assert len(list(observer.glob('observer*.zbxyz')))==2
    for path in observer.glob('observer*.zbxyz'): assert len(list(iter_raw(path)))==2
    # QNEP supplies real Born effective charge tensors; preserve all 9 components.
    training=ROOT/'examples/qnep_train/train.xyz'
    with training.open() as file:
        size=int(file.readline());header=file.readline();rows=[file.readline().split()[:4] for _ in range(size)]
    meta=_header(header)
    qmodel=str(size)+'\nLattice="'+meta['Lattice']+'" pbc="T T T" Properties=species:S:1:pos:R:3:vel:R:3\n'
    qmodel+=''.join(' '.join(row)+' 0 0 0\n' for row in rows)
    qprefix=f'potential {ROOT}/examples/qnep_train/nep.txt\ntime_step 0.01\nensemble nve\n'
    qtext=run('bec_text',qprefix+'dump_xyz -1 -1 1 bec.xyz charge bec force\nrun 1\n',model_text=qmodel)
    qbinary=run('bec_binary',qprefix+'output_format zbxyz precision lossless\ndump_xyz -1 -1 1 bec.xyz charge bec force\nrun 1\n',model_text=qmodel)
    expected=text_frames(qtext/'bec.xyz')[0][0];actual=next(iter_raw(qbinary/'bec.zbxyz')).arrays
    assert actual['bec'].shape==(size,9)
    for key in ('bec','charge','forces'): np.testing.assert_allclose(actual[key],expected[key],rtol=1e-7,atol=6e-8)
    # Independently encoded model fixtures exercise validation after checksum verification.
    def fixture(name, arrays, lattice=cell):
        path=work/(name+'.bxyz'); count=len(next(iter(arrays.values())))
        specs=[]
        for key,data in arrays.items():
            data=np.asarray(data);c=1 if data.ndim==1 else data.shape[1]
            kind='S' if data.dtype.kind in 'USO' else 'L' if data.dtype.kind=='b' else 'I' if data.dtype.kind in 'iu' else 'R'
            specs.append((key,kind,c,data.reshape(count,c)))
        payload=bytearray()
        for i in range(count):
            for _,kind,c,data in specs:
                for value in data[i]:
                    if kind=='S':
                        b=str(value).encode();payload.extend(struct.pack('<I',len(b))+b)
                    else:payload.extend(struct.pack({'R':'<d','I':'<q','L':'<B'}[kind],float(value) if kind=='R' else int(value)))
        header='Properties='+':'.join(f'{k}:{t}:{c}' for k,t,c,_ in specs)
        header+=' BXYZStorage='+':'.join({'R':'R64','I':'I64','S':'S','L':'L8'}[t] for _,t,_,_ in specs)
        if lattice is not None:header+=' Lattice="'+' '.join(str(x) for x in np.asarray(lattice).ravel())+'"'
        header+=f' BXYZVersion=4 BXYZPayloadBytes={len(payload)}'
        first=str(count)+'\n';digest=hashlib.sha256((first+header+'\n').encode()+payload).hexdigest()
        path.write_bytes((first+header+' BXYZChecksum=sha256:'+digest+'\n').encode()+payload)
        return path
    simple={'species':np.array(['C']*n),'pos':atoms.positions.copy()}
    no_optionals=fixture('defaults',simple)
    defaults=run('model_defaults',prefix+'velocity 10\noutput_format zbxyz precision lossless\ndump_xyz -1 -1 1 defaults.xyz mass charge\nrun 1\n',model_path=no_optionals)
    data=next(iter_raw(defaults/'defaults.zbxyz')).arrays
    np.testing.assert_array_equal(data['mass'],np.full(n,12.011))
    np.testing.assert_array_equal(data['charge'],np.zeros(n))
    run('model_z_only',prefix+'velocity 10\nrun 1\n',model_path=fixture('z_only',{'Z':np.full(n,6,dtype=np.int64),'pos':atoms.positions}))
    bad_models=[
      ('missing_pos',{'species':simple['species']},cell),
      ('missing_species',{'pos':simple['pos']},cell),
      ('missing_cell',simple,None),('singular_cell',simple,np.zeros((3,3))),
      ('bad_mass',dict(simple,mass=np.full(n,-1.)),cell),
      ('bad_position',dict(simple,pos=np.full((n,3),np.nan)),cell),
      ('bad_group',dict(simple,group=np.full(n,2**40,dtype=np.int64)),cell),
      ('logical_group',dict(simple,group=np.zeros(n,dtype=bool)),cell),
      ('bad_velocity',dict(simple,vel=np.zeros((n,2))),cell),
      ('one_atom',{k:v[:1] for k,v in simple.items()},cell)]
    for name,arrays,lattice in bad_models:
        run('model_'+name,prefix+'run 1\n',model_path=fixture(name,arrays,lattice),fail=True)
    # Explicit non-XYZ suffixes are retained before the binary extension.
    custom=run('custom_suffix',prefix+'output_format zbxyz\ndump_xyz -1 -1 1 custom.out\nrun 1\n')
    assert (custom/'custom.out.zbxyz').exists()
    for i,opt in enumerate(['output_format unknown','output_format xyz precision auto','output_format bxyz compression_level 3','output_format zbxyz precision','output_format zbxyz compression_level 23']):
        run('bad_'+str(i),prefix+opt+'\nrun 1\n',fail=True)
    # Same run length/fields: size and wall time are descriptive, not an MD speed benchmark.
    for row in report:
        directory=work/row['case'];row['output_bytes']=sum(p.stat().st_size for p in directory.iterdir() if p.name not in ('model.xyz','run.in','log','bead_name.txt'))
    (work/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ALL GPU OUTPUT CHECKS PASSED',flush=True)

if __name__=='__main__':main()
