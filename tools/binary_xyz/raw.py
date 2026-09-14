"""Independent NumPy reader for GPUMD BXYZ property tables (v1-v5).

Use iter_raw(path) for streaming frames; metadata values retain their string form.
No ASE Atoms object or synthetic species/positions is created for array-only frames.
"""
from dataclasses import dataclass
import hashlib
import re
import struct
from pathlib import Path
import numpy as np

SYMBOLS = ('X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
           'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr '
           'Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra '
           'Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
CONTROLS = {'Properties', 'BXYZVersion', 'BXYZStorage', 'BXYZPayloadBytes', 'BXYZRawBytes',
            'BXYZCodec', 'BXYZFilter', 'BXYZChecksum'}

@dataclass
class RawFrame:
    count: int
    arrays: dict
    info: dict
    version: int


def _header(text):
    result = {}
    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace(): i += 1
        if i == len(text): break
        match = re.match(r'[A-Za-z_][A-Za-z_0-9]*', text[i:])
        if not match: raise ValueError('invalid metadata key')
        key = match[0]; i += len(key)
        while i < len(text) and text[i].isspace(): i += 1
        value = 'T'
        if i < len(text) and text[i] == '=':
            i += 1
            while i < len(text) and text[i].isspace(): i += 1
            end = {'"': '"', "'": "'", '[': ']', '{': '}'}.get(text[i:i+1])
            if end: i += 1
            chars = []
            while i < len(text):
                char = text[i]; i += 1
                if end and char == end: break
                if not end and char.isspace(): break
                if char == '\\':
                    if i == len(text): raise ValueError('unterminated escape')
                    char = text[i]; i += 1
                chars.append(char)
            else:
                if end: raise ValueError('unterminated metadata')
            value = ''.join(chars)
        if key in result: raise ValueError('duplicate metadata key')
        result[key] = value
    return result


def _natural(value):
    if not re.fullmatch(r'[0-9]+', value): raise ValueError('invalid nonnegative integer')
    return int(value)


def _decode(n, header, payload, version):
    parts = header['Properties'].split(':')
    if not parts or len(parts) % 3: raise ValueError('invalid Properties')
    storage = header.get('BXYZStorage', '').split(':') if version >= 2 else [
        {'R': 'R64', 'I': 'I64', 'L': 'L8', 'S': 'S'}[parts[i+1]] for i in range(0, len(parts), 3)]
    if len(storage) != len(parts)//3: raise ValueError('storage count mismatch')
    specs = []
    names = set()
    widths = {'R32': 4, 'R64': 8, 'I32': 4, 'I64': 8, 'L8': 1, 'Z8': 1}
    for i, token in enumerate(storage):
        name, kind, count = parts[3*i:3*i+3]; count = _natural(count)
        if name in names or not count or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', name):
            raise ValueError('invalid/duplicate property')
        names.add(name)
        valid = token in {'R': ('R32','R64'), 'I': ('I32','I64'), 'L': ('L8',), 'S': ('S',)}.get(kind, ())
        if kind == 'S' and version >= 3 and (re.fullmatch(r'S[1-9][0-9]*', token) or (token == 'Z8' and name == 'species' and count == 1)):
            valid = True
        if not valid: raise ValueError('invalid storage type')
        width = widths.get(token, int(token[1:]) if token.startswith('S') and len(token)>1 else 0)
        specs.append((name, kind, count, token, width))
    minimum = sum(c*(w or 4) for _,_,c,_,w in specs) + (version == 2)
    if n*minimum > len(payload): raise ValueError('truncated atom records')
    arrays = {name: np.empty((n,c), dtype=object if kind=='S' else np.float64 if kind=='R' else np.int64 if kind=='I' else np.bool_)
              for name,kind,c,_,_ in specs}
    offset = 0
    def scalar(spec, row, component):
        nonlocal offset
        name, kind, count, token, width = spec
        if kind == 'S' and token != 'Z8':
            size = width
            if not width:
                if offset+4 > len(payload): raise ValueError('truncated string length')
                size = struct.unpack_from('<I', payload, offset)[0]; offset += 4
            if offset+size > len(payload): raise ValueError('truncated string')
            data = payload[offset:offset+size]; offset += size
            if width and b'\0' in data:
                end = data.index(b'\0')
                if any(data[end:]): raise ValueError('invalid string padding')
                data = data[:end]
            value = data.decode('utf-8')
        else:
            if offset+width > len(payload): raise ValueError('truncated scalar')
            value = struct.unpack_from({'R32':'<f','R64':'<d','I32':'<i','I64':'<q','L8':'<B','Z8':'<B'}[token], payload, offset)[0]
            offset += width
            if token == 'Z8':
                if value >= len(SYMBOLS): raise ValueError('invalid atomic number')
                value = SYMBOLS[value]
            if token == 'L8' and value > 1: raise ValueError('invalid logical value')
        arrays[name][row, component] = value
    if version == 1:
        for spec in specs:
            for row in range(n):
                for c in range(spec[2]): scalar(spec,row,c)
    else:
        for row in range(n):
            for spec in specs:
                for c in range(spec[2]): scalar(spec,row,c)
            if version == 2:
                if payload[offset:offset+1] != b'\n': raise ValueError('missing record LF')
                offset += 1
    if offset != len(payload): raise ValueError('extra payload bytes')
    return {k: v[:,0] if v.shape[1] == 1 else v for k,v in arrays.items()}


def iter_raw(path):
    """Yield verified frames; v1-v3 legacy frames have no checksum."""
    with open(path, 'rb') as file:
        size = file.seek(0, 2); file.seek(0)
        index = 0
        while True:
            first = file.readline()
            if not first: return
            index += 1
            try:
                second = file.readline()
                if not first.endswith(b'\n') or not second.endswith(b'\n'): raise ValueError('truncated frame header')
                n = _natural(first.strip().decode('ascii'))
                header = _header(second.decode('utf-8'))
                version = _natural(header['BXYZVersion'])
                if version not in range(1,6): raise ValueError('unsupported version')
                length = _natural(header['BXYZPayloadBytes'])
                if length > size-file.tell(): raise ValueError('truncated payload')
                payload = file.read(length)
                if version >= 4:
                    match = re.search(rb' BXYZChecksum=sha256:([0-9a-f]{64})(\r?\n)$', second)
                    if not match: raise ValueError('missing/malformed checksum')
                    prefix = first + second[:match.start()] + match[2]
                    if hashlib.sha256(prefix+payload).hexdigest().encode() != match[1]: raise ValueError('SHA-256 mismatch')
                elif 'BXYZChecksum' in header: raise ValueError('checksum in legacy frame')
                if version == 5:
                    import zstandard
                    raw_size = _natural(header['BXYZRawBytes'])
                    if header['BXYZCodec'] != 'zstd': raise ValueError('unsupported codec')
                    if zstandard.frame_content_size(payload) != raw_size: raise ValueError('Zstd content size mismatch')
                    payload = zstandard.ZstdDecompressor().decompress(payload, max_output_size=raw_size, allow_extra_data=False)
                    if len(payload) != raw_size: raise ValueError('raw length mismatch')
                    filt = header['BXYZFilter']
                    if filt == 'shuffle':
                        parts = header['Properties'].split(':'); tokens = header['BXYZStorage'].split(':')
                        width = 0
                        for i,t in enumerate(tokens):
                            w = {'Z8':1,'L8':1,'R32':4,'R64':8,'I32':4,'I64':8}.get(t)
                            if w is None:
                                if not re.fullmatch(r'S[1-9][0-9]*',t): raise ValueError('shuffle requires fixed records')
                                w = int(t[1:])
                            width += w*_natural(parts[3*i+2])
                        if raw_size != n*width: raise ValueError('shuffle dimensions mismatch')
                        payload = np.frombuffer(payload,dtype=np.uint8).reshape(width,n).T.copy().tobytes()
                    elif filt != 'none': raise ValueError('unsupported filter')
                elif any(k in header for k in ('BXYZCodec','BXYZRawBytes','BXYZFilter')):
                    raise ValueError('compression controls in legacy frame')
                yield RawFrame(n,_decode(n,header,payload,version),{k:v for k,v in header.items() if k not in CONTROLS},version)
            except (ValueError, KeyError, IndexError, struct.error) as exc:
                raise ValueError(f'{path}: frame {index}: {exc}') from exc


def read_raw(path, index=0):
    if index < 0: raise ValueError('index must be nonnegative')
    for i, frame in enumerate(iter_raw(path)):
        if i == index: return frame
    raise IndexError('frame index out of range')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    for i, frame in enumerate(iter_raw(args.path)):
        print(i, frame.count, ', '.join(frame.arrays))
