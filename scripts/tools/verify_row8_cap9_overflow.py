#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT/'work/formal-probes/row8_cap9_overflow_wfa.log'
DEFAULT_CK  = ROOT/'work/formal-probes/raw_wfa_r8_cap9_overflow.ck'
SRC = ROOT/'src/cpp/probes/row8_cap9_overflow_wfa.cpp'

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def checkpoint(p: Path):
    with p.open('rb') as f: b=f.read(40)
    magic,ver,r,col,res,n,hh=struct.unpack('<8sIIIIQQ',b)
    return {'magic':magic.rstrip(b'\0').decode(),'version':ver,'row':r,
            'col':col,'states':n,'payload_hash_fnv64':f'{hh:016x}'}

def verify(log: Path, ck: Path):
    text=log.read_text()
    rows=[]
    for m in re.finditer(r'r=8 col=(\d+) states=(\d+) overflow=(\d+) maxsp=(\d+)(?: fixed=1)?',text):
        rows.append(tuple(map(int,m.groups())))
    if not rows or rows[0][0] != 1:
        raise SystemExit('missing col=1 reachability record')
    cols=[x[0] for x in rows]
    if cols != list(range(1,cols[-1]+1)):
        raise SystemExit(f'non-contiguous columns: {cols[:4]}...{cols[-4:]}')
    checks={int(c):(int(a),int(t)) for c,a,t in re.findall(r'final_check col=(\d+) accept=(\d+) tested=(\d+)',text)}
    if set(checks) != set(cols):
        raise SystemExit('final_check coverage mismatch')
    bad=[c for c,(a,_) in checks.items() if a]
    if bad: raise SystemExit(f'overflow accepted at columns {bad}')
    m=re.search(r'cap9_all_widths_no_overflow_accept=1 fixed_col=(\d+) states=(\d+) overflow_states=(\d+) exact=1',text)
    if not m: raise SystemExit('missing fixed-point exact summary')
    fixed_col,states,overflow=map(int,m.groups())
    if rows[-1][0]!=fixed_col or rows[-1][1]!=states or rows[-1][2]!=overflow:
        raise SystemExit('fixed-point summary disagrees with last row')
    if 'fixed=1' not in next(line for line in text.splitlines() if line.startswith(f'r=8 col={fixed_col} ')):
        raise SystemExit('last reachability row is not marked fixed')
    ci=checkpoint(ck)
    if ci['magic']!='R9OVERF' or ci['version']!=2 or ci['row']!=8:
        raise SystemExit(f'bad checkpoint schema: {ci}')
    if ci['col']!=fixed_col or ci['states']!=states:
        raise SystemExit('checkpoint does not match fixed point')
    # The cap-9 machine must actually explore overflow histories, otherwise the
    # certificate would be vacuous.
    first_overflow=next((c for c,_,ov,_ in rows if ov),None)
    if first_overflow is None:
        raise SystemExit('no overflow histories were explored')
    result={
      'schema':'row8-cap9-overflow-v1', 'row':8, 'cap':9,
      'first_overflow_col':first_overflow, 'fixed_col':fixed_col,
      'states':states, 'overflow_states':overflow,
      'all_final_checks_zero':True,
      'source_sha256':sha256(SRC), 'log_sha256':sha256(log),
      'checkpoint_sha256':sha256(ck), 'checkpoint':ci,
    }
    print(json.dumps(result,sort_keys=True))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--log',type=Path,default=DEFAULT_LOG);ap.add_argument('--checkpoint',type=Path,default=DEFAULT_CK);a=ap.parse_args()
    verify(a.log,a.checkpoint)
if __name__=='__main__':main()
