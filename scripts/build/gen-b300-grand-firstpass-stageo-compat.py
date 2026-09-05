#!/usr/bin/env python3
from __future__ import annotations
import pathlib,subprocess,sys,tempfile

if len(sys.argv)!=3: raise SystemExit('usage: gen-b300-grand-firstpass-stageo-compat.py INPUT.sh OUTPUT.sh')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
old=pathlib.Path(__file__).with_name('gen-b300-grand-firstpass-stageo.py')
needle="  printf 'B300_GRAND_SELECTED_STAGEN_SEARCH_BLOCK_POLICIES=%q\\n' \"$STAGEN_BLOCK_POLICY_LIST\"\n"
if needle not in s:
    anchor="  printf 'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1\\n'\n"
    n=s.count(anchor)
    if n!=1: raise SystemExit(f'Stage-O compat selected anchor: expected one complete-prime marker, got {n}')
    s=s.replace(anchor,needle+anchor,1)
out.parent.mkdir(parents=True,exist_ok=True)
with tempfile.NamedTemporaryFile('w',prefix='stageo-firstpass-',suffix='.sh',dir=out.parent,delete=False) as f:
    f.write(s); tmp=pathlib.Path(f.name)
try:
    subprocess.run([sys.executable,str(old),str(tmp),str(out)],check=True)
finally:
    tmp.unlink(missing_ok=True)
print(f'normalized Stage-O firstpass input {src}: stagen_search_block_provenance=1')
