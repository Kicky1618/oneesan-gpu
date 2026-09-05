#!/usr/bin/env python3
"""Check n=9 batched solver residues, including grid-stride and partial batches."""
import argparse
from itertools import product
import os
from pathlib import Path
import re
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary',type=Path)
    args=parser.parse_args()
    expected={4294967291:2674633373,4294966997:3332982389}
    for batch,graphs,threads in product((2,3,8,32),(0,1,2),(32,256)):
        env={k:v for k,v in os.environ.items() if not k.startswith('GRIDFP_')}
        env.update(GRIDFP_TRANSFER_BATCH=str(batch),GRIDFP_TRANSFER_MAP_MIB='1',
                   GRIDFP_TRANSITION_GRAPHS=str(graphs),GRIDFP_VERIFY_TRANSFER_MAPS='1',
                   GRIDFP_FRONTIER_THREADS=str(threads),GRIDFP_FRONTIER_BLOCKS='1' if threads==32 else '0',
                   GRIDFP_PIPELINE_GROUPS=str(batch!=3))
        result=subprocess.run([str(args.binary.resolve()),'9','1','14','1',*map(str,expected)],
                              env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
        rows=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in result.stdout.splitlines() if line.startswith('backend=')]
        actual={int(row['modulus']):int(row['residue']) for row in rows}
        batches=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in result.stdout.splitlines() if line.startswith('transfer_batches=')]
        if len(rows)!=2 or actual!=expected or len(batches)!=2 or any(int(row['transfer_batches'])<=0 for row in batches):
            raise RuntimeError(result.stdout)
        print(f'PASS batch={batch} graphs={graphs} threads={threads}: {actual}',flush=True)


if __name__=='__main__':
    main()
