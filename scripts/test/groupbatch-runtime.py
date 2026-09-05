#!/usr/bin/env python3
"""End-to-end n=9 groupbatch checks across launch and scheduling policies."""
import argparse
import itertools
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('--logs', type=Path, required=True)
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    expected = {4294967291: 2674633373, 4294966997: 3332982389}
    cases = 0
    for lanes, threads, graph, dynamic in itertools.product((1, 16, 32), (64, 256), (0, 1, 2), (0, 1)):
        env = {k: v for k, v in os.environ.items() if not k.startswith('GRIDFP_')}
        env.update(GRIDFP_GROUPBATCH_LANES=str(lanes), GRIDFP_THREADS=str(threads),
                   GRIDFP_GROUPBATCH_GRAPH=str(graph), GRIDFP_GROUPBATCH_DYNAMIC=str(dynamic),
                   GRIDFP_GROUPBATCH_AFFINITY=str(dynamic), GRIDFP_GROUPBATCH_STICKY=str(dynamic),
                   GRIDFP_GROUPBATCH_RECLAIM=str(dynamic))
        log = args.logs/f'{lanes}-{threads}-{graph}-{dynamic}.log'
        with log.open('w') as handle:
            subprocess.run([str(args.binary.resolve()), '9', '1', '14', '1',
                            *map(str, expected)], env=env, stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        rows = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                for line in log.read_text().splitlines() if line.startswith('backend=')]
        if len(rows) != 2:
            raise RuntimeError(f'expected two residues: {log}')
        for row, (modulus, residue) in zip(rows, expected.items()):
            if int(row['modulus']) != modulus or int(row['residue']) != residue:
                raise RuntimeError(f'wrong residue: {log}')
        cases += 1
    print(f'PASS {cases} runtime policies, {2*cases} known CRT residues')


if __name__ == '__main__':
    main()
