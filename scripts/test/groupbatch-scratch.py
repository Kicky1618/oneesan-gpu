#!/usr/bin/env python3
"""Regression for the former two-copy admission check in the one-copy solver."""
import argparse
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path, help='groupbatch solver specialized for n=20')
    parser.add_argument('--logs', type=Path, required=True)
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GRIDFP_')}
    expected = {4294967291: 2308006916, 4294966997: 3704549185}
    for scratch, plan in ((18, True), (17, False), (18, False)):
        current = dict(env, GRIDFP_PLAN_ONLY=str(int(plan)))
        run = subprocess.run([str(args.binary.resolve()), '20', str(scratch), '14', '1',
                              *map(str, expected)], env=current, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        log = args.logs/f'scratch-{scratch}-plan-{int(plan)}.log'
        log.write_text(run.stdout)
        if scratch == 17:
            assert run.returncode == 4 and 'forced window does not fit' in run.stdout, str(log)
        elif plan:
            assert run.returncode == 0 and 'scratch_fits=1' in run.stdout, str(log)
        else:
            rows = [dict(re.findall(r'(\w+)=([^\s]+)', line))
                    for line in run.stdout.splitlines() if line.startswith('backend=')]
            assert run.returncode == 0 and len(rows) == 2, str(log)
            assert {int(row['modulus']): int(row['residue']) for row in rows} == expected, str(log)
    print('PASS: n=20 plan and execution accept 18 MiB, reject 17 MiB; two known residues')


if __name__ == '__main__':
    main()
