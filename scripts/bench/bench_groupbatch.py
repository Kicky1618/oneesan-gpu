#!/usr/bin/env python3
"""Alternate two frozen groupbatch binaries and retain every checked sample."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--modulus', type=int, default=4294967291)
    parser.add_argument('--expected', type=int)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--scratch-mib', type=int, default=512)
    parser.add_argument('--lanes', type=int, choices=(1, 2, 4, 8, 16, 32), default=16)
    parser.add_argument('--graph', type=int, choices=(0, 1, 2), default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.scratch_mib < 1 or not 3 <= args.n <= 27:
        parser.error('require positive repeats and scratch, 3 <= n <= 27')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GRIDFP_')}
    env.update(GRIDFP_GROUPBATCH_LANES=str(args.lanes), GRIDFP_GROUPBATCH_GRAPH=str(args.graph),
               GRIDFP_GROUPBATCH_WRAP32='0', GRIDFP_GROUPBATCH_MATCHLUT='0',
               GRIDFP_GROUPBATCH_P1_DEST='1', GRIDFP_GROUPBATCH_DYNAMIC='0')
    binaries = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    report = dict(n=args.n, modulus=args.modulus,
                  method='AB/BA alternating; one warmup pair excluded',
                  environment={k: v for k, v in env.items() if k.startswith('GRIDFP_')},
                  binaries={k: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                            for k, p in binaries.items()}, samples={k: [] for k in binaries})
    report['hardware'] = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
        text=True).strip()
    report['nvcc'] = subprocess.check_output(['nvcc', '--version'], text=True).strip()
    expected = args.expected
    for iteration in range(args.repeats+1):
        for name in (('baseline', 'candidate') if iteration % 2 == 0 else ('candidate', 'baseline')):
            command = [str(binaries[name]), str(args.n), str(args.scratch_mib), '14', '1',
                       str(args.modulus)]
            start = time.perf_counter()
            run = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, check=True)
            elapsed = time.perf_counter()-start
            log = args.output.with_name(f'{args.output.stem}-{iteration}-{name}.log')
            log.write_text(run.stdout)
            lines = [line for line in run.stdout.splitlines() if line.startswith('backend=')]
            if len(lines) != 1:
                raise RuntimeError(f'expected one result: {log}')
            result = dict(re.findall(r'(\w+)=([^\s]+)', lines[0]))
            residue = int(result['residue'])
            if expected is None:
                expected = residue
            if residue != expected or int(result['modulus']) != args.modulus:
                raise RuntimeError(f'residue mismatch: {log}')
            result.update(iteration=iteration, process_s=elapsed, command=command, log=str(log))
            if iteration:
                report['samples'][name].append(result)
            report.update(expected=expected, independently_known_expected=args.expected is not None)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(args.n, iteration, name, result['wall_s'], 's', flush=True)
    summary = {}
    for name, samples in report['samples'].items():
        summary[name] = {key: statistics.median(float(s[key]) for s in samples)
                         for key in ('wall_s', 'process_s', 'gather_sum_s', 'transition_sum_s',
                                     'scatter_sum_s')}
        summary[name]['range_s'] = [min(float(s['wall_s']) for s in samples),
                                    max(float(s['wall_s']) for s in samples)]
    summary['speedup'] = summary['baseline']['wall_s']/summary['candidate']['wall_s']
    report['summary'] = summary
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
