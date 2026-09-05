#!/usr/bin/env python3
"""Detect CUDA hardware, tune GROUPBATCH (n=20..27), then run checkpointed exact CRT."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'build/autotune'
SOURCE = ROOT / 'src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_blockfused_groupbatch_batch.cu'
PROVENANCE = ROOT / 'scripts/solve/build_provenance.py'
KNOWN = {20: (4294967291, 2308006916), 21: (4294966997, 2124618149)}
MIB = 1 << 20


@contextmanager
def locked(path, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f'another process is using {path}') from exc
        yield


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def clean_env():
    # An inherited PLAN_ONLY / experimental prefix must never affect the solve.
    removed = {'SRC', 'BIN', 'OUT', 'N', 'ARCH', 'LOW_LUT_K', 'HIGH_LUT_K',
               'GROUPBATCH_CHUNK', 'GROUPBATCH', 'OWNERFUSED', 'BLOCKFUSED',
               'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS'}
    return {k: v for k, v in os.environ.items()
            if not k.startswith(('GRIDFP_', 'ROW6_', 'ROW7_', 'ROW8_')) and k not in removed}


def run(command, env, log=None, timeout=None):
    """Stop the entire child process group on timeout or Ctrl-C."""
    handle = log.open('w') if log else None
    try:
        with subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                              stdout=handle, stderr=subprocess.STDOUT if handle else None,
                              start_new_session=True) as child:
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                raise
        if code:
            raise RuntimeError(f'exit {code}: {command[0]}; log: {log}')
    finally:
        if handle:
            handle.close()
    return log.read_text() if log else ''


def detect(logs):
    BUILD.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'scripts/tools/gpu_inventory.cu'
    nvcc = shutil.which('nvcc')
    if not nvcc:
        raise RuntimeError('nvcc is required (CUDA development toolkit)')
    version = subprocess.check_output([nvcc, '--version'], text=True)
    key = hashlib.sha256((digest(source) + nvcc + version).encode()).hexdigest()[:16]
    binary = BUILD / f'inventory-{key}'
    with locked(BUILD / 'build.lock'):
        if not binary.exists():
            run([nvcc, '-std=c++17', source, '-o', binary, '-lcuda'], clean_env(),
                logs / 'inventory-build.log', 180)
    data = json.loads(run([binary], clean_env(), logs / 'inventory.json', 60))
    data['nvcc'] = version
    physical = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,driver_version,mig.mode.current',
         '--format=csv,noheader'], text=True, timeout=30)
    physical = {row[0].strip(): [x.strip() for x in row[1:]]
                for row in csv.reader(physical.splitlines())}
    if not data['gpus']:
        raise RuntimeError('no visible CUDA GPUs')
    for g in data['gpus']:
        if g['uuid'] not in physical or physical[g['uuid']][1].lower() == 'enabled':
            raise RuntimeError('MIG is not supported; select physical GPUs without MIG')
        g['driver_release'] = physical[g['uuid']][0]
    return data


def groups(hardware, maximum):
    """Best-capacity full-P2P subset for each size and compute capability."""
    devices = hardware['gpus']
    if len(devices) > 16:
        raise ValueError('limit CUDA_VISIBLE_DEVICES to at most 16 GPUs')
    eligible = [i for i, g in enumerate(devices) if g['vmm_gran_bytes'] > 0]
    best = {}
    for size in range(1, min(maximum, len(eligible)) + 1):
        for ids in itertools.combinations(eligible, size):
            arch = {(devices[i]['major'], devices[i]['minor']) for i in ids}
            if len(arch) != 1 or not all(hardware['p2p'][i][j] for i in ids for j in ids):
                continue
            key = (size, next(iter(arch)))
            score = min(devices[i]['free_bytes'] for i in ids)
            if key not in best or score > best[key][0]:
                best[key] = (score, ids)
    return [ids for _, ids in best.values()]


@dataclass(frozen=True)
class Config:
    devices: tuple[int, ...]
    scratch_mib: int = 512
    lanes: int = 16
    threads: int = 256
    graph: int = 1
    dynamic: int = 0

    def __post_init__(self):
        if (not self.devices or len(set(self.devices)) != len(self.devices)
                or any(i < 0 for i in self.devices) or len(self.devices) > 8
                or self.scratch_mib < 1 or self.lanes not in (1, 8, 16, 32)
                or self.threads not in (128, 256, 512) or self.graph not in (0, 1, 2)
                or self.dynamic not in (0, 1)):
            raise ValueError('invalid tuning configuration')


def environment(config, hardware, reserve):
    env = clean_env()
    devices = [hardware['gpus'][i] for i in config.devices]
    env.update(CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in devices),
               GRIDFP_GROUPBATCH_LANES=str(config.lanes), GRIDFP_THREADS=str(config.threads),
               GRIDFP_GROUPBATCH_GRAPH=str(config.graph),
               GRIDFP_GROUPBATCH_WRAP32='0', GRIDFP_GROUPBATCH_MATCHLUT='0',
               GRIDFP_GROUPBATCH_P1_DEST='1', GRIDFP_VRAM_RESERVE_MIB=str(reserve),
               GRIDFP_VMM_GRAN_KIB=str(max(g['vmm_gran_bytes'] for g in devices) // 1024))
    for suffix in ('DYNAMIC', 'AFFINITY', 'STICKY', 'RECLAIM'):
        env['GRIDFP_GROUPBATCH_' + suffix] = str(config.dynamic)
    env['GRIDFP_GROUPBATCH_BATCHES_PER_GPU'] = '6'
    return env


def fields(text, prefix):
    return [dict(re.findall(r'(\w+)=([^\s]+)', line))
            for line in text.splitlines() if line.startswith(prefix)]


def fits(plan, config, hardware, reserve):
    peak = max(float(plan['dp_peak_gib']), float(plan['row6_peak_gib'])) * (1 << 30)
    # PLAN prints rounded decimals; allow a small extra margin beyond the reserve.
    return (plan['scratch_fits'] == '1' and math.isfinite(peak)
            and all(peak + (reserve + 16) * MIB < hardware['gpus'][i]['free_bytes']
                    for i in config.devices))


class Tuner:
    def __init__(self, args, hardware, logs):
        self.args, self.hardware, self.logs = args, hardware, logs
        self.binaries, self.plans, self.results, self.planning = {}, {}, [], []
        self.expected = KNOWN.get(args.bench_n, (4294967291, None))[1]
        self.modulus = KNOWN.get(args.bench_n, (4294967291, None))[0]

    def binary(self, n, config):
        with locked(BUILD / 'build.lock'):
            return self._binary(n, config)

    def _binary(self, n, config):
        gpu = self.hardware['gpus'][config.devices[0]]
        arch = f"sm_{gpu['major']}{gpu['minor']}"
        key = (n, arch)
        if key not in self.binaries:
            binary = BUILD / f'groupbatch-n{n}-{arch}'
            expected = [f'-DTARGET_W={n+1}', f'-DLOW_LUT_K={(n+1)//2}',
                        f'-DHIGH_LUT_K={n-(n+1)//2}', f'-arch={arch}', str(SOURCE)]
            verify = [sys.executable, PROVENANCE, 'verify', str(binary)+'.provenance.json',
                      '--binary', binary, '--root', ROOT, '--verify-sources']
            verify += ['--expect-compile-arg='+x for x in expected]
            try:
                if not binary.exists():
                    raise RuntimeError('not built')
                run(verify, clean_env(), self.logs / f'verify-n{n}-{arch}.log', 60)
            except RuntimeError:
                env = clean_env()
                env.update(N=str(n), ARCH=arch, GROUPBATCH='1', SRC=str(SOURCE), OUT=str(binary),
                           LOW_LUT_K=str((n+1)//2), HIGH_LUT_K=str(n-(n+1)//2))
                print(f'Build n={n} {arch}', flush=True)
                run([ROOT / 'scripts/build/b300-hbm32-batch.sh'], env,
                    self.logs / f'build-n{n}-{arch}.log', 900)
                run(verify, clean_env(), self.logs / f'verify-n{n}-{arch}.log', 60)
            self.binaries[key] = binary
        return self.binaries[key]

    def plan(self, n, config):
        key = (n, config.devices, config.scratch_mib, config.lanes, config.dynamic)
        if key not in self.plans:
            env = environment(config, self.hardware, self.args.reserve_mib)
            env['GRIDFP_PLAN_ONLY'] = '1'
            text = run([self.binary(n, config), n, config.scratch_mib, 14,
                        len(config.devices), self.modulus], env,
                       self.logs / f'plan-{len(self.plans)}.log', 180)
            rows = fields(text, 'PLAN n=')
            if len(rows) != 1 or int(rows[0]['n']) != n or int(rows[0]['gpus']) != len(config.devices):
                raise RuntimeError('invalid memory plan')
            self.plans[key] = rows[0]
        return self.plans[key]

    def admissible(self, config):
        return all(fits(self.plan(n, config), config, self.hardware, self.args.reserve_mib)
                   for n in {self.args.n, self.args.bench_n})

    def measure(self, config, deadline):
        record = {'config': asdict(config), 'samples': []}
        self.results.append(record)
        try:
            if not self.admissible(config):
                record['status'] = 'memory-rejected'
                return None
            binary = self.binary(self.args.bench_n, config)
            for trial in range(self.args.repeats + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('tuning budget exhausted')
                log = self.logs / f'bench-{len(self.results)}-{trial}.log'
                start = time.monotonic()
                text = run([binary, self.args.bench_n, config.scratch_mib, 14,
                            len(config.devices), self.modulus],
                           environment(config, self.hardware, self.args.reserve_mib), log,
                           min(remaining, self.args.sample_timeout))
                elapsed = time.monotonic() - start
                rows = fields(text, 'backend=')
                if len(rows) != 1:
                    raise RuntimeError('missing/ambiguous solver result')
                row = rows[0]
                if (int(row['n']) != self.args.bench_n or int(row['modulus']) != self.modulus
                        or int(row['gpus']) != len(config.devices)):
                    raise RuntimeError('solver result configuration mismatch')
                residue = int(row['residue'])
                if not 0 <= residue < self.modulus:
                    raise RuntimeError('invalid residue')
                if self.expected is None:
                    self.expected = residue
                if residue != self.expected:
                    raise ValueError(f'WRONG RESIDUE: {log}')
                wall = float(row['wall_s'])
                if not math.isfinite(wall) or wall <= 0:
                    raise RuntimeError('invalid timing')
                record['samples'].append(dict(warmup=trial == 0, wall_s=wall,
                                              process_s=elapsed, residue=residue,
                                              solver=row, log=str(log)))
            record.update(status='ok', median_s=statistics.median(
                x['wall_s'] for x in record['samples'] if not x['warmup']))
            print(f"{config}: {record['median_s']:.5f}s", flush=True)
            return record['median_s']
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as exc:
            record.update(status='failed', error=str(exc))
            print(f'Skip {config}: {exc}', flush=True)
            return None

    def search(self):
        candidates = []
        for ids in groups(self.hardware, self.args.max_gpus):
            for scratch in self.args.scratch_mib:
                c = Config(ids, scratch, dynamic=int(len(ids) > 1))
                record = {'config': asdict(c)}
                self.planning.append(record)
                try:
                    admitted = self.admissible(c)
                    record.update(admitted=admitted, target_plan=self.plan(self.args.n, c))
                    if admitted:
                        candidates.append(c)
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    record.update(admitted=False, error=str(exc))
                    print(f'Skip planning {c}: {exc}', flush=True)
        if not candidates:
            raise RuntimeError('no memory-feasible full-P2P GPU configuration for the target size')
        # Compilation and CPU memory planning are outside the GPU benchmark budget.
        for c in candidates:
            self.binary(self.args.bench_n, c)
        deadline = time.monotonic() + self.args.budget_seconds
        best, best_time = None, float('inf')
        seen = set()

        def evaluate(config):
            nonlocal best, best_time
            if config in seen or time.monotonic() >= deadline:
                return
            seen.add(config)
            value = self.measure(config, deadline)
            if value is not None and value < best_time:
                best, best_time = config, value

        # Start with moderate scratch, then compare hardware and scratch choices.
        candidates.sort(key=lambda c: (abs(c.scratch_mib - 512), -len(c.devices)))
        for c in candidates:
            evaluate(c)
        if best is None:
            raise RuntimeError('no complete valid benchmark; increase budget/timeout or lower --bench-n')
        # Bounded coordinate search; does not claim an exhaustive global optimum.
        for axis, values in [('lanes', (1, 8, 16, 32)), ('threads', (128, 256, 512)),
                             ('graph', (0, 1, 2)), ('dynamic', (0, 1)),
                             ('lanes', (1, 8, 16, 32))]:
            anchor = best
            for value in values:
                if axis == 'dynamic' and len(anchor.devices) == 1 and value:
                    continue
                evaluate(replace(anchor, **{axis: value}))
        return best


def fingerprint(hardware, binaries, args):
    h = json.loads(json.dumps(hardware))
    for g in h['gpus']:
        g.pop('free_bytes', None)
    return {'hardware': h, 'binary_sha256': {str(p): digest(p) for p in binaries},
            'script_sha256': digest(__file__), 'n': args.n, 'bench_n': args.bench_n,
            'reserve_mib': args.reserve_mib, 'max_gpus': args.max_gpus,
            'scratch_mib': args.scratch_mib, 'repeats': args.repeats,
            'budget_seconds': args.budget_seconds, 'sample_timeout': args.sample_timeout}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('n', type=int, nargs='?', default=27)
    parser.add_argument('--bench-n', type=int, default=20,
                        help='benchmark size; default 20 is a proxy for larger targets')
    parser.add_argument('--max-gpus', type=int, default=8)
    parser.add_argument('--scratch-mib', type=int, nargs='+', default=[512, 2048, 10240])
    parser.add_argument('--reserve-mib', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--budget-seconds', type=float, default=600)
    parser.add_argument('--sample-timeout', type=float, default=90)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--work-dir', type=Path, help='exact CRT checkpoint directory')
    parser.add_argument('--max-runs', type=int, default=0, help='limit new exact CRT residues')
    parser.add_argument('--detect-only', action='store_true')
    parser.add_argument('--tune-only', action='store_true')
    parser.add_argument('--reuse', action='store_true', help='reuse only a matching validated config')
    args = parser.parse_args(argv)
    if not 20 <= args.bench_n <= args.n <= 27:
        parser.error('require 20 <= bench-n <= n <= 27')
    if (not 1 <= args.max_gpus <= 8 or args.repeats < 2 or args.reserve_mib < 256
            or min(args.scratch_mib) < 1 or args.max_runs < 0
            or not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0
            or not math.isfinite(args.sample_timeout) or args.sample_timeout <= 0):
        parser.error('invalid GPU count, memory, repeats (>=2), or time limit')
    args.output = (args.output or ROOT / f'work/autotune/n{args.n}.json').resolve()
    with locked(args.output.with_suffix(args.output.suffix + '.lock'), blocking=False):
        return execute(args)


def execute(args):
    logs = args.output.parent / (args.output.stem + '-' + time.strftime('%Y%m%d-%H%M%S') + f'-{os.getpid()}')
    logs.mkdir(parents=True, exist_ok=False)
    hardware = detect(logs)
    print(json.dumps(hardware, indent=2), flush=True)
    if args.detect_only:
        return 0
    tuner = Tuner(args, hardware, logs)
    chosen = None
    if args.reuse and args.output.exists():
        old = json.loads(args.output.read_text())
        c = old.get('selected', {})
        if c:
            candidate = Config(**{**c, 'devices': tuple(c['devices'])})
            valid_groups = groups(hardware, args.max_gpus)
            if candidate.devices in valid_groups:
                binaries = [tuner.binary(n, candidate) for n in {args.n, args.bench_n}]
                if old.get('fingerprint') == fingerprint(hardware, binaries, args):
                    if tuner.admissible(candidate):
                        chosen = candidate
                        print('Reusing validated configuration', flush=True)
    if chosen is None:
        try:
            chosen = tuner.search()
        except BaseException:
            save(logs / 'failed-search.json', dict(hardware=hardware,
                                                  planning=tuner.planning, results=tuner.results))
            raise
        binaries = [tuner.binary(n, chosen) for n in {args.n, args.bench_n}]
        save(args.output, dict(schema=1, selected=asdict(chosen),
                              fingerprint=fingerprint(hardware, binaries, args),
                              hardware=hardware, planning=tuner.planning, results=tuner.results,
                              selected_target_plan=tuner.plan(args.n, chosen),
                              selected_benchmark_plan=tuner.plan(args.bench_n, chosen),
                              benchmark_is_proxy=args.bench_n != args.n,
                              expected_residue=tuner.expected,
                              independently_known=args.bench_n in KNOWN,
                              method='warmup excluded; median wall_s; bounded coordinate search'))
    print(f'Selected: {chosen}\nReport: {args.output}', flush=True)
    if args.tune_only:
        return 0
    # Refresh free memory after tuning, and validate the target plan again.
    fresh = detect(logs)
    if [g['uuid'] for g in fresh['gpus']] != [g['uuid'] for g in hardware['gpus']]:
        raise RuntimeError('visible GPU order changed; rerun detection')
    if not fits(tuner.plan(args.n, chosen), chosen, fresh, args.reserve_mib):
        raise RuntimeError('free VRAM fell below the target plan; retry after freeing memory')
    command = [sys.executable, ROOT / 'scripts/solve/solve_b300_exact_batch.py', args.n,
               '--binary', tuner.binary(args.n, chosen), '--target-mib', chosen.scratch_mib,
               '--max-window', 14, '--gpus', len(chosen.devices), '--max-runs', args.max_runs]
    if args.work_dir:
        command += ['--work-dir', args.work_dir.resolve()]
    run(command, environment(chosen, fresh, args.reserve_mib))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'autotune: {error}', file=sys.stderr)
        sys.exit(1)
