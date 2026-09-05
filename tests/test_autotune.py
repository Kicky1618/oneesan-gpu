"""CPU-only integration tests: never discover or execute a real GPU."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('oneesan_autotune', ROOT / 'scripts/run/autotune.py')
a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = a
spec.loader.exec_module(a)


def hardware(count=3):
    return dict(driver=13020, runtime=13020, nvcc='test', gpus=[
        dict(index=i, uuid=f'GPU-test-{i}', name='test', major=8, minor=6, sms=80,
             free_bytes=24 << 30, total_bytes=24 << 30, vmm_gran_bytes=2 << 20,
             driver_release='test') for i in range(count)],
        p2p=[[1] * count for _ in range(count)])


FAKE = '''#!/usr/bin/env python3
import os,sys
n,scratch,window,gpus,mod=map(int,sys.argv[1:])
if os.environ.get('GRIDFP_PLAN_ONLY')=='1':
 print(f'PLAN n={n} gpus={gpus} scratch_fits={int(scratch>=512)} dp_peak_gib=2 row6_peak_gib=3')
else:
 lanes=int(os.environ['GRIDFP_GROUPBATCH_LANES'])
 threads=int(os.environ['GRIDFP_THREADS'])
 graph=int(os.environ['GRIDFP_GROUPBATCH_GRAPH'])
 wall=1 + abs(lanes-8)/10 + abs(threads-128)/1000 + abs(graph-1)/10 + (gpus-1)
 residue=2308006916 if n==20 else 2124618149
 print(f'backend=fake n={n} gpus={gpus} modulus={mod} residue={residue} wall_s={wall}')
'''


class AutotuneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / 'fake'
        self.binary.write_text(FAKE)
        self.binary.chmod(0o755)

    def args(self, **changes):
        result = dict(n=20, bench_n=20, reserve_mib=1024, max_gpus=2,
                      scratch_mib=[512], repeats=2, budget_seconds=60, sample_timeout=5)
        result.update(changes)
        return argparse.Namespace(**result)

    def test_topology_is_bidirectional_and_architecture_homogeneous(self):
        h = hardware()
        h['p2p'][1][0] = 0
        h['gpus'][2]['major'] = 9
        self.assertEqual(a.groups(h, 8), [(0,), (2,)])
        h['gpus'][1]['free_bytes'] += 1
        self.assertEqual(a.groups(h, 8), [(1,), (2,)])

    def test_initialization_peak_and_reserve_must_fit_every_gpu(self):
        h, c = hardware(2), a.Config((0, 1))
        plan = dict(dp_peak_gib='1', row6_peak_gib='23', scratch_fits='1')
        self.assertFalse(a.fits(plan, c, h, 1024))
        plan['row6_peak_gib'] = '3'
        self.assertTrue(a.fits(plan, c, h, 1024))
        h['gpus'][1]['free_bytes'] = 4 << 30
        self.assertFalse(a.fits(plan, c, h, 1024))
        plan['scratch_fits'] = '0'
        self.assertFalse(a.fits(plan, a.Config((0,)), h, 1024))

    def test_env_remaps_uuids_and_removes_inherited_experimental_flags(self):
        with patch.dict(os.environ, {'GRIDFP_PLAN_ONLY': '1', 'GRIDFP_DIRECT_ROW8_TENSOR': '1',
                                     'CUDA_VISIBLE_DEVICES': '2,0', 'GROUPBATCH_CHUNK': '256'}):
            env = a.environment(a.Config((1, 0)), hardware(), 1024)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-test-1,GPU-test-0')
        self.assertNotIn('GRIDFP_PLAN_ONLY', env)
        self.assertNotIn('GRIDFP_DIRECT_ROW8_TENSOR', env)
        self.assertNotIn('GROUPBATCH_CHUNK', env)

    def test_search_selects_fastest_measured_and_excludes_warmup(self):
        tuner = a.Tuner(self.args(), hardware(2), self.root)
        with patch.object(tuner, 'binary', return_value=self.binary), contextlib.redirect_stdout(io.StringIO()):
            selected = tuner.search()
        self.assertEqual(selected, a.Config((0,), lanes=8, threads=128))
        for result in tuner.results:
            if result['status'] == 'ok':
                self.assertEqual(len(result['samples']), 3)
                self.assertTrue(result['samples'][0]['warmup'])
                self.assertFalse(result['samples'][1]['warmup'])

    def test_wrong_residue_aborts_instead_of_winning(self):
        self.binary.write_text(FAKE.replace('2308006916', '123'))
        tuner = a.Tuner(self.args(), hardware(1), self.root)
        with patch.object(tuner, 'binary', return_value=self.binary):
            with self.assertRaisesRegex(ValueError, 'WRONG RESIDUE'):
                tuner.search()

    def test_timeout_and_memory_rejection_do_not_produce_a_winner(self):
        tuner = a.Tuner(self.args(scratch_mib=[256]), hardware(1), self.root)
        with patch.object(tuner, 'binary', return_value=self.binary):
            with self.assertRaisesRegex(RuntimeError, 'no memory-feasible'):
                tuner.search()
        tuner = a.Tuner(self.args(), hardware(1), self.root)
        with patch.object(tuner, 'admissible', return_value=True), \
             patch.object(tuner, 'binary', return_value=self.binary), \
             patch.object(a, 'run', side_effect=subprocess.TimeoutExpired('fake', 1)), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(tuner.measure(a.Config((0,)), float('inf')))
        self.assertEqual(tuner.results[0]['status'], 'failed')

    def test_planning_failure_is_not_reported_as_memory_rejection(self):
        tuner = a.Tuner(self.args(), hardware(1), self.root)
        with patch.object(tuner, 'binary', side_effect=RuntimeError('syntax error')):
            with self.assertRaisesRegex(RuntimeError, 'planning/build failures'):
                tuner.search()

    def test_timeout_kills_solver_descendants(self):
        marker = self.root / 'should-not-exist'
        script = self.root / 'spawn.py'
        script.write_text('import subprocess,time,sys\n'
                          'subprocess.Popen([sys.executable,"-c",'
                          f'"import time,pathlib;time.sleep(.5);pathlib.Path({str(marker)!r}).touch()"] )\n'
                          'time.sleep(30)\n')
        with self.assertRaises(subprocess.TimeoutExpired):
            a.run([sys.executable, script], a.clean_env(), self.root / 'timeout.log', .15)
        # The child would create this file after the parent timeout without killpg.
        import time
        time.sleep(.65)
        self.assertFalse(marker.exists())

    def test_cache_fingerprint_tracks_driver_and_binary_but_not_free_memory(self):
        h, args = hardware(1), self.args()
        original = a.fingerprint(h, [self.binary], args)
        h['gpus'][0]['free_bytes'] -= 1 << 30
        self.assertEqual(original, a.fingerprint(h, [self.binary], args))
        h['gpus'][0]['driver_release'] = 'changed'
        self.assertNotEqual(original, a.fingerprint(h, [self.binary], args))
        self.binary.write_text(FAKE + '\n# changed\n')
        self.assertNotEqual(original, a.fingerprint(hardware(1), [self.binary], args))

    def test_same_output_cannot_be_tuned_concurrently(self):
        with a.locked(self.root / 'config.lock'):
            with self.assertRaisesRegex(RuntimeError, 'another process'):
                with a.locked(self.root / 'config.lock', blocking=False):
                    self.fail('lock should not be acquired')

    def test_inventory_respects_physical_uuid_and_rejects_mig(self):
        h = hardware(1)
        def mock_run(command, env, log, timeout):
            return json.dumps(h)
        with patch.object(a, 'BUILD', self.root), patch.object(a.shutil, 'which', return_value='/nvcc'), \
             patch.object(a, 'run', side_effect=mock_run), \
             patch.object(a.subprocess, 'check_output', side_effect=['version', 'sm_100 sm_100f', 'GPU-test-0, 580.1, [N/A]\n']):
            data = a.detect(self.root)
        self.assertEqual(data['gpus'][0]['driver_release'], '580.1')
        with patch.object(a, 'BUILD', self.root), patch.object(a.shutil, 'which', return_value='/nvcc'), \
             patch.object(a, 'run', side_effect=mock_run), \
             patch.object(a.subprocess, 'check_output', side_effect=['version', 'sm_100 sm_100f', 'GPU-test-0, 580.1, Enabled\n']):
            with self.assertRaisesRegex(RuntimeError, 'MIG'):
                a.detect(self.root)

    def test_b300_uses_native_or_cuda_128_family_target(self):
        gpu = hardware(1)['gpus'][0]
        gpu.update(major=10, minor=3)
        self.assertEqual(a.compile_arch(['sm_100', 'sm_100f', 'sm_103'], gpu), 'sm_103')
        self.assertEqual(a.compile_arch(['sm_100', 'sm_100f'], gpu), 'sm_100f')
        with self.assertRaisesRegex(RuntimeError, 'CUDA Toolkit >= 12.9'):
            a.compile_arch(['sm_100'], gpu)

    def test_main_tunes_then_reuses_and_passes_config_to_exact_runner(self):
        output = self.root / 'chosen.json'
        common = ['20', '--output', str(output), '--scratch-mib', '512', '--repeats', '2']
        with patch.object(a, 'detect', return_value=hardware(1)), \
             patch.object(a.Tuner, 'binary', return_value=self.binary), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(a.main(common + ['--tune-only']), 0)
        data = json.loads(output.read_text())
        self.assertTrue(data['independently_known'])
        self.assertEqual(data['selected']['lanes'], 8)
        calls = []
        real_run = a.run

        def capture(command, env, *args, **kwargs):
            if 'solve_b300_exact_batch.py' in str(command[1]):
                calls.append((command, env))
                return ''
            return real_run(command, env, *args, **kwargs)

        with patch.object(a, 'detect', return_value=hardware(1)), \
             patch.object(a.Tuner, 'binary', return_value=self.binary), \
             patch.object(a.Tuner, 'search', side_effect=AssertionError('must reuse')), \
             patch.object(a, 'run', side_effect=capture), contextlib.redirect_stdout(io.StringIO()):
            # Logs have unique run directories even within the same second.
            with patch.object(a.time, 'strftime', return_value='second-run'):
                self.assertEqual(a.main(common + ['--reuse', '--max-runs', '1']), 0)
        self.assertEqual(len(calls), 1)
        command, env = calls[0]
        self.assertEqual(command[command.index('--gpus') + 1], 1)
        self.assertEqual(env['GRIDFP_GROUPBATCH_LANES'], '8')
        self.assertNotIn('GRIDFP_PLAN_ONLY', env)
        self.assertIn('--max-runs', command)


if __name__ == '__main__':
    unittest.main()
