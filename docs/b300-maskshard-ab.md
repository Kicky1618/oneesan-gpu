# B300 mask-shard A/B runbook

Use `scripts/bench/b300_maskshard_ab.py` for the first full-P2P B300 comparison of the experimental mask-shard backends.

The driver runs candidates **sequentially** so each process releases its HBM before the next candidate starts. Every variant receives exactly the same `n`, GPU count, thread count and modulus list. The run fails immediately if residues disagree.

## First correctness run

From the repository root:

```bash
python3 scripts/bench/b300_maskshard_ab.py \
  --variants v0.4 v0.7 v0.8 v0.9 \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291
```

Default sources are:

- v0.4: `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu`
- v0.7: `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_batch_guarded.cu`
- v0.8: `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_highclosurerows_batch_guarded.cu`
- v0.9: `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosurerows_batch_guarded.cu`

The build uses `scripts/build/b300-hbm32-batch.sh`. `ARCH=native` is used by default; override with `--arch` when required by the installed CUDA toolkit.

## HBM reserve

To preserve an explicit amount of free HBM for the guarded allocator:

```bash
python3 scripts/bench/b300_maskshard_ab.py \
  --vram-reserve-mib 1024 \
  --variants v0.4 v0.7 v0.8 v0.9 \
  --modulus 4294967291
```

This sets `GRIDFP_VRAM_RESERVE_MIB` for every candidate.

## Multiple CRT primes

The batch backends accept multiple moduli in one process. Repeat `--modulus`:

```bash
python3 scripts/bench/b300_maskshard_ab.py \
  --variants v0.4 v0.7 v0.8 v0.9 \
  --modulus 4294967291 \
  --modulus 4294967279
```

The driver verifies residue equality separately for each modulus.

## Inspect commands without running

```bash
python3 scripts/bench/b300_maskshard_ab.py --dry-run
```

To reuse already-built binaries:

```bash
python3 scripts/bench/b300_maskshard_ab.py --skip-build
```

The expected binaries are under `build/maskshard-ab/` unless `--build-dir` is supplied.

## Results

By default results are written to:

```text
build/bench/maskshard-ab-YYYYMMDD-HHMMSS/
```

Files include:

- `manifest.json`: git head, build inputs, sources and binary paths;
- `v0.X.stdout.log`: raw solver stdout;
- `v0.X.stderr.log`: raw solver diagnostics;
- `summary.json`: parsed result rows after each completed candidate.

The terminal comparison includes:

- `wall_s`;
- `high_io_sum_s`;
- `high_orbit_sum_s`;
- `high_closure_sum_s`;
- `low_orbit_sum_s`;
- `low_closure_sum_s`;
- `max_scratch_gib`;
- wall-time ratio against the first variant for each modulus.

## Failure policy

- Any build or solver nonzero exit stops the comparison.
- A missing/malformed result row stops with exit code 90.
- A residue mismatch stops with exit code 91.
- A candidate that fails guarded HBM admission is not treated as a performance result.

Do not promote v0.7/v0.8/v0.9 from research status based only on the analytical work models. Require identical residues first, then compare phase timing and total wall time on the same B300 node.
