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

## Strict test identity

The A/B driver treats the requested hardware/input tuple as part of correctness, not just metadata.

Before a real run it performs a best-effort visible-GPU preflight using `CUDA_VISIBLE_DEVICES`, or `nvidia-smi` when that variable is absent. If fewer GPUs are visible than `--gpus`, the run is rejected before building/running the huge n=27 case. `--skip-gpu-preflight` exists only for environments where those visibility mechanisms do not reflect CUDA runtime visibility.

After every solver run the parsed result rows must report exactly:

- the requested `n`;
- the requested GPU count;
- the modulus list in the same order;
- `residue_index=0..k-1` and the correct `residues_total`;
- finite nonnegative setup/phase/scratch values and positive `wall_s`.

This catches the solver's current fallback behavior where requesting more GPUs than are visible can otherwise reduce the actual GPU count via `min(requested, visible, 8)`.

## Build provenance

`scripts/build/b300-hbm32-batch.sh` writes a sidecar next to every binary:

```text
<binary>.build.json
```

It records:

- binary SHA-256;
- source SHA-256;
- build-script SHA-256;
- git head at build time;
- source path;
- n / width / LOW / HIGH;
- architecture and the effective nvcc flags.

The A/B driver validates this sidecar before running every candidate, including `--skip-build`. A stale binary, modified source, modified build script, wrong split, wrong architecture, or altered binary is rejected and must be rebuilt. The accepted provenance records are copied into `manifest.json`.

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

The driver verifies the exact modulus sequence and residue equality separately for each modulus.

## Inspect commands without running

```bash
python3 scripts/bench/b300_maskshard_ab.py --dry-run
```

The parser/identity checks have a GPU-free self-test:

```bash
python3 scripts/bench/b300_maskshard_ab.py --self-test
```

To reuse already-built binaries:

```bash
python3 scripts/bench/b300_maskshard_ab.py --skip-build
```

The expected binaries and provenance sidecars are under `build/maskshard-ab/` unless `--build-dir` is supplied.

## Results

By default results are written to:

```text
build/bench/maskshard-ab-YYYYMMDD-HHMMSS/
```

Files include:

- `manifest.json`: git head, build inputs, sources, binary paths and validated build provenance;
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

The `*_sum_s` phase counters are sums over worker/GPU activity, not phase wall-clock durations. Use `wall_s` for end-to-end candidate ranking and use the summed counters to identify which phase changed.

## Failure policy

- Any build or solver nonzero exit stops the comparison.
- Missing/stale build provenance stops before the candidate is run.
- A missing, malformed, wrong-GPU-count or wrong-modulus result stops with exit code 90.
- A residue mismatch stops with exit code 91.
- A candidate that fails guarded HBM admission is not treated as a performance result.

Do not promote v0.7/v0.8/v0.9 from research status based only on the analytical work models. Require identical residues first, then compare phase timing and total wall time on the same B300 node.
