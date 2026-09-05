# oneesan-gpu

GPU frontier-DP experiments for counting simple corner-to-corner paths on an `(n+1) x (n+1)` vertex grid (`n x n` cells).

The main target is an 8-GPU NVIDIA B300 node. The current production path stores the authoritative DP state as `uint32` residues sharded across all 8 GPUs, then reconstructs the exact integer with CRT.

## Local / single GPU

For development and correctness checks on an ordinary CUDA GPU, use the local wrapper. It builds for the GPU installed in the machine and forces a single-GPU run.

Quick one-residue run:

```bash
./scripts/run/local.sh 18
```

The defaults are intentionally conservative for an 8 GiB-class development GPU:

```text
N=18
NGPU=1
ARCH=native
TARGET_MIB=512
GRIDFP_VRAM_RESERVE_MIB=1024
modulus=4294967291
```

On the development RTX 3070 8GB, the current `n=18` one-residue path completes in about 3 seconds and uses about 169 MiB for the authoritative state, plus scratch/LUT allocations.

To reconstruct the exact count with CRT instead of computing only one residue:

```bash
./scripts/run/local.sh 18 --exact
```

The exact run is checkpointed under `work/b300_exact_n18/` and can be resumed with the same command **only with the same solver binary**. Exact checkpoint format V3 records the binary SHA-256 and Git commit, adds a SHA-256 integrity checksum over the canonical checkpoint payload, rejects duplicate/unknown/missing JSON fields, and uses fsync + atomic rename before a residue is considered durable. Legacy V1/V2 checkpoints and checkpoints produced by a different binary are rejected. To verify the exact pipeline without running every CRT modulus:

```bash
./scripts/run/local.sh 18 --exact --max-runs 1
```

You can override the memory budget through environment variables:

```bash
TARGET_MIB=256 \
GRIDFP_VRAM_RESERVE_MIB=1536 \
./scripts/run/local.sh 18
```

Larger `n` values require rapidly increasing state memory; the B300 x8 path below is the intended route for `n=27`.

## Automatic GPU configuration (GROUPBATCH, n=20..27)

Detect visible GPUs, check full P2P connectivity and memory capacity, benchmark
configuration candidates, then launch the checkpointed exact CRT solver:

```bash
python3 scripts/run/autotune.py 27
```

The default n=20 benchmark is a proxy for larger problems, so the chosen settings
are the best measured candidates rather than a guarantee of a global optimum.
Use `--detect-only` for inventory, `--tune-only` to save settings without solving,
or `--reuse` to reuse matching hardware/binary settings after memory validation.
See [configuration search, limits, and examples](docs/autotune.md).

## B300 x8: exact n=27

### Requirements

- Linux
- NVIDIA driver + CUDA Toolkit (`nvcc`)
- 8 visible B300 GPUs
- full GPU-to-GPU P2P connectivity between all 8 GPUs (NVLink/NVSwitch)
- Python 3 for checkpointing and CRT reconstruction

Check the node first:

```bash
nvidia-smi -L
nvidia-smi topo -m
nvcc --version
```

The HBM solver requires full directed P2P connectivity. If all GPU pairs are not accessible, the solver aborts instead of silently falling back to a slow path.

### Recommended: exact count

From the repository root:

```bash
ARCH=native ./scripts/run/b300x8-exact.sh 27
```

This command automatically:

1. builds `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu` if necessary;
2. allocates/shards the authoritative state across 8 GPUs;
3. evaluates the required near-`2^32` CRT moduli in one process;
4. checkpoints each completed residue;
5. reconstructs the exact integer once the CRT modulus is large enough.

For `n=27`, the solver uses a rigorous checkerboard-strip upper bound with row partition `[9, 9, 9]`. The bound has 633 bits, so 20 near-`2^32` primes are sufficient; their product has 640 bits.

Checkpoint and final result files are written to:

```text
work/b300_exact_n27/checkpoint.json
work/b300_exact_n27/exact.txt
work/b300_exact_n27/exact_manifest.json
```

A stopped run can be resumed with the same command; completed residues are reused. On successful
completion, the runner writes both `exact.txt` and `exact_manifest.json`. The manifest contains
the full CRT congruence list, checkerboard-strip bound provenance, solver binary SHA-256, exact
checkpoint SHA-256, `exact.txt` SHA-256, optional build-provenance SHA-256, and its own SHA-256
integrity checksum. New builds use `ONEESAN_BUILD_PROVENANCE_V2`: besides the recursive quoted-include
source closure, n=27 row-6 builds record the exact rational certificate, CRT20 generator, verifier,
and shared path-bound/prime source as auxiliary generation dependencies. Legacy V1 sidecars remain
readable. The V3 exact-result verifier recomputes the checkerboard-strip bound with an implementation independent of the runner, checks
each CRT modulus for 64-bit primality and coprimality, reconstructs CRT from scratch, and then
checks the checkpoint and result-file hashes. With `--verify-sources`, a V2 row-6 provenance also
requires the canonical four auxiliary-generation roles and reruns the exact-rational certificate
check against the current generated CRT20 header. Verify it with:

```bash
./scripts/tools/verify_exact_result.py work/b300_exact_n27/exact_manifest.json \
  --binary build/oneesan_cuda_gridfp_b300_hbm32_batch_n27 \
  --verify-sources
```


To compute only one new residue as a smoke test:

```bash
ARCH=native ./scripts/run/b300x8-exact.sh 27 --max-runs 1
```

### One-residue run

For benchmarking or regression testing without CRT:

```bash
ARCH=native ./scripts/run/b300x8.sh 27 4294967291
```

The output contains fields such as:

```text
residue=...
modulus=4294967291
gpus=8
peers=56
wall_s=...
```

`peers=56` means all directed links between 8 GPUs are available (`8 * 7`).

### Build manually

Optimized multi-modulus batch binary:

```bash
N=27 ARCH=native ./scripts/build/b300-hbm32-batch.sh
```

Output:

```text
build/oneesan_cuda_gridfp_b300_hbm32_batch_n27
```

The same optimized batch binary is also used for one-residue runs; passing a single modulus avoids maintaining a separate slower n=27 execution path.

`ARCH=native` is preferred when building directly on the target B300 node. An explicit architecture can still be supplied through `ARCH` when needed.

### B300 tuning variables

| Variable | Default | Meaning |
| --- | ---: | --- |
| `NGPU` | `8` | number of GPUs |
| `TARGET_MIB` | `16384` | requested scratch arena per GPU |
| `MAX_WINDOW` | `14` | maximum transition window |
| `GRIDFP_VRAM_RESERVE_MIB` | `8192` | HBM kept free after authoritative-state allocation |
| `GRIDFP_BOUNDED_PREFIX_K` | `6` for `n=27` | initialize directly after six rows |
| `GRIDFP_ROW6_LANES` | `4` | CUDA lanes cooperating on each row-6 initialized state |
| `ARCH` | `native` | CUDA architecture passed to `nvcc` |
| `ONEESAN_BUILD_DIR` | `./build` | build output directory |

Example:

```bash
TARGET_MIB=12288 \
GRIDFP_VRAM_RESERVE_MIB=8192 \
ARCH=native \
./scripts/run/b300x8-exact.sh 27
```

For the current `n=27` layout, the authoritative `uint32` state is about 1939.9 GiB total, or about 242.5 GiB per GPU on 8 GPUs. LUTs and scratch are allocated on top of that, so this mode is specifically designed for very-high-HBM multi-GPU nodes.

## mmap multi-GPU backend

There is also an external-store backend that keeps the authoritative state in mmap files instead of sharding it entirely in HBM. It accepts a 64-bit modulus and does not require P2P for correctness.

Build and run:

```bash
ARCH=native ./scripts/build/gridfp-multigpu-mmap.sh
TARGET_MIB=245760 ./scripts/run/b300x8-mmap.sh \
  26 \
  2305843009213693951 \
  /fast-local-storage/gridfp-n26
```

For `n=27`, the two 64-bit-residue external state files total roughly 3.79 TiB. The backend preallocates them with `posix_fallocate()` before computation so ENOSPC fails early. The multi-GPU mmap runner now uses crash-safe per-group undo journals and a durable completion bitmap by default, so rerunning the same command resumes from the interrupted transition window. `GRIDFP_FRESH=1` explicitly starts over; `GRIDFP_RESUME=0` is available only when benchmark speed is more important than crash recovery. Use local NVMe/RAID rather than a network filesystem and leave additional temporary space for in-flight undo journals.


## Correctness audit

Before a long record attempt or release, run the repository-wide local correctness gate:

```bash
./scripts/test/correctness.sh
```

It checks exact/CRT checkpoint safety, arbitrary-precision CPU golden values, ZDD validation,
mmap persistence/corruption/crash-restart behavior, shell input hardening, formal-source hygiene,
and (when `nvcc` is installed) the production Grid-FP partition invariant. On a real NVIDIA
machine, `./scripts/test/mmap-fault-integration.sh` additionally injects crashes at all three
durable mmap transaction boundaries and compares every resumed residue with an uninterrupted
baseline.

## ZDD output

The complete family of simple corner-to-corner paths can be reconstructed as a ZDD over **all real grid edges** by replaying the same Grid-FP `MateID` transition system used by the factorized CUDA solver. RAPiDD is used only as the reduced ZDD node manager.

```bash
./scripts/build/zdd-replay.sh
./scripts/run/zdd-replay.sh 9 work/zdd/replay_n9.zdd \
  --sapporo work/zdd/replay_n9.sapporo.zbdd
./scripts/tools/validate_zdd.py work/zdd/replay_n9.zdd
```

An independent SIMPATH-style builder is also kept for differential verification. See [`docs/zdd.md`](docs/zdd.md) for transition sharing, full edge-set equality tests, the format, and SAPPOROBDD interoperability.

## Repository layout

```text
.
├── README.md
├── src/
│   ├── cuda/
│   │   ├── b300/       # B300/HBM32 kernels and optimization variants
│   │   ├── gridfp/     # generic Grid-FP and mmap/multi-GPU kernels
│   │   ├── hopper/     # H100/Hopper experiments
│   │   └── legacy/     # older hash/multi-residue CUDA implementations
│   └── cpp/
│       ├── oneesan_cpu.cpp
│       ├── oneesan_frontier.cpp
│       └── probes/     # rank/state/transition analysis helpers
├── scripts/
│   ├── build/          # compilation entry points
│   ├── run/            # B300, mmap, regression and remote runners
│   ├── solve/          # CRT/exact-count drivers
│   ├── bench/          # benchmark drivers
│   ├── tools/          # analysis utilities
│   └── lib/            # shared shell helpers
├── docs/
│   ├── b300-hbm32.md
│   ├── multigpu-mmap.md
│   ├── h100.md
│   ├── results/
│   └── assets/
├── third_party/
│   └── ggcount/        # reference implementation + license
└── visualization/
    └── manim/          # explanation-video project
```

Generated binaries and temporary CUDA files go under `build/` and are ignored by Git.

## Documentation

- [`docs/b300-hbm32.md`](docs/b300-hbm32.md): B300 HBM32 implementation, memory budget and optimization history
- [`docs/multigpu-mmap.md`](docs/multigpu-mmap.md): external-store multi-GPU backend
- [`docs/h100.md`](docs/h100.md): older H100 backend
- [`docs/results/`](docs/results/): benchmark and regression notes
- [`visualization/manim/`](visualization/manim/): Manim explanation video
