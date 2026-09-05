# Reproducible RAM-streaming factorized A/B benchmark

This protocol compares the canonical RAM-streaming baseline (`gridfp-ramstream32-v1`) with the occupancy-major factorized forced-two-window backend on the same machine and under the same modulus, scratch budget, and CPU-thread count.

Use the dedicated harness instead of invoking the binaries with a shared positional argument list. The two programs intentionally have different CLI layouts: v1 uses argument 4 for `max_window`, while factorized v3 uses argument 4 for `cpu_threads`.

## Quick regression

For the established `n=21` case:

```bash
N=21 \
MODULUS=4294967291 \
SCRATCH_MIB=256 \
CPU_THREADS=16 \
REPEATS=3 \
bash scripts/bench/ramstream32-factorized-ab.sh
```

For this `(n, modulus)` pair, the harness automatically requires the known residue

```text
998035516
```

as well as agreement between the two implementations.

## Larger machine run

Choose a scratch budget that is feasible for both backends and keep it identical across the A/B pair. For example:

```bash
N=27 \
MODULUS=4294967291 \
SCRATCH_MIB=16384 \
CPU_THREADS=32 \
REPEATS=3 \
bash scripts/bench/ramstream32-factorized-ab.sh
```

If an independently known residue is available for another case, set it explicitly:

```bash
EXPECTED_RESIDUE=<known-value> ... bash scripts/bench/ramstream32-factorized-ab.sh
```

## Recorded provenance

Each invocation writes a `.meta` file containing:

- repository commit hash and dirty-tree flag;
- hostname;
- GPU name and UUID;
- NVIDIA driver and `nvcc` version;
- `n`, width, architecture, modulus, scratch MiB, CPU threads, v1 `max_window`, and repeat count;
- baseline and factorized binary paths;
- SHA-256 digest of each binary.

The matching `.tsv` contains one row per run with the raw backend output plus parsed `wall_s`, `pack_s`, `h2d_s`, `kernel_s`, `d2h_s`, and `unpack_s` values.

Odd repeats run baseline first; even repeats run factorized first. This does not remove thermal or clock drift, but prevents one backend from always receiving the same execution-order advantage.

The harness finally reports mean wall/pack/kernel time, wall-time speedup, and wall-time reduction.

## Acceptance criteria for paper numbers

A timing result should be treated as publication-grade only when all of the following hold:

1. `dirty=0` in the metadata.
2. Both binary SHA-256 values are recorded.
3. Baseline and factorized residues agree on every repeat.
4. A known independent residue is checked when one exists.
5. The same node, modulus, scratch budget, and CPU-thread count are used for both implementations.
6. At least three repeats are recorded; more are preferable for long runs when allocation cost and system load are variable.
7. The reported table keeps wall time and transition-kernel time separate, because factorized topology can shift work between packing and GPU transition phases.

The result files are written under `work/bench_ramstream32_ab/` by default so experimental output does not need to be committed to the source tree.
