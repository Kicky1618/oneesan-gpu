# Grid-FP multi-GPU

Build on the target machine so nvcc chooses the installed GPU architecture:

```bash
ARCH=native ./scripts/build/gridfp-multigpu-mmap.sh
```

B300 x8 example, one CRT modulus:

```bash
./scripts/run/b300x8-mmap.sh 27 2305843009213693951 /raid/gridfp/n27_p0
```

Arguments to the binary:

```text
binary n modulus target_mib_per_gpu max_window gpu_count store_directory
```

The scheduler uses the paper's transition-closed groups. A host thread owns each GPU,
and an atomic work queue assigns the largest groups first. Each device has independent
CUDA constant state and buffers, so kernels execute concurrently. CUDA peer access is
enabled automatically wherever the platform reports P2P (NVLink/NVSwitch ready).
The current data path keeps the authoritative state in the mmap external store; it does
not yet require peer copies for correctness.

For n=27 with one 64-bit residue the two external state files are about 3.79 TiB in total.
The solver calls `posix_fallocate()` before mapping them, so insufficient free space is reported
before a multi-day computation begins instead of surfacing later as `SIGBUS`. The backing
filesystem must support `posix_fallocate`; use local NVMe/RAID and leave additional space for
filesystem metadata and other jobs.

Crash-safe resume is enabled by default in the multi-GPU mmap backend. Each transition-closed
group is committed transactionally with a small checkpoint bitmap and a per-group undo journal:

1. gather the original main/blocked ranges;
2. durably write and rename the undo journal;
3. run the GPU transition and scatter the new ranges into the mmap files;
4. `msync(MS_SYNC)` only the touched page ranges;
5. durably set the group-complete bit;
6. durably remove the undo journal.

After a crash, any journal whose group-complete bit is still clear is restored before that group
is recomputed. Completed groups are skipped. Checkpoints bind `n`, modulus, memory/window
configuration, state sizes, and an executable fingerprint, so a different solver binary is not
silently mixed with old state. The store directory is protected by an exclusive file lock.

Checkpoint and undo formats are versioned (`V2`) and carry integrity checksums. Checkpoint
parsing rejects truncation, unknown/duplicate/missing fields, malformed completion flags, and a
checksum mismatch. Undo recovery validates the header checksum, exact file size, metadata, and
payload checksum *before* restoring any bytes. The mmap checkpoint/journal layer uses FNV-1a
64-bit checksums: these are intended for accidental storage-corruption detection, not
cryptographic authentication. Final exact-result provenance uses SHA-256 instead. V1
experimental mmap checkpoints are deliberately not accepted by the V2 reader.

Use the same command to resume. Set `GRIDFP_FRESH=1` to deliberately discard an existing
checkpoint and start from zero. `GRIDFP_RESUME=0` disables journaling for benchmark-only runs;
it is not the recommended mode for a long computation. Because up to one undo journal can be
live per GPU worker, leave temporary free space in addition to the 3.79 TiB authoritative state.
Before each transition window, the runner computes the worst-case sum of the largest `GPU_COUNT`
journal payloads and checks filesystem free space. If that reserve plus 64 MiB is unavailable,
it stops before modifying any group in that window. The final run summary reports
`max_journal_reserve_bytes`.

### Partition invariant self-test

Crash-safe parallel scatter relies on all groups in one transition window owning disjoint main
and blocked state ranges. The production binary contains a GPU-independent exhaustive self-test
for this invariant. It enumerates widths 3 through 10, every window, every prefix of the fixed
occupancy positions, and every resulting group, then checks both pairwise non-overlap and exact
coverage of the global ranked state arrays.

```bash
./scripts/test/gridfp-partition.sh
```

This requires `nvcc` to build the CUDA translation unit but does not require a working NVIDIA GPU
or driver at runtime. The current exhaustive suite checks 824 partition cases and 14,196 groups.


### Real-GPU crash injection

The production binary also has opt-in fault points at the three durability boundaries that matter
for recovery: immediately after the undo journal is durable (`journal`), after the updated mmap
ranges have been synchronously flushed (`scatter`), and after the completion bit is durable but
before the stale undo journal is removed (`commit`). The test process exits with code 86 at the
selected point, approximating an abrupt process death without running destructors.

On a machine with a working NVIDIA GPU/driver, run:

```bash
./scripts/test/mmap-fault-integration.sh
```

The script computes a small baseline, injects each of the three crashes into a fresh store, resumes
without fault injection, and requires the resumed residue to equal the uninterrupted baseline for
every phase. On machines without a usable GPU it exits with code 77 (skip). The underlying debug
controls are `GRIDFP_FAULT_GROUP=<g>` and `GRIDFP_FAULT_PHASE=journal|scatter|commit`; they are
inert unless both are explicitly set.
