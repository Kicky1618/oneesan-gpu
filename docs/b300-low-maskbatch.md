# B300 LOW mask-batch research

This note tracks the LOW-side execution redesign after the exact row-depth
kernels.  The goal is to preserve the exact in-place DP schedule while removing
per-mask launch/configuration overhead and keeping the n=27 authoritative state
arrays resident on an HGX B300 x8 node.

All performance numbers below are structural models unless explicitly marked as
measured.  Keep PR #12 draft until fresh nvcc validation and a real B300 x8
full-P2P residue/profile run complete.

## Invariants

For W=28 / n=27:

- authoritative states: 520,735,012,027 uint32 entries;
- MAIN: 385,719,506,620;
- BLOCKED: 135,015,505,407;
- HIGH-mask LPT sharding owns each LOW group on exactly one GPU;
- LOW orbit and LOW closure execute in the original `p=LOW_LUT_K..1` order;
- every optimization must return the same residue for every tested CRT prime.

The LOW mask-batch backends do not change the authoritative state layout or the
mathematical transition relation.  They change only scheduling, metadata
representation, and launch/configuration mechanics.

## v0.43-v0.49: batch and remove repeated configuration

### v0.43: mask-batched executor

One launch contains work from many owner-local HIGH masks.  A descriptor selects
`mask/local/replica/replicas`; the kernel grid-strides the exact LOW orbit or
closure warp-task set belonging to that mask.

### v0.44: static-cache baseline

v0.44 gives the per-mask baseline the same static cache as v0.43 so the direct
v0.44 -> v0.43 comparison isolates batching rather than cache construction.

### v0.45: resident row descriptors

All row-cap descriptor plans are generated at setup and copied once.  The DP
loop no longer rebuilds/copies descriptor lists on every row/residue.

At the old `max_replicas=16` setting the conservative n=27 bound was about
26.25 MiB/GPU resident descriptors, replacing up to about 52.5 MiB/GPU of LOW
descriptor H2D traffic per residue and up to 448 descriptor-copy calls.

### v0.46: CTA shared config cache

Mask/static configuration is staged once per CTA into shared memory.  For
W=28 the modeled shared structures are approximately:

```text
orbit   2552 B / CTA
closure 3144 B / CTA
```

This only became attractive after batching because one CTA now processes many
warp tasks and amortizes the shared-memory staging barrier.

### v0.47: compact resident dynamic config

`closure_begin` and `closure_selected` are mask-independent.  Staging those
arrays from canonical closure tables instead of duplicating them for every
mask/cap changes the dynamic struct from 11,164 B to 3,884 B per mask/cap.
For the n=27 x8 layout this removes about 99.53125 MiB/GPU.

### v0.48: rebuild dynamic metadata per CTA

The remaining orbit/closure prefixes are reconstructed from canonical exact
count tables during CTA staging.  Persistent dynamic config becomes zero; the
only added device constant table is about 900 B.

### v0.49: fast rebuild setup

Once dynamic configs are no longer stored, setup does not need to rebuild every
mask/cap packed config merely to compare it with the independent task table.
For n=27 this removes 114,688 redundant packed-config constructions across the
8 GPUs while retaining the independent task-table construction used by the
runtime schedule.

## v0.50: uint16 replica ABI

The original uint8 replica fields capped a large mask/stage group at 16 CTAs in
practice.  v0.50 widens replica indices/counts to uint16 while keeping the GPU
descriptor at 8 B:

```text
uint16 mask
uint16 local
uint16 replica
uint16 replicas
```

The exact W=28 task model with `target_tasks_per_cta=16384` gives:

```text
max orbit group tasks   15,954,186
max closure group tasks 14,097,070

max replicas  worst warp-tasks/CTA
16            997,137
64            249,285
256            62,322
512            31,161
1024           16,384
```

At the 1024 cap the largest group actually needs 974 replicas.  The worst GPU
holds 4,545,074 expanded 8-byte descriptors, about 34.676 MiB.

## v0.51-v0.52: CTA work target sweep

Replica width exposed a second trade-off: very small CTA work targets control
the tail but multiply CTA/shared-staging overhead.  Exact structural modeling
for W=28, maxrep=1024 gives approximately:

```text
target warp-tasks/CTA   LOW CTAs/residue   shared-stage traffic/residue
16,384                  168.41 M           0.4326 TiB
32,768                   86.00 M           0.2209 TiB
65,536                   44.41 M           0.1142 TiB
131,072                  23.78 M           0.0613 TiB
262,144                  14.11 M           0.0365 TiB
524,288                   9.55 M           0.0247 TiB
1,048,576                 7.51 M           0.0194 TiB
```

These numbers do **not** choose the fastest B300 setting.  They quantify the
launch/staging side of the trade-off; the final setting must come from measured
`wall_s` and `low_batch_sum_s`.

## v0.53: compact replica ranges

Expanded descriptors redundantly store one 8-byte record for every replica CTA.
Replicas of a mask are consecutive `0..R-1`, so v0.53 stores one range per
active mask/stage:

```text
uint32 cta_end
uint16 mask
uint16 local
```

`blockIdx.x` binary-searches cumulative `cta_end`, then reconstructs:

```text
begin    = previous cta_end
replica  = blockIdx.x - begin
replicas = cta_end - begin
```

The host self-test expands the range representation and checks every generated
CTA against the original expanded descriptor schedule, including replica counts
above 255.

For the modeled W=28 LPT shard, the worst GPU drops from 4,545,074 expanded
descriptors to 214,128 range descriptors at the 16K target:

```text
expanded descriptor HBM ~34.676 MiB/GPU
range descriptor HBM     ~1.634 MiB/GPU
reduction                 ~95.29%
```

The trade-off is one small range lookup per CTA.

## v0.54: compact ranges + 64K CTA target

v0.54 combines v0.53 with `target_tasks_per_cta=65536`.  The compact range table
remains indexed by active mask/stage rather than replica count, while the
modeled LOW CTA count falls to about 44.41 M/residue.  The largest group needs
at most 244 replicas.

## v0.55: CUDA Graph row replay

One LOW row contains up to 28 ordered kernel nodes:

```text
for p = LOW_LUT_K .. 1:
    orbit(p)
    closure(p)
```

v0.55 lazily captures one graph per `(GPU, cap)` and replays it on later rows and
CRT residues.  This preserves the same GPU dependency order; it changes host
submission only.

For W=28 x8 the steady-state host launch-call model is:

```text
ordinary kernel submissions / residue = 8 * 28 * 28 = 6,272
graph submissions / residue           = 8 * 28      =   224
reduction                                           = 96.43%
max graph executables                  = 8 * 14      =   112
```

HIGH work is synchronized inside each HIGH job before the LOW phase begins, so
using a nonblocking graph stream does not remove a required HIGH -> LOW device
dependency.

## v0.56: one-binary runtime tuner

The range partition is a setup-time scheduling choice, not a kernel-template
semantic.  v0.56 therefore reads optional environment variables before building
resident row plans:

```text
ONEESAN_LOW_TARGET_TASKS_PER_CTA
ONEESAN_LOW_MAX_REPLICAS
```

The default remains target=65,536 and maxrep=1024.  The sweep driver builds one
binary and runs multiple target settings while checking that every CRT residue
matches:

```bash
python3 scripts/bench/b300_maskshard_lowmaskbatch_runtime_target_sweep.py \
  --n 27 --gpus 8 --threads 256 --low 14 --high 13 \
  --modulus 4294967291 --vram-reserve-mib 1024
```

The default sweep is:

```text
16K, 32K, 64K, 128K, 256K, 512K, 1M warp-tasks/CTA
```

This is the preferred B300 tuning path: compile once, preserve exact residue
checks, and select by measured wall time rather than by the structural model.

## Validation ladder

Before promoting any v0.43+ backend:

1. CPU structural/self-tests must pass with `-Wall -Wextra -Werror`;
2. W=10, W=22 and W=28 backend translation units must compile with fresh nvcc;
3. small-width GPU residues must match the preceding backend for multiple
   uint32 CRT primes;
4. full-P2P multi-GPU residues must match;
5. B300 x8 profile must attribute any wall-time change to the intended LOW
   phase;
6. only then select the runtime CTA target and promote the backend.

GitHub Actions queue/infrastructure failures with no executed steps are not
compile evidence and should not be treated as algorithm failures.
