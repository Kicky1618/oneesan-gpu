#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

TU="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_bucket_snake_onepass_pattern10_depthcode_orbitcta_directgather_graph_batch.cu"
BUILD="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh"
RUN="$ONEESAN_ROOT/scripts/run/b300x8-exact-orbit64.sh"
AB="$ONEESAN_ROOT/scripts/bench/b300-orbit64-sparse64-ab.sh"
AUTO="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto.sh"

for f in "$BUILD" "$RUN" "$AB" "$AUTO"; do bash -n "$f"; done

python3 - "$TU" "$BUILD" "$RUN" <<'PY'
from pathlib import Path
import sys
tu,build,run=map(lambda x:Path(x).read_text(),sys.argv[1:4])
base='#include "../gridfp/ramstream32_bucket_low_rankformula_directgather64.cuh"'
sparse='#include "../gridfp/ramstream32_bucket_low_rankformula_directgather_sparse64.cuh"'
if tu.count(base)!=1 or tu.count(sparse)!=1:
    raise SystemExit('directgather64/sparse64 include count mismatch')
if tu.index(base)>tu.index(sparse):
    raise SystemExit('sparse64 extension must be included after directgather64 base')
for needle in (
    '#if P10DC_RANKFORMULA_DIRECTGATHER_SPARSE64',
    '#define BucketFusedDeviceTables BucketFusedDirectHighRowsRankFormulaNometa4DirectGatherSparse64Tables',
):
    if needle not in tu:raise SystemExit(f'missing TU sparse wiring: {needle}')
for needle in (
    'DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"',
    '-DP10DC_RANKFORMULA_DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64"',
):
    if needle not in build:raise SystemExit(f'missing build sparse wiring: {needle}')
for needle in (
    'DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"',
    'DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64"',
):
    if needle not in run:raise SystemExit(f'missing exact-runner sparse wiring: {needle}')
print('b300-orbit64-sparse-wiring-proof OK include_order=base_then_sparse table_class=sparse64 build_macro=1 runner_flag=1 shell_syntax=1 gpu_required=0')
PY
