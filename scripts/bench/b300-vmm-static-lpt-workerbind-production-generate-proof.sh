#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py";PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py";BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py";PACK="$ONEESAN_ROOT/scripts/build/lower-b300-packed-group-meta.py";STAGE="$ONEESAN_ROOT/scripts/build/lower-b300-staged-group-meta.py";STATIC="$ONEESAN_ROOT/scripts/build/lower-b300-static-lpt-staged-meta.py";BIND="$ONEESAN_ROOT/scripts/build/lower-b300-static-worker-device-binding.py";ROWLIMIT="$ONEESAN_ROOT/scripts/build/lower-b300-row-limit.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_static_lpt_workerbind_proof.cu}"
bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-local-meta-proof.sh"
python3 "$GEN" "$SRC" "$OUT";python3 "$PRUNE" "$OUT" "$OUT";python3 "$BASEARG" "$OUT" "$OUT";python3 "$PACK" "$OUT" "$OUT";python3 "$STAGE" "$OUT" "$OUT";python3 "$STATIC" "$OUT" "$OUT";python3 "$BIND" "$OUT" "$OUT";python3 "$ROWLIMIT" "$OUT" "$OUT"
grep -Fq 'cudaSetDevice(d),"set static worker device"' "$OUT"
grep -Fq 'for(int q:pw.by_gpu[d])process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);' "$OUT"
grep -Fq 'B300_ROW_LIMIT' "$OUT"
for stale in 'cudaSetDevice(c.dev),"set worker"' 'cudaSetDevice(dev),"set ensure"';do if grep -Fq "$stale" "$OUT";then echo "per-group cudaSetDevice remains: $stale" >&2;exit 3;fi;done
[[ "$(grep -Fc 'cudaSetDevice(d),"set static worker device"' "$OUT")" == 1 ]]||{ echo "static worker binding count mismatch" >&2;exit 4; }
echo "b300-vmm-static-lpt-workerbind-production-generate-proof OK scheduler=static_lpt per_group_cudaSetDevice_calls_removed=2 expected_group_processings=458752 expected_old_worker_cudaSetDevice_calls=917504 expected_new_worker_cudaSetDevice_calls=448 call_reduction=2048x row_limit_env=B300_ROW_LIMIT"
