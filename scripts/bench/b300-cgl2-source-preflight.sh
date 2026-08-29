#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

for f in \
  "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" \
  "$ONEESAN_ROOT/scripts/bench/b300-nextgen-cg-l2size-sweep.sh" \
  "$ONEESAN_ROOT/scripts/run/b300x8-cgl2-fullprime-race.sh"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
  bash -n "$f"
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-cgl2-preflight.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
cat >"$tmp/in.cu" <<'CU'
using Count=unsigned int; using Code=unsigned long long;
unsigned long long high_rec_groups=1;
__global__ void main_pull_kernel_ilp2(const Count* in,const Count* in_block,Count* out){
  Code i0=0,i1=1,pj0=2,pj1=3,bj0=4,bj1=5; bool hp0=true,hp1=true,hb0=true,hb1=true;
  const Count self0=in[i0]; const Count self1=in[i1];
  const Count pair0=hp0?in[pj0]:Count(0); const Count pair1=hp1?in[pj1]:Count(0);
  const Count block0=hb0?in_block[bj0]:Count(0); const Count block1=hb1?in_block[bj1]:Count(0);
  out[0]=self0+pair0+block0; out[1]=self1+pair1+block1;
}
CU

gen="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py"
for z in 0 64 128 256; do
  out="$tmp/out${z}.cu"; log="$tmp/out${z}.log"
  python3 "$gen" "$tmp/in.cu" "$out" "$z" >"$log"
  [[ -s "$out" ]] || exit 3
  grep -Fq "l2_prefetch_bytes=$z" "$log" || exit 3
  if [[ "$z" == 0 ]]; then
    grep -Fq 'ld.global.cg.u32' "$out" || exit 3
    ! grep -Fq 'ld.global.cg.L2::' "$out" || exit 3
  else
    grep -Fq "ld.global.cg.L2::${z}B.u32" "$out" || exit 3
  fi
  grep -Fq 'const Count self0=in[i0];' "$out" || exit 3
done
# Historical two-argument invocation must remain byte-policy compatible.
python3 "$gen" "$tmp/in.cu" "$tmp/legacy.cu" >"$tmp/legacy.log"
grep -Fq 'l2_prefetch_bytes=0' "$tmp/legacy.log" || exit 3
grep -Fq 'ld.global.cg.u32' "$tmp/legacy.cu" || exit 3

# Source-level builder contract: the new knob is isolated and off by default.
grep -Fq 'RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"' "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" || exit 4
grep -Fq 'RANDOM_CG_L2_FETCH_BYTES>0 requires RANDOM_CG=1' "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" || exit 4
grep -Fq 'gen-b300-mainrec-random-cg.py" "$CURRENT" "$NEXT" "$RANDOM_CG_L2_FETCH_BYTES"' "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" || exit 4

echo 'b300_cgl2_source_preflight=OK legacy_cg=1 l2_fetch_sizes=64,128,256 shell_syntax=1 gpu_work=0'
