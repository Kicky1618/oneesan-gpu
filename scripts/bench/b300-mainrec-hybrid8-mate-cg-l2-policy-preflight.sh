#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-mate-cg-l2-policy.py"
python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagep-mate-l2.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
src="$tmp/in.cu"
cat >"$src" <<'EOF'
using Count=unsigned int; using MateID=unsigned long long; using Code=unsigned long long;
__device__ __forceinline__ MateID b300_mainrec_hybrid8_mate_load_policy_cg(const MateID* p){
    return __ldcg(p);
}
__global__ void main_pull_kernel_ilp2(const Count* in,MateID* mates){
    unsigned long long i0=0; const MateID m0=mates[i0]; const Count self0=in[i0]; (void)m0;(void)self0;
}
__global__ void main_pull_kernel_ilp8_hybrid(const Count* in,MateID* mates){
    unsigned long long i0=0,i1=1,i2=2,i3=3,i4=4,i5=5,i6=6,i7=7;
    const MateID m0=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i0);
    const MateID m1=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i1);
    const MateID m2=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i2);
    const MateID m3=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i3);
    const MateID m4=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i4);
    const MateID m5=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i5);
    const MateID m6=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i6);
    const MateID m7=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i7);
    const Count self0=in[i0]; mates[i7]=b300_high_state_advance(m7); (void)m0;(void)self0;
}
EOF
for bytes in 0 64 128 256; do
  out="$tmp/out-$bytes.cu"; log="$tmp/out-$bytes.log"
  python3 "$GEN" "$src" "$out" "$bytes" >"$log"
  grep -Fq "b300_mainrec_stagep_mate_cg_l2=1 l2_bytes=$bytes" "$log" || exit 3
  grep -Fq "// b300_mainrec_stagep_mate_cg_l2=1 l2_bytes=$bytes scope=ilp8_mate_reads_only" "$out" || exit 3
  if [[ "$bytes" == 0 ]]; then grep -Fq 'ld.global.cg.u64' "$out" || exit 3; else grep -Fq "ld.global.cg.L2::${bytes}B.u64" "$out" || exit 3; fi
  [[ "$(grep -o 'b300_mainrec_hybrid8_mate_load_policy_cg(mates+i' "$out" | wc -l)" == 8 ]] || { echo 'Stage-P changed mate call count' >&2; exit 3; }
  grep -Fq 'const MateID m0=mates[i0];' "$out" || { echo 'Stage-P changed ILP2 mate load' >&2; exit 3; }
  grep -Fq 'const Count self0=in[i0];' "$out" || { echo 'Stage-P changed Count load' >&2; exit 3; }
  grep -Fq 'mates[i7]=b300_high_state_advance' "$out" || { echo 'Stage-P changed mate write' >&2; exit 3; }
done
set +e
python3 "$GEN" "$tmp/out-128.cu" "$tmp/double.cu" 64 >/dev/null 2>"$tmp/double.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'already contains Stage-P' "$tmp/double.err" || { echo 'Stage P double transform not rejected' >&2; exit 3; }
cs="$tmp/cs.cu"; sed 's/b300_mainrec_hybrid8_mate_load_policy_cg/b300_mainrec_hybrid8_mate_load_policy_cs/g; s/__ldcg/__ldcs/g' "$src" >"$cs"
set +e
python3 "$GEN" "$cs" "$tmp/cs-out.cu" 128 >/dev/null 2>"$tmp/cs.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'not applicable to Stage-M cs' "$tmp/cs.err" || { echo 'Stage P accepted Stage-M cs upstream' >&2; exit 3; }
set +e
python3 "$GEN" "$src" "$tmp/bad.cu" 32 >/dev/null 2>"$tmp/bad.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'L2_BYTES must be 0,64,128,256' "$tmp/bad.err" || { echo 'Stage P accepted invalid L2 size' >&2; exit 3; }
echo 'b300-mainrec-hybrid8-mate-cg-l2-policy-preflight OK stage=P l2_sizes=0,64,128,256 scope=ilp8_mate_reads_only mate_calls=8 ilp2_unchanged=1 count_loads_unchanged=1 mate_writes_unchanged=1 cs_rejected=1 double_transform_rejected=1 gpu_work=0'
