#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-mate-cg-l2-policy.py"
python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageu-matel2.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
src="$tmp/in.cu"
cat >"$src" <<'EOF'
using Count=unsigned int; using MateID=unsigned long long; using Code=unsigned long long;
__device__ __forceinline__ MateID b300_mainrec_hybrid8_mate_load_policy_cg(const MateID* p){ return __ldcg(p); }
__device__ __forceinline__ MateID b300_mainrec_staget_ilp2_mate_load_policy_cg(const MateID* p){ return __ldcg(p); }
__global__ void main_pull_kernel_ilp2(const Count* in,MateID* mates,const Count* in_block){
 bool v1=true; Code i0=0,i1=1,pj0=0,bj0=0;
 const MateID m0=b300_mainrec_staget_ilp2_mate_load_policy_cg(mates+i0); const MateID m1=v1?b300_mainrec_staget_ilp2_mate_load_policy_cg(mates+i1):MateID(0);
 const Count pair0=in[pj0]; const Count block0=in_block[bj0]; const Count self0=in[i0];
 (void)m0;(void)m1;(void)pair0;(void)block0;(void)self0;
}
__global__ void main_pull_kernel_ilp8_hybrid(const Count* in,MateID* mates,const Count* in_block){
 Code i0=0; const MateID m0=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i0); const Count self0=in[i0]; (void)m0;(void)self0;(void)in_block;
}
// b300_mainrec_stagep_mate_cg_l2=1 l2_bytes=128 scope=ilp8_mate_reads_only
// b300_mainrec_stager_ilp2_pair_block_policy=1 pair=cg block=default high_pair=cg high_block=cg high_pair_l2=256 high_block_l2=64 stageq_preserved=1
// b300_mainrec_stages_ilp2_pair_block_cg_l2=1 pair_policy=cg block_policy=default pair_l2_bytes=128 block_l2_bytes=0 high_pair=cg high_block=cg high_pair_l2=256 high_block_l2=64 stageq_preserved=1
// b300_mainrec_staget_ilp2_mate_load_policy=1 policy=cg high_policy=cg high_l2_bytes=128 stages_preserved=1 scope=ilp2_mate_reads_only
EOF
python3 - "$src" "$tmp/high8" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text(); m=re.search(r'__global__\s+void\s+main_pull_kernel_ilp8_hybrid\s*\(',s); assert m
b=s.find('{',m.end()); d=0
for i in range(b,len(s)):
 d+=(s[i]=='{')-(s[i]=='}')
 if d==0: Path(sys.argv[2]).write_text(s[m.start():i+1]); break
PY
for l2 in 0 64 128 256; do
  out="$tmp/u${l2}.cu"; log="$tmp/u${l2}.log"
  python3 "$GEN" "$src" "$out" "$l2" >"$log"
  if [[ "$l2" == 0 ]]; then qual='ld.global.cg.u64'; else qual="ld.global.cg.L2::${l2}B.u64"; fi
  grep -Fq "l2_bytes=$l2 qualifier=$qual low_policy=cg high_policy=cg high_l2_bytes=128 stages_preserved=1 ilp8_byte_identical=1" "$log"
  grep -Fq "asm volatile(\"$qual %0, [%1];\"" "$out"
  grep -Fq 'b300_mainrec_staget_ilp2_mate_load_policy_cg(mates+i0)' "$out"
  grep -Fq 'v1?b300_mainrec_staget_ilp2_mate_load_policy_cg(mates+i1):MateID(0)' "$out"
  grep -Fq '// b300_mainrec_stages_ilp2_pair_block_cg_l2=1' "$out"
  grep -Fq "// b300_mainrec_stageu_ilp2_mate_cg_l2=1 l2_bytes=$l2 high_policy=cg high_l2_bytes=128 stages_preserved=1 scope=ilp2_mate_reads_only" "$out"
  python3 - "$out" "$tmp/high8" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text(); want=Path(sys.argv[2]).read_text(); m=re.search(r'__global__\s+void\s+main_pull_kernel_ilp8_hybrid\s*\(',s); assert m
b=s.find('{',m.end()); d=0
for i in range(b,len(s)):
 d+=(s[i]=='{')-(s[i]=='}')
 if d==0:
  if s[m.start():i+1]!=want: raise SystemExit('Stage U changed ILP8 bytes')
  break
PY
done
# Low-state Stage-T cs is not applicable to Stage U.
sed -e 's/b300_mainrec_staget_ilp2_mate_load_policy_cg/b300_mainrec_staget_ilp2_mate_load_policy_cs/g' -e 's/policy=cg high_policy=/policy=cs high_policy=/' "$src" >"$tmp/lowcs.cu"
set +e; python3 "$GEN" "$tmp/lowcs.cu" "$tmp/lowcs-out.cu" 64 >/dev/null 2>"$tmp/lowcs.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'applicable only when Stage-T ILP2 mate policy is cg' "$tmp/lowcs.err" || exit 3
# High-state cs with no high L2 remains valid: Stage U only changes low-state mate loads.
sed -e 's/b300_mainrec_hybrid8_mate_load_policy_cg/b300_mainrec_hybrid8_mate_load_policy_cs/g' -e '/b300_mainrec_stagep_mate_cg_l2=/d' -e 's/high_policy=cg high_l2_bytes=128/high_policy=cs high_l2_bytes=0/' "$src" >"$tmp/highcs.cu"
python3 "$GEN" "$tmp/highcs.cu" "$tmp/highcs-out.cu" 256 >"$tmp/highcs.log"
grep -Fq 'l2_bytes=256 qualifier=ld.global.cg.L2::256B.u64 low_policy=cg high_policy=cs high_l2_bytes=0' "$tmp/highcs.log"
# Invalid high-state non-cg + nonzero L2 provenance must fail.
sed -e 's/high_policy=cg high_l2_bytes=128/high_policy=cs high_l2_bytes=64/' "$src" >"$tmp/badhigh.cu"
set +e; python3 "$GEN" "$tmp/badhigh.cu" "$tmp/badhigh-out.cu" 64 >/dev/null 2>"$tmp/badhigh.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'invalid high mate L2 provenance on non-cg policy' "$tmp/badhigh.err" || exit 3
# No Stage-T marker is invalid.
grep -v 'b300_mainrec_staget_ilp2_mate_load_policy=' "$src" >"$tmp/not.cu"
set +e; python3 "$GEN" "$tmp/not.cu" "$tmp/not-out.cu" 64 >/dev/null 2>"$tmp/not.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'Stage U requires Stage-T' "$tmp/not.err" || exit 3
# Double transform is rejected.
set +e; python3 "$GEN" "$tmp/u64.cu" "$tmp/double.cu" 128 >/dev/null 2>"$tmp/double.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'already contains Stage-U' "$tmp/double.err" || exit 3
echo 'b300-mainrec-ilp2-mate-cg-l2-stageu-preflight OK stage=U low_mate_policy=cg low_l2=0,64,128,256 high_mate_policy_locked=1 high_mate_l2_locked=1 stage_t_required=1 stage_s_optional=1 ilp8_byte_identical=1 count_loads_unchanged=1 gpu_work=0'
