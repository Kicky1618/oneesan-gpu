#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-mate-load-policy.py"
python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-staget-mate.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
src="$tmp/in.cu"
cat >"$src" <<'EOF'
using Count=unsigned int; using MateID=unsigned long long; using Code=unsigned long long;
__device__ __forceinline__ MateID b300_mainrec_hybrid8_mate_load_policy_cg(const MateID* p){ return __ldcg(p); }
__global__ void main_pull_kernel_ilp2(const Count* in,MateID* mates,const Count* in_block){
 bool v1=true; Code i0=0,i1=1,pj0=0,bj0=0;
 const MateID m0=mates[i0]; const MateID m1=v1?mates[i1]:MateID(0);
 const Count pair0=in[pj0]; const Count block0=in_block[bj0]; const Count self0=in[i0];
 (void)m0;(void)m1;(void)pair0;(void)block0;(void)self0;
}
__global__ void main_pull_kernel_ilp8_hybrid(const Count* in,MateID* mates,const Count* in_block){
 Code i0=0,i1=1; const MateID m0=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i0); const MateID m1=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i1); const Count self0=in[i0]; (void)m0;(void)m1;(void)self0;(void)in_block;
}
// b300_mainrec_stagep_mate_cg_l2=1 l2_bytes=128 scope=ilp8_mate_reads_only
// b300_mainrec_stager_ilp2_pair_block_policy=1 pair=cg block=default high_pair=cg high_block=cg high_pair_l2=256 high_block_l2=64 stageq_preserved=1
// b300_mainrec_stages_ilp2_pair_block_cg_l2=1 pair_policy=cg block_policy=default pair_l2_bytes=128 block_l2_bytes=0 high_pair=cg high_block=cg high_pair_l2=256 high_block_l2=64 stageq_preserved=1
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
python3 "$GEN" "$src" "$tmp/cs.cu" cs >"$tmp/cs.log"
grep -Fq 'policy=cs intrinsic=__ldcs high_policy=cg high_l2_bytes=128 stages_preserved=1 lanes=2 ilp8_byte_identical=1' "$tmp/cs.log"
grep -Fq 'const MateID m0=b300_mainrec_staget_ilp2_mate_load_policy_cs(mates+i0);' "$tmp/cs.cu"
grep -Fq 'const MateID m1=v1?b300_mainrec_staget_ilp2_mate_load_policy_cs(mates+i1):MateID(0);' "$tmp/cs.cu"
grep -Fq '// b300_mainrec_stages_ilp2_pair_block_cg_l2=1' "$tmp/cs.cu"
python3 - "$tmp/cs.cu" "$tmp/high8" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text(); want=Path(sys.argv[2]).read_text(); m=re.search(r'__global__\s+void\s+main_pull_kernel_ilp8_hybrid\s*\(',s); assert m
b=s.find('{',m.end()); d=0
for i in range(b,len(s)):
 d+=(s[i]=='{')-(s[i]=='}')
 if d==0:
  if s[m.start():i+1]!=want: raise SystemExit('Stage T changed high-state ILP8 bytes')
  break
PY
# High-state cs, no Stage-P L2 and no accepted Stage-S marker: Stage T must still compose directly after R.
sed -e 's/b300_mainrec_hybrid8_mate_load_policy_cg/b300_mainrec_hybrid8_mate_load_policy_cs/g' -e 's/return __ldcg(p)/return __ldcs(p)/' -e '/b300_mainrec_stagep_mate_cg_l2=/d' -e '/b300_mainrec_stages_ilp2_pair_block_cg_l2=/d' "$src" >"$tmp/r-only.cu"
python3 "$GEN" "$tmp/r-only.cu" "$tmp/cg.cu" cg >"$tmp/cg.log"
grep -Fq 'policy=cg intrinsic=__ldcg high_policy=cs high_l2_bytes=0 stages_preserved=0 lanes=2 ilp8_byte_identical=1' "$tmp/cg.log"
grep -Fq 'b300_mainrec_staget_ilp2_mate_load_policy_cg(mates+i0)' "$tmp/cg.cu"
# Conflicting Stage-P high L2 provenance on high-state cs must fail.
cp "$tmp/r-only.cu" "$tmp/badp.cu"; echo '// b300_mainrec_stagep_mate_cg_l2=1 l2_bytes=64 scope=ilp8_mate_reads_only' >>"$tmp/badp.cu"
set +e; python3 "$GEN" "$tmp/badp.cu" "$tmp/badp-out.cu" cg >/dev/null 2>"$tmp/badp.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'Stage-P mate L2 marker requires high-state cg mate policy' "$tmp/badp.err" || exit 3
# No Stage R provenance is invalid.
grep -v 'b300_mainrec_stager_ilp2_pair_block_policy=' "$src" >"$tmp/nor.cu"
set +e; python3 "$GEN" "$tmp/nor.cu" "$tmp/nor-out.cu" cs >/dev/null 2>"$tmp/nor.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'Stage T requires Stage-R' "$tmp/nor.err" || exit 3
# Double transform is rejected.
set +e; python3 "$GEN" "$tmp/cs.cu" "$tmp/double.cu" cg >/dev/null 2>"$tmp/double.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'already contains Stage-T' "$tmp/double.err" || exit 3
echo 'b300-mainrec-ilp2-mate-load-policy-staget-preflight OK stage=T low_mate_policies=cg,cs high_mate_policy=M high_mate_l2=P stage_s_optional=1 stage_r_required=1 guarded_lane_preserved=1 ilp8_byte_identical=1 count_loads_unchanged=1 gpu_work=0'
