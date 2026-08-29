#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

TRANSFORM="$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-partition-proof.py"
python3 -m py_compile "$TRANSFORM" "$PROOF"

proof_out="$(python3 "$PROOF")"
printf '%s\n' "$proof_out"
grep -Fq 'b300-mainrec-hybrid-ilp8-partition-proof OK' <<<"$proof_out"
grep -Fq 'selector=n_ge_threshold' <<<"$proof_out"
grep -Fq 'exact_partition=1' <<<"$proof_out"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BASE="$TMP/base.cu"
cat >"$BASE" <<'CU'
using Code = unsigned long long;
using Count = unsigned int;
using MateID = unsigned long long;
// Synthetic recurrence artifacts required by the transform:
// b300_main_pull_prepare
// b300_low_window_cache_active
// b300_high_main_state_active
// b300_low_state_advance
// b300_high_state_advance
// high_rec_groups=
static int b300_main_pull_ilp2_blocks(Code,int){return 1;}
__global__ void main_pull_kernel_ilp2(
    const Count*,MateID*,Code,const Count*,Code,Count*,int){}
void production_launch(){
    if(useMate)main_pull_kernel_ilp2<<<b300_main_pull_ilp2_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
}

static Code rank_full(MateID m,int width){return Code(m)+Code(width);}
CU

for threshold in 0 1 257 1048576 16777216; do
  out="$TMP/hybrid_${threshold}.cu"
  log="$TMP/hybrid_${threshold}.log"
  python3 "$TRANSFORM" "$BASE" "$out" "$threshold" >"$log"
  grep -Fq 'b300_main_recurrence_hybrid_ilp8=1' "$log"
  grep -Fq "ilp8_min_states=$threshold" "$log"
  grep -Fq 'batch_abi_preserved=1' "$log"
  grep -Fq 'main_pull_kernel_ilp8_hybrid' "$out"
  grep -Fq 'b300_main_recurrence_ilp8_hybrid_blocks(ms.size,threads)' "$out"
  grep -Fq 'b300_main_pull_ilp2_blocks(ms.size,threads)' "$out"
  grep -Fq "if(ms.size>=Code($threshold))" "$out"
  grep -Fq 'base+=Code(8)*grid' "$out"
  grep -Fq 'const Code i7=' "$out"
  grep -Fq 'const Count pair7=' "$out"
  grep -Fq 'const Count block7=' "$out"
  grep -Fq 'const Count self7=' "$out"
  [[ "$(grep -Fc '__global__ void main_pull_kernel_ilp8_hybrid(' "$out")" -eq 1 ]]
  [[ "$(grep -Fc 'if(useMate){' "$out")" -eq 1 ]]
  [[ "$(grep -Fc "if(ms.size>=Code($threshold))" "$out")" -eq 1 ]]

done

set +e
python3 "$TRANSFORM" "$BASE" "$TMP/negative.cu" -1 >"$TMP/negative.out" 2>"$TMP/negative.err"
rc=$?
set -e
(( rc != 0 )) || { echo 'negative threshold unexpectedly accepted' >&2; exit 3; }
grep -Fq 'ILP8_MIN_STATES must be non-negative' "$TMP/negative.err"

# A duplicated production launch must fail rather than silently transforming an
# arbitrary first match.
DUP="$TMP/duplicate.cu"
python3 - "$BASE" "$DUP" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]).read_text()
needle='    if(useMate)main_pull_kernel_ilp2<<<b300_main_pull_ilp2_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);\n'
p=p.replace(needle,needle+needle,1)
pathlib.Path(sys.argv[2]).write_text(p)
PY
set +e
python3 "$TRANSFORM" "$DUP" "$TMP/duplicate_out.cu" 1024 >"$TMP/duplicate.out" 2>"$TMP/duplicate.err"
rc=$?
set -e
(( rc != 0 )) || { echo 'duplicate launch anchor unexpectedly accepted' >&2; exit 3; }
grep -Eq 'launch anchor expected one|enclosing launch expected one' "$TMP/duplicate.err"

# Applying the transform twice must also be rejected.
python3 "$TRANSFORM" "$BASE" "$TMP/once.cu" 1024 >/dev/null
set +e
python3 "$TRANSFORM" "$TMP/once.cu" "$TMP/twice.cu" 1024 >"$TMP/twice.out" 2>"$TMP/twice.err"
rc=$?
set -e
(( rc != 0 )) || { echo 'double hybrid transform unexpectedly accepted' >&2; exit 3; }
grep -Fq 'source already contains hybrid recurrence ILP8 kernel' "$TMP/twice.err"

echo 'b300-mainrec-hybrid-ilp8-transform-preflight OK python_ast=1 partition=1 thresholds=5 anchor_unique=1 negative_rejected=1 double_transform_rejected=1 gpu_work=0'
