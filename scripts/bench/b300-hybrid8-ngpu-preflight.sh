#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid-ilp8-sweep.sh"
LOCAL="$ONEESAN_ROOT/scripts/bench/b300-local-sm86-hybrid8-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-local-sm86-hybrid8-staged.sh"
for f in "$SWEEP" "$LOCAL" "$STAGED"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 2; }
  bash -n "$f"
done

for s in \
  'NGPU="${NGPU:-8}"' \
  '[[ "$NGPU" =~ ^[1-8]$ ]]' \
  'VISIBLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"' \
  '(( VISIBLE_GPUS >= NGPU ))' \
  'head -n "$NGPU"' \
  '"$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD"' \
  "('B300_HYBRID8_NGPU',ngpu)" \
  'b300_nextgen_hybrid8_ngpu='; do
  grep -Fq "$s" "$SWEEP" || { echo "portable sweep marker missing: $s" >&2; exit 3; }
done

for s in \
  'ARCH="${ARCH:-sm_86}"' \
  'NGPU="${NGPU:-1}"' \
  'TARGET_MIB="${TARGET_MIB:-1024}"' \
  'ILP8_THRESHOLDS="${ILP8_THRESHOLDS:-0 262144 1048576 4194304}"' \
  'b300-nextgen-hybrid-ilp8-sweep.sh'; do
  grep -Fq "$s" "$LOCAL" || { echo "local sweep marker missing: $s" >&2; exit 3; }
done

for s in \
  'SEARCH_ROWS="${SEARCH_ROWS:-1}"' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'SEARCH_THRESHOLDS="${SEARCH_THRESHOLDS:-0 262144 1048576 4194304}"' \
  'VALIDATE_THRESHOLD="${VALIDATE_THRESHOLD:-auto}"' \
  'FORCE_ILP8_ROWS="${FORCE_ILP8_ROWS:-8}"' \
  'SEARCH_THRESHOLD="$B300_HYBRID8_WINNER_THRESHOLD"' \
  'SELECTED_THRESHOLD="$SEARCH_THRESHOLD"' \
  'run_stage "$rows" "$SELECTED_THRESHOLD"' \
  'run_stage "$FORCE_ILP8_ROWS" 0 forced_ilp8' \
  'b300_nextgen_hybrid8_exact_intermediate_match=1' \
  'b300_nextgen_hybrid8_ngpu=$NGPU' \
  'selected_validate_threshold=$SELECTED_THRESHOLD' \
  'forced_ilp8_rows=$FORCE_ILP8_ROWS' \
  'b300_local_sm86_hybrid8_staged=OK'; do
  grep -Fq "$s" "$STAGED" || { echo "local staged marker missing: $s" >&2; exit 3; }
done

echo 'b300_hybrid8_ngpu_preflight=OK default_b300_gpus=8 local_sm86_gpus=1 residue_gate=canonical spill_gate=canonical staged_rows=1,4,8 selected_threshold_gate=1 forced_ilp8_deep_gate=1 gpu_work=0 actions_triggered=0'
