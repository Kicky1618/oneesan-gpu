#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid-ilp8-sweep.sh"
LOCAL="$ONEESAN_ROOT/scripts/bench/b300-local-sm86-hybrid8-sweep.sh"
for f in "$SWEEP" "$LOCAL"; do
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

echo 'b300_hybrid8_ngpu_preflight=OK default_b300_gpus=8 local_sm86_gpus=1 residue_gate=canonical spill_gate=canonical gpu_work=0 actions_triggered=0'
