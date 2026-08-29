#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

shells=(
  "$ONEESAN_ROOT/scripts/build/b300-forced-mainrec-hybrid-ilp8.sh"
  "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-transform-preflight.sh"
  "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-threshold-ab.sh"
  "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-calibrate-staged.sh"
  "$ONEESAN_ROOT/scripts/run/b300x8-mainrec-hybrid-ilp8-staged-fullprime-race.sh"
)
for s in "${shells[@]}"; do
  bash -n "$s"
done

py=(
  "$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py"
  "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-partition-proof.py"
  "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py"
)
for p in "${py[@]}"; do
  python3 - "$p" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1])
compile(p.read_text(),str(p),'exec')
PY
done

transform_out="$(bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-transform-preflight.sh")"
printf '%s\n' "$transform_out"
grep -Fq 'b300-mainrec-hybrid-ilp8-transform-preflight OK' <<<"$transform_out"
grep -Fq 'gpu_work=0' <<<"$transform_out"

builder="$ONEESAN_ROOT/scripts/build/b300-forced-mainrec-hybrid-ilp8.sh"
sweep="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-threshold-ab.sh"
staged="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-calibrate-staged.sh"
fullprime="$ONEESAN_ROOT/scripts/run/b300x8-mainrec-hybrid-ilp8-staged-fullprime-race.sh"
cg="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py"

for marker in \
  'gen-b300-main-recurrence-hybrid-ilp8.py' \
  'b300-mainrec-hybrid-ilp8-transform-preflight.sh' \
  'transform_order=production_mainrec_ilp2' \
  'batch_abi=forced2window_opt_batch' \
  'source_after_hybrid='; do
  grep -Fq "$marker" "$builder" || { echo "builder marker missing: $marker" >&2; exit 3; }
done

for marker in \
  'ILP8_THRESHOLDS must include 0' \
  'FATAL mainrec hybrid residue mismatch' \
  'no candidate has known spill-free main recurrence ptxas' \
  'expected=2 if mode==' \
  'B300_MAINREC_HYBRID_WINNER_THRESHOLD' \
  'B300_MAINREC_HYBRID_EXACT_INTERMEDIATE_MATCH=1'; do
  grep -Fq "$marker" "$sweep" || { echo "sweep marker missing: $marker" >&2; exit 3; }
done

for marker in \
  'SEARCH_ROWS="${SEARCH_ROWS:-1}"' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'MIN_SPEEDUP="${MIN_SPEEDUP:-1.01}"' \
  'B300_MAINREC_HYBRID_STAGE_VALID' \
  'search winner failed MIN_SPEEDUP' \
  'B300_MAINREC_HYBRID_STAGED_VALIDATED=1'; do
  grep -Fq "$marker" "$staged" || { echo "staged marker missing: $marker" >&2; exit 3; }
done

for marker in \
  'sha256sum -c "$MANIFEST"' \
  'B300_MAINREC_HYBRID_STAGED_VALIDATED' \
  'FORCED_OVERRIDE_BIN="$B300_MAINREC_HYBRID_SELECTED_BIN"' \
  'FORCED_BASE_BIN="$B300_MAINREC_HYBRID_SELECTED_BASE_BIN"' \
  'b300x8-race-external-forced-profiled-once.sh'; do
  grep -Fq "$marker" "$fullprime" || { echo "full-prime marker missing: $marker" >&2; exit 3; }
done

for marker in \
  "kernel_names=['main_pull_kernel_ilp2']" \
  "kernel_names.append('main_pull_kernel_ilp8_hybrid')" \
  'hybrid_policy_consistent=1'; do
  grep -Fq "$marker" "$cg" || { echo "hybrid CG marker missing: $marker" >&2; exit 3; }
done

echo 'b300_mainrec_hybrid_ilp8_preflight=OK shell_syntax=OK python_ast=OK transform_anchor=OK partition=OK spill_gate=OK staged_gate=OK fingerprint_gate=OK batch_abi=OK hybrid_cg_policy=OK gpu_work=0 actions_triggered=0'
