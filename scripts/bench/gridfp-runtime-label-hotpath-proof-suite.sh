#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROOFS=(
  gridfp-experimental-shell-syntax-proof.sh
  gridfp-runtime-ab-env-proof.sh
  gridfp-runtime-owner-prefix-carry-begin-proof.sh
  gridfp-runtime-owner-local-sector-carry-begin-proof.sh
  gridfp-runtime-owner-local-sector-compact-table-proof.sh
  gridfp-runtime-owner-local-sector-w28-tree-proof.sh
  gridfp-component-support-adjacent-marks-proof.sh
  gridfp-materialize-primitive-last-r-proof.sh
  gridfp-materialize-primitive-packed-proof.sh
  gridfp-runtime-primitive-rank-packed-proof.sh
  gridfp-primitive1-u32-table-proof.sh
  gridfp-primitive-sym-u32-table-proof.sh
  gridfp-choose-sym-u32-table-proof.sh
  gridfp-codec-table-budget-proof.sh
  gridfp-support-unrank-len13-table-proof.sh
  gridfp-runtime-turn-local-sector-carry-begin-proof.sh
  gridfp-runtime-turn-local-sector-w28-tree-proof.sh
  gridfp-runtime-turn-discovery-nonn-scan-proof.sh
)

LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/gridfp_runtime_label_hotpath_proof_suite_logs}"
SUMMARY="${SUMMARY:-$ONEESAN_ROOT/work/gridfp_runtime_label_hotpath_proof_suite_summary.txt}"
mkdir -p "$LOGDIR" "$(dirname "$SUMMARY")"
: >"$SUMMARY"

for proof in "${PROOFS[@]}"; do
  path="$ONEESAN_ROOT/scripts/bench/$proof"
  [[ -f "$path" ]] || { echo "missing proof runner: $path" >&2; exit 2; }
  out="$LOGDIR/${proof%.sh}.out"
  err="$LOGDIR/${proof%.sh}.err"
  echo "=== $proof ===" >&2
  bash "$path" >"$out" 2>"$err"
  {
    echo "[$proof]"
    tail -n 2 "$out" || true
    tail -n 1 "$err" || true
    echo
  } >>"$SUMMARY"
done

cat "$SUMMARY"
echo "gridfp-runtime-label-hotpath-proof-suite OK proofs=${#PROOFS[@]} summary=$SUMMARY" >&2
