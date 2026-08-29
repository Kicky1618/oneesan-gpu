#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

shells=(
  scripts/build/b300-forced-nextgen.sh
  scripts/build/b300-profiled-buckets-only.sh
  scripts/bench/b300-mainrec-ilp-cg-sweep.sh
  scripts/bench/b300-mainrec-ilpcg-calibrate.sh
  scripts/bench/b300-nextgen-block-combo-sweep.sh
  scripts/bench/b300-nextgen-calibrate.sh
  scripts/bench/b300-nextgen-latency-regcap-sweep.sh
  scripts/bench/b300-nextgen-calibrate-latency.sh
  scripts/run/b300x8-race-forced-set-profiled-once.sh
  scripts/run/b300x8-nextgen-select.sh
)
py=(
  scripts/build/gen-b300-main-recurrence-ilp.py
  scripts/build/gen-b300-mainrec-random-cg.py
  scripts/build/gen-b300-mainrec-prefetch-l2.py
  scripts/bench/b300-main-recurrence-ilp-partition-proof.py
)
for f in "${shells[@]}"; do p="$ONEESAN_ROOT/$f"; [[ -f "$p" ]] || { echo "missing $f" >&2; exit 2; }; bash -n "$p"; done
python3 - "$ONEESAN_ROOT" "${py[@]}" <<'PY'
import ast,pathlib,sys
root=pathlib.Path(sys.argv[1])
for rel in sys.argv[2:]:
 p=root/rel
 if not p.is_file():raise SystemExit('missing '+rel)
 ast.parse(p.read_text(),filename=str(p))
 print('AST_OK',rel)
PY
python3 "$ONEESAN_ROOT/scripts/bench/b300-main-recurrence-ilp-partition-proof.py" | grep -F 'b300-main-recurrence-ilp-partition-proof OK exact=1'

builder="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh"
selector="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-select.sh"
race="$ONEESAN_ROOT/scripts/run/b300x8-race-forced-set-profiled-once.sh"
lat="$ONEESAN_ROOT/scripts/bench/b300-nextgen-latency-regcap-sweep.sh"
for s in \
  'RECURRENCE_ILP' \
  'RANDOM_CG' \
  'PREFETCH_L2' \
  'DUALMASK' \
  'CLOSURE_BATCH' \
  'MAXRREGCOUNT' \
  'transform_order=production_recurrence,recurrence_ilp,random_cg,prefetch_l2,dualmask,closure_batch,register_cap'; do
  grep -Fq "$s" "$builder" || { echo "builder marker missing: $s" >&2; exit 3; }
done
for s in \
  'SELECT_ONLY="${SELECT_ONLY:-1}"' \
  'b300-nextgen-calibrate-latency.sh' \
  'build_candidate final' \
  'build_candidate uncapped' \
  'build_candidate main' \
  'build_candidate base' \
  'b300x8-race-forced-set-profiled-once.sh'; do
  grep -Fq "$s" "$selector" || { echo "selector marker missing: $s" >&2; exit 3; }
done
for s in \
  'EXPECTED=$((FORCED_COUNT + 2))' \
  'FATAL forced-set/profiled residue mismatch' \
  'SELECT_ONLY=1: selected' \
  'b300-profiled-buckets-only.sh'; do
  grep -Fq "$s" "$race" || { echo "race marker missing: $s" >&2; exit 3; }
done
for s in \
  'for cap in 0 $REGCAP_LIST' \
  'uncapped sync baseline did not compile' \
  'spill_store_bytes' \
  'spill_load_bytes' \
  'b300_nextgen_latency_speedup_vs_uncapped_sync'; do
  grep -Fq "$s" "$lat" || { echo "latency marker missing: $s" >&2; exit 3; }
done

echo 'b300_nextgen_preflight=OK bash_syntax=OK python_ast=OK ilp_partition=OK transform_order=OK uncapped_baseline=OK spill_gate=OK forced_set_single_pass=OK selection_default=only gpu_work=0 actions_triggered=0'
