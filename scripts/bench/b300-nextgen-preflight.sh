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
  scripts/bench/b300-nextgen-cg-l2size-sweep.sh
  scripts/bench/b300-nextgen-calibrate-cgl2.sh
  scripts/bench/b300-nextgen-hybrid-ilp8-sweep.sh
  scripts/bench/b300-nextgen-hybrid8-staged-calibrate.sh
  scripts/bench/b300-nextgen-hybrid8-nextself-width-sweep.sh
  scripts/bench/b300-nextgen-hybrid8-nextself-geometry-sweep.sh
  scripts/bench/b300-nextgen-hybrid8-nextself-staged-calibrate.sh
  scripts/bench/b300-mainrec-hybrid-ilp8-transform-preflight.sh
  scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh
  scripts/run/b300x8-nextgen-hybrid8-nextself-staged-fullprime-race.sh
  scripts/run/b300x8-race-forced-set-profiled-once.sh
  scripts/run/b300x8-nextgen-select.sh
)
py=(
  scripts/build/gen-b300-main-recurrence-ilp.py
  scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py
  scripts/build/gen-b300-mainrec-random-cg.py
  scripts/build/gen-b300-mainrec-prefetch-l2.py
  scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py
  scripts/bench/b300-main-recurrence-ilp-partition-proof.py
  scripts/bench/b300-mainrec-hybrid-ilp8-partition-proof.py
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
python3 "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-partition-proof.py" | grep -F 'b300-mainrec-hybrid-ilp8-partition-proof OK'
bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-transform-preflight.sh" | grep -F 'b300-mainrec-hybrid-ilp8-transform-preflight OK'

builder="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh"
randomcg="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py"
prefetch="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-prefetch-l2.py"
hybrid_nextself="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
selector="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-select.sh"
race="$ONEESAN_ROOT/scripts/run/b300x8-race-forced-set-profiled-once.sh"
lat="$ONEESAN_ROOT/scripts/bench/b300-nextgen-latency-regcap-sweep.sh"
cgl2="$ONEESAN_ROOT/scripts/bench/b300-nextgen-cg-l2size-sweep.sh"
staged="$ONEESAN_ROOT/scripts/bench/b300-nextgen-calibrate-cgl2.sh"
hybrid8="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid-ilp8-sweep.sh"
hybrid8_staged="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-staged-calibrate.sh"
hybrid8_race="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staged-fullprime-race.sh"
hybrid8_ns_width="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-width-sweep.sh"
hybrid8_ns_geometry="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-geometry-sweep.sh"
hybrid8_ns_staged="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-staged-calibrate.sh"
hybrid8_ns_race="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-nextself-staged-fullprime-race.sh"
for s in \
  'RECURRENCE_ILP' \
  'RECURRENCE_HYBRID_ILP8' \
  'RECURRENCE_HYBRID_ILP8_MIN_STATES' \
  'RECURRENCE_HYBRID_ILP8_NEXTSELF' \
  'RECURRENCE_HYBRID_ILP8_NEXTSELF_WIDTH' \
  'gen-b300-main-recurrence-hybrid-ilp8.py' \
  'gen-b300-mainrec-hybrid8-next-self-prefetch.py' \
  'RANDOM_CG_L2_FETCH_BYTES' \
  'PREFETCH_L2' \
  'DUALMASK' \
  'CLOSURE_BATCH' \
  'MAXRREGCOUNT' \
  'transform_order=production_recurrence,recurrence_ilp,recurrence_hybrid_ilp8,random_cg_l2_fetch,prefetch_l2,hybrid8_nextself,dualmask,closure_batch,register_cap'; do
  grep -Fq "$s" "$builder" || { echo "builder marker missing: $s" >&2; exit 3; }
done
for f in "$randomcg" "$prefetch"; do
  for s in 'main_pull_kernel_ilp8_hybrid' 'hybrid_policy_consistent=1'; do
    grep -Fq "$s" "$f" || { echo "hybrid cache-policy transform marker missing file=$f marker=$s" >&2; exit 3; }
  done
done
for s in \
  'b300_mainrec_hybrid8_next_self_prefetch=1' \
  'WIDTH must be one of 1,2,4,8' \
  'DISTANCE must be one of 1,2,4' \
  'prefetch_width=' \
  'prefetch_distance_iterations=' \
  'main_pull_kernel_ilp8_hybrid' \
  'b300_mainrec_hybrid8_prefetch_next_self_l2'; do
  grep -Fq "$s" "$hybrid_nextself" || { echo "hybrid next-self transform marker missing: $s" >&2; exit 3; }
done
for s in \
  'BASE_RECURRENCE_ILP' \
  'ILP8_THRESHOLDS' \
  'RECURRENCE_HYBRID_ILP8="$hybrid"' \
  'RECURRENCE_HYBRID_ILP8_MIN_STATES="$threshold"' \
  'main_pull_kernel_ilp8_hybrid' \
  'B300_HYBRID8_WINNER_THRESHOLD' \
  'B300_HYBRID8_WINNER_SPEEDUP_VS_BASELINE' \
  'baseline lacks known spill-free recurrence ptxas' \
  'resource_ok=(len(rv)>=2 if mode=='"'"'hybrid'"'"' else len(rv)>=1)' \
  'b300_nextgen_hybrid8_exact_intermediate_match=1'; do
  grep -Fq "$s" "$hybrid8" || { echo "hybrid ILP8 sweep marker missing: $s" >&2; exit 3; }
done
for s in \
  'CANONICAL_SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid-ilp8-sweep.sh"' \
  'HYBRID_ILP8_THRESHOLDS' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'ILP8_THRESHOLDS="$thresholds"' \
  'B300_HYBRID8_WINNER_SPILL_FREE=1' \
  'B300_HYBRID8_FINAL_SPILL_FREE' \
  'if [[ "$rows" == "$SEARCH_ROWS" && "$stage_res" != "$CORE_RES" ]]' \
  'B300_HYBRID8_FINAL_STAGE_ROWS' \
  'B300_HYBRID8_FINAL_STAGE_RESIDUE' \
  'row_scoped_residue_gate=1' \
  'canonical_sweep=1'; do
  grep -Fq "$s" "$hybrid8_staged" || { echo "staged hybrid8 marker missing: $s" >&2; exit 3; }
done
for s in \
  'B300_HYBRID8_STAGED_VALIDATED' \
  'B300_HYBRID8_FINAL_ENABLED' \
  'B300_HYBRID8_FINAL_SPILL_FREE' \
  'B300_HYBRID8_FINAL_STAGE_ROWS' \
  'B300_HYBRID8_FINAL_STAGE_RESIDUE' \
  'MANIFEST="${MANIFEST:-${WINNER_ENV%.env}_promotion-inputs.sha256}"' \
  'sha256sum -c "$MANIFEST"' \
  'B300_HYBRID8_PROMOTION_SPILL_FREE=1' \
  'B300_HYBRID8_PROMOTION_FINAL_STAGE_RESIDUE' \
  'FORCED_OVERRIDE_BIN="$B300_HYBRID8_FINAL_BIN"' \
  'FORCED_BASE_BIN="$B300_HYBRID8_BASE_BIN"' \
  'b300x8-race-external-forced-profiled-once.sh'; do
  grep -Fq "$s" "$hybrid8_race" || { echo "hybrid8 full-prime marker missing: $s" >&2; exit 3; }
done
# Keep the one-dimensional width sweep as a diagnostic and regression tool.
for s in \
  'WIDTH_LIST="${WIDTH_LIST:-1 2 4 8}"' \
  'RECURRENCE_HYBRID_ILP8_NEXTSELF_WIDTH="$width"' \
  'B300_HYBRID8_NEXTSELF_WIDTH' \
  'B300_HYBRID8_NEXTSELF_BEST_WIDTH' \
  'resource_ok=len(rv)>=2' \
  'b300_nextgen_hybrid8_nextself_exact_intermediate_match=1' \
  'b300_nextgen_hybrid8_nextself_width_sweep=1'; do
  grep -Fq "$s" "$hybrid8_ns_width" || { echo "hybrid8 next-self width-sweep marker missing: $s" >&2; exit 3; }
done
for s in \
  'WIDTH_LIST="${WIDTH_LIST:-1 2 4 8}"' \
  'DISTANCE_LIST="${DISTANCE_LIST:-1 2 4}"' \
  'Every geometry candidate is derived' \
  'B300_HYBRID8_NEXTSELF_WIDTH' \
  'B300_HYBRID8_NEXTSELF_DISTANCE' \
  'B300_HYBRID8_NEXTSELF_BEST_WIDTH' \
  'B300_HYBRID8_NEXTSELF_BEST_DISTANCE' \
  'geometry control lacks known spill-free ILP2/ILP8 ptxas' \
  'b300_nextgen_hybrid8_nextself_geometry_sweep=1'; do
  grep -Fq "$s" "$hybrid8_ns_geometry" || { echo "hybrid8 next-self geometry marker missing: $s" >&2; exit 3; }
done
for s in \
  'WIDTH_LIST="${WIDTH_LIST:-1 2 4 8}"' \
  'DISTANCE_LIST="${DISTANCE_LIST:-1 2 4}"' \
  'b300-nextgen-hybrid8-nextself-geometry-sweep.sh' \
  'SELECTED_WIDTH=0' \
  'SELECTED_DISTANCE=0' \
  'FATAL Stage-F geometry changed during validation' \
  'B300_HYBRID8_NEXTSELF_FINAL_WIDTH' \
  'B300_HYBRID8_NEXTSELF_FINAL_DISTANCE' \
  'B300_HYBRID8_NEXTSELF_SEARCH_WIDTHS' \
  'B300_HYBRID8_NEXTSELF_SEARCH_DISTANCES' \
  'geometry_locked=1'; do
  grep -Fq "$s" "$hybrid8_ns_staged" || { echo "hybrid8 next-self staged-geometry marker missing: $s" >&2; exit 3; }
done
for s in \
  'B300_HYBRID8_NEXTSELF_FINAL_WIDTH' \
  'B300_HYBRID8_NEXTSELF_FINAL_DISTANCE' \
  'B300_HYBRID8_NEXTSELF_PROMOTION_WIDTH' \
  'B300_HYBRID8_NEXTSELF_PROMOTION_DISTANCE' \
  'B300_HYBRID8_NEXTSELF_PREPARED_WIDTH' \
  'B300_HYBRID8_NEXTSELF_PREPARED_DISTANCE' \
  'nextgen_hybrid8_nextself_w${B300_HYBRID8_NEXTSELF_FINAL_WIDTH}_d${B300_HYBRID8_NEXTSELF_FINAL_DISTANCE}_t${B300_HYBRID8_NEXTSELF_THRESHOLD}'; do
  grep -Fq "$s" "$hybrid8_ns_race" || { echo "hybrid8 next-self geometry promotion marker missing: $s" >&2; exit 3; }
done
for s in \
  'SELECT_ONLY="${SELECT_ONLY:-1}"' \
  'b300-nextgen-calibrate-cgl2.sh' \
  'build_candidate final' \
  'build_candidate stagec' \
  'build_candidate uncapped' \
  'build_candidate main' \
  'build_candidate base' \
  'RANDOM_CG_L2_FETCH_BYTES="$cgl2"' \
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
for s in \
  'RANDOM_CG_L2_FETCH_BYTES="$l2"' \
  'L2_SIZES="${L2_SIZES:-0 64 128 256}"' \
  'no CG L2 candidate has known spill-free main recurrence ptxas' \
  'B300_CGL2_WINNER_L2_FETCH_BYTES' \
  'b300_nextgen_cgl2_exact_intermediate_match=1'; do
  grep -Fq "$s" "$cgl2" || { echo "CG-L2 sweep marker missing: $s" >&2; exit 3; }
done
for s in \
  'b300-nextgen-calibrate-latency.sh' \
  'b300-nextgen-cg-l2size-sweep.sh' \
  'CGL2_MIN_SPEEDUP' \
  'b300_nextgen_cgl2_calibrate_exact_gates=1' \
  'complete-prime arbitration retains Stage-C candidate'; do
  grep -Fq "$s" "$staged" || { echo "Stage-D calibration marker missing: $s" >&2; exit 3; }
done

echo 'b300_nextgen_preflight=OK bash_syntax=OK python_ast=OK ilp_partition=OK hybrid_partition=OK hybrid_transform=OK transform_order=OK hybrid_ilp8_builder=OK hybrid_cache_policy=OK hybrid8_nextself=OK hybrid8_nextself_width=OK hybrid8_nextself_geometry=OK hybrid_ilp8_sweep=OK hybrid8_staged=OK row_scoped_residue=OK hybrid8_fullprime_gate=OK fingerprint_gate=OK uncapped_baseline=OK spill_gate=OK cgl2_stage_d=OK forced_set_single_pass=OK selection_default=only gpu_work=0 actions_triggered=0'
