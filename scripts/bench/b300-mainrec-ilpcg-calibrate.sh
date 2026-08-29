#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"; HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}"; REPEATS="${REPEATS:-1}"; TRANSFORM_MIN_SPEEDUP="${TRANSFORM_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_ilpcg_calibrate_row${ROWS}}"; SUMMARY="${SUMMARY:-${PREFIX}.tsv}"
mkdir -p "$(dirname "$PREFIX")"
printf 'high_drop\tresidue\tbase_threads\tbase_wall_s\tbase_mc_avg_pct\tbest_mode\tbest_threads\tbest_wall_s\tbest_mc_avg_pct\tspeedup_vs_local_base\tlog\n' >"$SUMMARY"
getv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }

for h in $HIGHDROP_LIST; do
  [[ "$h" == 0 || "$h" == 1 ]] || { echo 'HIGHDROP_LIST supports only 0 1' >&2; exit 2; }
  log="${PREFIX}.hd${h}.log"; p="${PREFIX}.hd${h}"
  echo "=== mainrec ILP/CG calibration highdrop=$h ===" >&2
  ARCH="$ARCH" MOD="$MOD" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" HIGH_DROP_CHUNK="$h" THREADS_LIST="$THREADS_LIST" REPEATS="$REPEATS" PREFIX="$p" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp-cg-sweep.sh" | tee "$log"
  [[ "$(getv b300_mainrec_ilpcg_exact_intermediate_match "$log")" == 1 ]] || { echo "ILP/CG exact gate missing highdrop=$h" >&2; exit 3; }
  res="$(getv b300_mainrec_ilpcg_residue "$log")"; bt="$(getv b300_mainrec_ilpcg_base_best_threads "$log")"; bw="$(getv b300_mainrec_ilpcg_base_best_wall_s "$log")"; bmc="$(getv b300_mainrec_ilpcg_base_best_mc_avg_pct "$log")"
  mode="$(getv b300_mainrec_ilpcg_best_mode "$log")"; t="$(getv b300_mainrec_ilpcg_best_threads "$log")"; w="$(getv b300_mainrec_ilpcg_best_wall_s "$log")"; mc="$(getv b300_mainrec_ilpcg_best_mc_avg_pct "$log")"; sp="$(getv b300_mainrec_ilpcg_speedup_vs_ilp2 "$log")"; sp="${sp%x}"
  [[ "$mode" =~ ^ilp(2|4|8)(cg)?$ ]] || { echo "bad best mode=$mode" >&2; exit 3; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$h" "$res" "$bt" "$bw" "$bmc" "$mode" "$t" "$w" "$mc" "$sp" "$log" >>"$SUMMARY"
done

DECISION="$(python3 - "$SUMMARY" "$TRANSFORM_MIN_SPEEDUP" <<'PY'
import csv,re,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); minsp=float(sys.argv[2])
if not rows:raise SystemExit('no ILP/CG calibration rows')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL ILP/CG highdrop partial residue mismatch '+repr({r['high_drop']:r['residue'] for r in rows}))
base=min(rows,key=lambda r:float(r['base_wall_s']))
best=min(rows,key=lambda r:(float(r['best_wall_s']),-float(r['best_mc_avg_pct']) if r['best_mc_avg_pct']!='nan' else 1e100))
global_speed=float(base['base_wall_s'])/float(best['best_wall_s'])
transformed=best['best_mode']!='ilp2'
adopt=transformed and global_speed>=minsp
chosen=best if adopt else base
if adopt:
 mode=best['best_mode']; threads=best['best_threads']; wall=best['best_wall_s']; mc=best['best_mc_avg_pct']; hd=best['high_drop']
else:
 mode='ilp2'; threads=base['base_threads']; wall=base['base_wall_s']; mc=base['base_mc_avg_pct']; hd=base['high_drop']
m=re.fullmatch(r'ilp(2|4|8)(cg)?',mode)
if not m:raise SystemExit('invalid selected mode '+mode)
lanes=m.group(1); cg='1' if m.group(2) else '0'
print('\t'.join([hd,mode,lanes,cg,threads,wall,mc,str(global_speed),str(int(adopt)),next(iter(res)),base['high_drop'],base['base_threads'],base['base_wall_s'],base['base_mc_avg_pct']]))
PY
)"
IFS=$'\t' read -r FINAL_HIGH FINAL_MODE FINAL_ILP FINAL_CG FINAL_THREADS FINAL_WALL FINAL_MC GLOBAL_SPEED ADOPT RESIDUE BASE_HIGH BASE_THREADS BASE_WALL BASE_MC <<<"$DECISION"
cat "$SUMMARY" >&2
printf 'b300_mainrec_ilpcg_calibrate_exact_gates=1\n'
printf 'b300_mainrec_ilpcg_calibrate_residue=%s\n' "$RESIDUE"
printf 'b300_mainrec_ilpcg_calibrate_global_base_high_drop=%s\n' "$BASE_HIGH"
printf 'b300_mainrec_ilpcg_calibrate_global_base_threads=%s\n' "$BASE_THREADS"
printf 'b300_mainrec_ilpcg_calibrate_global_base_wall_s=%s\n' "$BASE_WALL"
printf 'b300_mainrec_ilpcg_calibrate_global_base_mc_avg_pct=%s\n' "$BASE_MC"
printf 'b300_mainrec_ilpcg_calibrate_best_speedup_vs_global_base=%sx\n' "$GLOBAL_SPEED"
printf 'b300_mainrec_ilpcg_calibrate_transform_min_speedup=%sx\n' "$TRANSFORM_MIN_SPEEDUP"
printf 'b300_mainrec_ilpcg_calibrate_adopt_transform=%s\n' "$ADOPT"
printf 'b300_mainrec_ilpcg_calibrate_final_high_drop_chunk=%s\n' "$FINAL_HIGH"
printf 'b300_mainrec_ilpcg_calibrate_final_mode=%s\n' "$FINAL_MODE"
printf 'b300_mainrec_ilpcg_calibrate_final_ilp=%s\n' "$FINAL_ILP"
printf 'b300_mainrec_ilpcg_calibrate_final_random_cg=%s\n' "$FINAL_CG"
printf 'b300_mainrec_ilpcg_calibrate_final_threads=%s\n' "$FINAL_THREADS"
printf 'b300_mainrec_ilpcg_calibrate_final_wall_s=%s\n' "$FINAL_WALL"
printf 'b300_mainrec_ilpcg_calibrate_final_mc_avg_pct=%s\n' "$FINAL_MC"
printf 'b300_mainrec_ilpcg_calibrate_summary=%s\n' "$SUMMARY"
printf 'b300_mainrec_ilpcg_calibrate_note=wall_time selects; MC is diagnostic/tie context only; transformed main path requires configured speedup margin over globally fastest ILP2 baseline\n'
