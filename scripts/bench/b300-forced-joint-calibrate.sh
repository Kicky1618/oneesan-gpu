#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}";MOD="${MOD:-4294967291}";ROWS="${ROWS:-1}";TARGET_MIB="${TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512 1024}";BATCH_LIST="${BATCH_LIST:-2 4}";HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}";REPEATS="${REPEATS:-1}"
TRANSFORM_MIN_SPEEDUP="${TRANSFORM_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_forced_joint_row${ROWS}}";SUMMARY="${SUMMARY:-${PREFIX}.tsv}"
mkdir -p "$(dirname "$PREFIX")"
printf 'high_drop\tresidue\tbase_wall_s\tbase_threads\tbest_mode\tbest_dual\tbest_batch\tbest_threads\tbest_wall_s\tspeedup_vs_base\tlog\n' >"$SUMMARY"
getv(){ local k="$1" f="$2";sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f"|tail -n1; }

for h in $HIGHDROP_LIST;do
  [[ "$h" == 0 || "$h" == 1 ]]||{ echo 'HIGHDROP_LIST supports 0 1' >&2;exit 2; }
  p="${PREFIX}.hd${h}";log="${PREFIX}.hd${h}.log"
  echo "=== forced joint high_drop=$h ===" >&2
  ARCH="$ARCH" MOD="$MOD" ROWS="$ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS_LIST="$THREADS_LIST" BATCH_LIST="$BATCH_LIST" REPEATS="$REPEATS" HIGH_DROP_CHUNK="$h" PREFIX="$p" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-block-combo-sweep.sh" | tee "$log"
  [[ "$(getv b300_mainrec_block_combo_exact_intermediate_match "$log")" == 1 ]]||{ echo "joint combo exact gate failed highdrop=$h" >&2;exit 3; }
  res="$(getv b300_mainrec_block_combo_residue "$log")";bw="$(getv b300_mainrec_block_combo_base_best_wall_s "$log")";bt="$(getv b300_mainrec_block_combo_base_best_threads "$log")"
  mode="$(getv b300_mainrec_block_combo_best_mode "$log")";dual="$(getv b300_mainrec_block_combo_best_dualmask "$log")";batch="$(getv b300_mainrec_block_combo_best_closure_batch "$log")";threads="$(getv b300_mainrec_block_combo_best_threads "$log")";wall="$(getv b300_mainrec_block_combo_best_wall_s "$log")";sp="$(getv b300_mainrec_block_combo_speedup_vs_base_best "$log")";sp="${sp%x}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$h" "$res" "$bw" "$bt" "$mode" "$dual" "$batch" "$threads" "$wall" "$sp" "$log" >>"$SUMMARY"
done

DECISION="$(python3 - "$SUMMARY" "$TRANSFORM_MIN_SPEEDUP" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));minsp=float(sys.argv[2])
if not rows:raise SystemExit('no highdrop rows')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL highdrop partial residue mismatch '+repr({r['high_drop']:r['residue'] for r in rows}))
# Global reference includes the best no-dualmask/no-batch configuration across
# highdrop and threads. This makes transform adoption independent of which
# highdrop variant happened to be run first.
base=min(rows,key=lambda r:float(r['base_wall_s']))
best=min(rows,key=lambda r:float(r['best_wall_s']))
global_speed=float(base['base_wall_s'])/float(best['best_wall_s'])
use_transform=(best['best_mode']!='base' and global_speed>=minsp)
if use_transform:
 final=(best['high_drop'],best['best_dual'],best['best_batch'],best['best_threads'],best['best_wall_s'],best['best_mode'])
else:
 final=(base['high_drop'],'0','0',base['base_threads'],base['base_wall_s'],'base')
print('\t'.join([*final,str(global_speed),next(iter(res)),base['base_wall_s'],base['high_drop'],base['base_threads'],str(int(use_transform))]))
PY
)"
IFS=$'\t' read -r FINAL_HIGH FINAL_DUAL FINAL_BATCH FINAL_THREADS FINAL_WALL FINAL_MODE GLOBAL_SPEED RESIDUE GLOBAL_BASE_WALL GLOBAL_BASE_HIGH GLOBAL_BASE_THREADS ADOPT_TRANSFORM<<<"$DECISION"
cat "$SUMMARY" >&2
printf 'b300_forced_joint_exact_gates=1\n'
printf 'b300_forced_joint_residue=%s\n' "$RESIDUE"
printf 'b300_forced_joint_global_base_wall_s=%s\n' "$GLOBAL_BASE_WALL"
printf 'b300_forced_joint_global_base_high_drop=%s\n' "$GLOBAL_BASE_HIGH"
printf 'b300_forced_joint_global_base_threads=%s\n' "$GLOBAL_BASE_THREADS"
printf 'b300_forced_joint_best_speedup_vs_global_base=%sx\n' "$GLOBAL_SPEED"
printf 'b300_forced_joint_transform_min_speedup=%sx\n' "$TRANSFORM_MIN_SPEEDUP"
printf 'b300_forced_joint_adopt_transform=%s\n' "$ADOPT_TRANSFORM"
printf 'b300_forced_joint_final_mode=%s\n' "$FINAL_MODE"
printf 'b300_forced_joint_final_high_drop_chunk=%s\n' "$FINAL_HIGH"
printf 'b300_forced_joint_final_dualmask=%s\n' "$FINAL_DUAL"
printf 'b300_forced_joint_final_closure_batch=%s\n' "$FINAL_BATCH"
printf 'b300_forced_joint_final_threads=%s\n' "$FINAL_THREADS"
printf 'b300_forced_joint_final_wall_s=%s\n' "$FINAL_WALL"
printf 'b300_forced_joint_summary=%s\n' "$SUMMARY"
printf 'b300_forced_joint_note=transform flags are adopted only if the joint winner beats the globally fastest untransformed highdrop/thread baseline by the configured margin\n'
