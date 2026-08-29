#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_TARGET_DIVISOR="${PLAN_TARGET_DIVISOR:-3}"
PLAN_TARGET_MIB="${PLAN_TARGET_MIB:-}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${THREADS:-256}"
LOW_LUT_K="${LOW_LUT_K:-13}"
HIGH_LUT_K="${HIGH_LUT_K:-13}"
ARCH="${ARCH:-native}"
GPUS=8
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rank_delta_cache_row${ROWS}_d${PLAN_TARGET_DIVISOR}_ab}"
BASE_BIN="${BASE_BIN:-$ONEESAN_BUILD_DIR/b300_rank_delta_cache0_n27}"
CAND_BIN="${CAND_BIN:-$ONEESAN_BUILD_DIR/b300_rank_delta_cache1_n27}"

[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo "ROWS must be 1..28" >&2; exit 2; }
[[ "$PLAN_TARGET_DIVISOR" =~ ^[0-9]+$ ]] && ((PLAN_TARGET_DIVISOR>=1&&PLAN_TARGET_DIVISOR<=16)) || { echo "PLAN_TARGET_DIVISOR must be 1..16" >&2; exit 2; }
if [[ -n "$PLAN_TARGET_MIB" ]]; then [[ "$PLAN_TARGET_MIB" =~ ^[0-9]+$ ]] && ((PLAN_TARGET_MIB>=1)) || { echo "PLAN_TARGET_MIB must be a positive integer when set" >&2; exit 2; }; fi
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)";((visible>=GPUS))||{ echo "need 8 GPUs, visible=$visible" >&2;exit 2; }
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$BASE_BIN")"

common_build=(FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 PTXAS_VERBOSE=1)
echo "=== build rank-delta baseline ===" >&2
env N="$N" ARCH="$ARCH" OUT="$BASE_BIN" LOW_LUT_K="$LOW_LUT_K" HIGH_LUT_K="$HIGH_LUT_K" RANK_DELTA_CACHE=0 "${common_build[@]}" \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"${PREFIX}.base.build.out" 2>"${PREFIX}.base.build.err"
echo "=== build rank-delta candidate ===" >&2
env N="$N" ARCH="$ARCH" OUT="$CAND_BIN" LOW_LUT_K="$LOW_LUT_K" HIGH_LUT_K="$HIGH_LUT_K" RANK_DELTA_CACHE=1 "${common_build[@]}" \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"${PREFIX}.cand.build.out" 2>"${PREFIX}.cand.build.err"
grep -Fq 'rank_delta_cache=0' "${PREFIX}.base.build.out"
grep -Fq 'rank_delta_cache=1' "${PREFIX}.cand.build.out"
grep -Fq 'prefix_rank_walk_removed=main_drop,block_lift' "${PREFIX}.cand.build.out"

run_one(){
  local name="$1" bin="$2"
  local out="${PREFIX}.${name}.out" err="${PREFIX}.${name}.err" tele="${PREFIX}.${name}.gpu.csv"
  : >"$tele"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total \
    --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!
  sleep 1
  local runenv=(GRIDFP_PLAN_TARGET_DIVISOR="$PLAN_TARGET_DIVISOR" GRIDFP_THREADS="$THREADS" B300_ROW_LIMIT="$ROWS")
  [[ -n "$PLAN_TARGET_MIB" ]] && runenv+=(GRIDFP_PLAN_TARGET_MIB="$PLAN_TARGET_MIB")
  set +e
  env "${runenv[@]}" "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$GPUS" >"$out" 2>"$err"
  local rc=$?
  set -e
  kill "$mon" 2>/dev/null || true;wait "$mon" 2>/dev/null || true
  ((rc==0))||{ echo "$name failed rc=$rc" >&2;tail -n 100 "$err" >&2||true;return "$rc"; }
  local line residue wall rdg rdf plan effective
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]]||{ echo "$name missing backend result" >&2;return 4; }
  residue="$(sed -nE 's/.* residue=([^[:space:]]+).*/\1/p' <<<"$line")"
  wall="$(sed -nE 's/.* wall_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  rdg="$(sed -nE 's/.* rank_delta_groups=([^[:space:]]+).*/\1/p' <<<"$line")";rdg="${rdg:-0}"
  rdf="$(sed -nE 's/.* rank_delta_fallback_groups=([^[:space:]]+).*/\1/p' <<<"$line")";rdf="${rdf:-0}"
  plan="$(sed -nE 's/.* plan_target_mib=([^[:space:]]+).*/\1/p' "$err" | tail -n1)";plan="${plan:-unknown}"
  effective="$(sed -nE 's/.* effective_scratch_mib=([^[:space:]]+).*/\1/p' "$err" | tail -n1)";effective="${effective:-unknown}"
  local stats
  stats="$(python3 - "$tele" <<'PY'
import csv,sys
mem=[];busy=[];sm=[]
with open(sys.argv[1],newline='') as f:
    for r in csv.reader(f):
        if len(r)<4:continue
        try:s=float(r[2].strip());m=float(r[3].strip())
        except ValueError:continue
        sm.append(s);mem.append(m)
        if s>=50:busy.append(m)
def avg(x):return sum(x)/len(x) if x else float('nan')
print(f'{avg(mem):.9f} {max(mem) if mem else float("nan"):.9f} {avg(busy):.9f} {avg(sm):.9f} {len(mem)} {len(busy)}')
PY
)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$residue" "$wall" "$rdg" "$rdf" "$plan" "$effective" "$stats"
}

echo "=== run baseline rows=$ROWS divisor=$PLAN_TARGET_DIVISOR ===" >&2
read -r BR BW BRDG BRDF BPLAN BEFF BMEM BMAX BBUSY BSM BSAMP BBUSYSAMP <<<"$(run_one base "$BASE_BIN")"
echo "=== run rank-delta candidate rows=$ROWS divisor=$PLAN_TARGET_DIVISOR ===" >&2
read -r CR CW CRDG CRDF CPLAN CEFF CMEM CMAX CBUSY CSM CSAMP CBUSYSAMP <<<"$(run_one cand "$CAND_BIN")"
[[ "$BR" == "$CR" ]]||{ echo "rank-delta residue mismatch baseline=$BR candidate=$CR" >&2;exit 6; }
[[ "$BPLAN" == "$CPLAN" && "$BEFF" == "$CEFF" ]] || { echo "baseline/candidate planner mismatch base_plan=$BPLAN cand_plan=$CPLAN base_eff=$BEFF cand_eff=$CEFF" >&2;exit 7; }
python3 - "$BW" "$CW" "$BMEM" "$CMEM" "$BMAX" "$CMAX" "$BBUSY" "$CBUSY" "$BSM" "$CSM" <<'PY'
import sys
bw,cw,bm,cm,bx,cx,bb,cb,bs,cs=map(float,sys.argv[1:])
print(f'b300_rank_delta_wall_baseline_s={bw:.9f}')
print(f'b300_rank_delta_wall_candidate_s={cw:.9f}')
print(f'b300_rank_delta_wall_speedup={bw/cw:.9f}x')
print(f'b300_rank_delta_wall_delta_pct={(cw/bw-1)*100:.4f}%')
print(f'b300_rank_delta_memctl_all_baseline_avg_pct={bm:.3f}')
print(f'b300_rank_delta_memctl_all_candidate_avg_pct={cm:.3f}')
print(f'b300_rank_delta_memctl_busy_baseline_avg_pct={bb:.3f}')
print(f'b300_rank_delta_memctl_busy_candidate_avg_pct={cb:.3f}')
print(f'b300_rank_delta_memctl_busy_delta_pp={cb-bb:.3f}')
print(f'b300_rank_delta_memctl_baseline_max_pct={bx:.3f}')
print(f'b300_rank_delta_memctl_candidate_max_pct={cx:.3f}')
print(f'b300_rank_delta_sm_baseline_avg_pct={bs:.3f}')
print(f'b300_rank_delta_sm_candidate_avg_pct={cs:.3f}')
PY
printf 'b300_rank_delta_rows=%s\n' "$ROWS"
printf 'b300_rank_delta_residue=%s\n' "$BR"
printf 'b300_rank_delta_exact_intermediate_match=1\n'
printf 'b300_rank_delta_groups=%s\n' "$CRDG"
printf 'b300_rank_delta_fallback_groups=%s\n' "$CRDF"
printf 'b300_rank_delta_effective_scratch_mib=%s\n' "$CEFF"
printf 'b300_rank_delta_actual_plan_target_mib=%s\n' "$CPLAN"
printf 'b300_rank_delta_plan_target_divisor=%s\n' "$PLAN_TARGET_DIVISOR"
printf 'b300_rank_delta_baseline_samples=%s busy_samples=%s\n' "$BSAMP" "$BBUSYSAMP"
printf 'b300_rank_delta_candidate_samples=%s busy_samples=%s\n' "$CSAMP" "$CBUSYSAMP"
printf 'b300_rank_delta_target_mib=%s plan_target_mib_cap=%s max_window=%s threads=%s low_lut_k=%s high_lut_k=%s\n' "$TARGET_MIB" "${PLAN_TARGET_MIB:-none}" "$MAX_WINDOW" "$THREADS" "$LOW_LUT_K" "$HIGH_LUT_K"
printf 'b300_rank_delta_base_stdout=%s\n' "${PREFIX}.base.out"
printf 'b300_rank_delta_cand_stdout=%s\n' "${PREFIX}.cand.out"
