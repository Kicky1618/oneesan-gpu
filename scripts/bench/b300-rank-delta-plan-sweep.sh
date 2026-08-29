#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_TARGET_MIB="${PLAN_TARGET_MIB:-16384}"
DIVISORS="${DIVISORS:-2 3 4}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${THREADS:-256}"
LOW_LUT_K="${LOW_LUT_K:-13}"
HIGH_LUT_K="${HIGH_LUT_K:-13}"
ARCH="${ARCH:-native}"
GPUS=8
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rank_delta_plan_sweep_r${ROWS}}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_rank_delta_plan_sweep_n27}"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)";((visible>=GPUS))||{ echo "need 8 GPUs, visible=$visible" >&2;exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo "ROWS must be 1..28" >&2;exit 2; }
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$BIN")"

for d in $DIVISORS; do [[ "$d" =~ ^[0-9]+$ ]] && ((d>=1&&d<=16)) || { echo "invalid divisor=$d" >&2;exit 2; }; done

echo "=== build rank-delta full-pull candidate once ===" >&2
env N="$N" ARCH="$ARCH" OUT="$BIN" LOW_LUT_K="$LOW_LUT_K" HIGH_LUT_K="$HIGH_LUT_K" \
  FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 RANK_DELTA_CACHE=1 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"${PREFIX}.build.out" 2>"${PREFIX}.build.err"
grep -Fq 'rank_delta_cache=1' "${PREFIX}.build.out"
grep -Fq 'coverage_report=1' "${PREFIX}.build.out"

summary="${PREFIX}.tsv"
printf 'divisor\tresidue\twall_s\teffective_scratch_mib\tplan_target_mib\trank_delta_groups\trank_delta_fallback_groups\tmemctl_busy_avg_pct\tmemctl_max_pct\tsm_busy_avg_pct\tsamples\n' >"$summary"
ref_residue=""

for d in $DIVISORS; do
  echo "=== divisor=$d rows=$ROWS ===" >&2
  out="${PREFIX}.d${d}.out";err="${PREFIX}.d${d}.err";tele="${PREFIX}.d${d}.gpu.csv";: >"$tele"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total \
    --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & mon=$!
  sleep 1
  set +e
  GRIDFP_PLAN_TARGET_DIVISOR="$d" GRIDFP_PLAN_TARGET_MIB="$PLAN_TARGET_MIB" GRIDFP_THREADS="$THREADS" B300_ROW_LIMIT="$ROWS" \
    "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$GPUS" >"$out" 2>"$err"
  rc=$?
  set -e
  kill "$mon" 2>/dev/null || true;wait "$mon" 2>/dev/null || true
  ((rc==0))||{ echo "divisor=$d failed rc=$rc" >&2;tail -n 100 "$err" >&2||true;exit "$rc"; }
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$out" | tail -n1 || true)";[[ -n "$line" ]]||{ echo "missing backend line d=$d" >&2;exit 3; }
  field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p" <<<"$line"; }
  residue="$(field residue)";wall="$(field wall_s)";rdg="$(field rank_delta_groups)";rdf="$(field rank_delta_fallback_groups)"
  [[ -n "$rdg" ]]||rdg=0;[[ -n "$rdf" ]]||rdf=0
  eff="$(sed -nE 's/.* effective_scratch_mib=([^[:space:]]+).*/\1/p' "$err" | tail -n1)"
  plan="$(sed -nE 's/.* plan_target_mib=([^[:space:]]+).*/\1/p' "$err" | tail -n1)"
  [[ -n "$eff" && -n "$plan" ]]||{ echo "failed to parse scratch/plan d=$d" >&2;exit 4; }
  read -r busy maxmem busysm samples <<<"$(python3 - "$tele" <<'PY'
import csv,sys
busy=[];sm=[];mx=[]
with open(sys.argv[1],newline='') as f:
    for r in csv.reader(f):
        if len(r)<4:continue
        try:s=float(r[2].strip());m=float(r[3].strip())
        except ValueError:continue
        mx.append(m)
        if s>=50:busy.append(m);sm.append(s)
def avg(a):return sum(a)/len(a) if a else float('nan')
print(f'{avg(busy):.9f} {max(mx) if mx else float("nan"):.9f} {avg(sm):.9f} {len(mx)}')
PY
)"
  if [[ -z "$ref_residue" ]];then ref_residue="$residue";elif [[ "$residue" != "$ref_residue" ]];then echo "residue mismatch divisor=$d got=$residue ref=$ref_residue" >&2;exit 5;fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$residue" "$wall" "$eff" "$plan" "$rdg" "$rdf" "$busy" "$maxmem" "$busysm" "$samples" | tee -a "$summary"
done

python3 - "$summary" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
valid=[r for r in rows if int(r['rank_delta_fallback_groups'])==0]
pool=valid or rows
best=min(pool,key=lambda r:float(r['wall_s']))
print(f"b300_rank_delta_plan_best_divisor={best['divisor']}")
print(f"b300_rank_delta_plan_best_wall_s={float(best['wall_s']):.9f}")
print(f"b300_rank_delta_plan_best_memctl_busy_avg_pct={float(best['memctl_busy_avg_pct']):.3f}")
print(f"b300_rank_delta_plan_best_rank_delta_groups={best['rank_delta_groups']}")
print(f"b300_rank_delta_plan_best_rank_delta_fallback_groups={best['rank_delta_fallback_groups']}")
print(f"b300_rank_delta_plan_prefer_zero_fallback={1 if valid else 0}")
PY
printf 'b300_rank_delta_plan_exact_intermediate_match=1\n'
printf 'b300_rank_delta_plan_summary=%s\n' "$summary"
printf 'b300_rank_delta_plan_target_mib_cap=%s target_mib=%s rows=%s threads=%s low_lut_k=%s high_lut_k=%s\n' "$PLAN_TARGET_MIB" "$TARGET_MIB" "$ROWS" "$THREADS" "$LOW_LUT_K" "$HIGH_LUT_K"
