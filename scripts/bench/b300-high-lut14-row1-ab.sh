#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_TARGET_MIB="${PLAN_TARGET_MIB:-16384}"
PLAN_TARGET_DIVISOR="${PLAN_TARGET_DIVISOR:-3}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${THREADS:-256}"
LOW_LUT_K="${LOW_LUT_K:-13}"
ARCH="${ARCH:-native}"
GPUS=8
RANK_DELTA_CACHE="${RANK_DELTA_CACHE:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_high_lut14_row${ROWS}_ab}"
K13_BIN="${K13_BIN:-$ONEESAN_BUILD_DIR/b300_highlut13_n27}"
K14_BIN="${K14_BIN:-$ONEESAN_BUILD_DIR/b300_highlut14_n27}"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)";((visible>=GPUS))||{ echo "need 8 GPUs, visible=$visible" >&2;exit 2; }
mkdir -p "$(dirname "$PREFIX")" "$(dirname "$K13_BIN")"

build_one(){
  local k="$1" bin="$2" log="$3"
  env N="$N" ARCH="$ARCH" OUT="$bin" LOW_LUT_K="$LOW_LUT_K" HIGH_LUT_K="$k" \
    FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 RANK_DELTA_CACHE="$RANK_DELTA_CACHE" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"${PREFIX}.${log}.build.out" 2>"${PREFIX}.${log}.build.err"
  grep -Fq "high_lut_k=$k" "${PREFIX}.${log}.build.out"
}
echo '=== build HIGH K13 ===' >&2;build_one 13 "$K13_BIN" k13
echo '=== build HIGH K14 ===' >&2;build_one 14 "$K14_BIN" k14

run_one(){
  local name="$1" bin="$2"
  local out="${PREFIX}.${name}.out" err="${PREFIX}.${name}.err" tele="${PREFIX}.${name}.gpu.csv";:>"$tele"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!
  sleep 1
  set +e
  GRIDFP_PLAN_TARGET_DIVISOR="$PLAN_TARGET_DIVISOR" GRIDFP_PLAN_TARGET_MIB="$PLAN_TARGET_MIB" GRIDFP_THREADS="$THREADS" B300_ROW_LIMIT="$ROWS" \
    "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$GPUS" >"$out" 2>"$err"
  local rc=$?;set -e;kill "$mon" 2>/dev/null||true;wait "$mon" 2>/dev/null||true
  ((rc==0))||{ echo "$name failed rc=$rc" >&2;tail -n100 "$err" >&2||true;return "$rc"; }
  local line residue wall rdg rdf entries
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$out"|tail -n1)"
  residue="$(sed -nE 's/.* residue=([^[:space:]]+).*/\1/p'<<<"$line")";wall="$(sed -nE 's/.* wall_s=([^[:space:]]+).*/\1/p'<<<"$line")"
  rdg="$(sed -nE 's/.* rank_delta_groups=([^[:space:]]+).*/\1/p'<<<"$line")";rdf="$(sed -nE 's/.* rank_delta_fallback_groups=([^[:space:]]+).*/\1/p'<<<"$line")";rdg="${rdg:-0}";rdf="${rdf:-0}"
  entries="$(sed -nE 's/.*high_lut width=28 K=[0-9]+ entries=([^[:space:]]+).*/\1/p' "$err"|tail -n1)";entries="${entries:-unknown}"
  read -r busy maxmem sm <<<"$(python3 - "$tele" <<'PY'
import csv,sys
b=[];m=[];s=[]
for r in csv.reader(open(sys.argv[1])):
    if len(r)<4:continue
    try:sv=float(r[2].strip());mv=float(r[3].strip())
    except ValueError:continue
    m.append(mv)
    if sv>=50:b.append(mv);s.append(sv)
def a(x):return sum(x)/len(x) if x else float('nan')
print(f'{a(b):.9f} {max(m) if m else float("nan"):.9f} {a(s):.9f}')
PY
)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$residue" "$wall" "$busy" "$maxmem" "$sm" "$rdg" "$rdf" "$entries"
}
read -r R13 W13 M13 X13 S13 G13 F13 E13 <<<"$(run_one k13 "$K13_BIN")"
read -r R14 W14 M14 X14 S14 G14 F14 E14 <<<"$(run_one k14 "$K14_BIN")"
[[ "$R13" == "$R14" ]]||{ echo "HIGH LUT residue mismatch k13=$R13 k14=$R14" >&2;exit 6; }
python3 - "$W13" "$W14" "$M13" "$M14" "$X13" "$X14" <<'PY'
import sys
w13,w14,m13,m14,x13,x14=map(float,sys.argv[1:])
print(f'b300_high_lut14_wall_k13_s={w13:.9f}')
print(f'b300_high_lut14_wall_k14_s={w14:.9f}')
print(f'b300_high_lut14_speedup={w13/w14:.9f}x')
print(f'b300_high_lut14_memctl_busy_k13_pct={m13:.3f}')
print(f'b300_high_lut14_memctl_busy_k14_pct={m14:.3f}')
print(f'b300_high_lut14_memctl_busy_delta_pp={m14-m13:.3f}')
print(f'b300_high_lut14_memctl_max_k13_pct={x13:.3f}')
print(f'b300_high_lut14_memctl_max_k14_pct={x14:.3f}')
PY
printf 'b300_high_lut14_exact_intermediate_match=1 residue=%s\n' "$R13"
printf 'b300_high_lut14_main_entries_k13=%s main_entries_k14=%s\n' "$E13" "$E14"
printf 'b300_high_lut14_rank_delta_groups_k13=%s fallback_k13=%s groups_k14=%s fallback_k14=%s\n' "$G13" "$F13" "$G14" "$F14"
printf 'b300_high_lut14_rows=%s plan_target_divisor=%s plan_target_mib_cap=%s rank_delta_cache=%s\n' "$ROWS" "$PLAN_TARGET_DIVISOR" "$PLAN_TARGET_MIB" "$RANK_DELTA_CACHE"
