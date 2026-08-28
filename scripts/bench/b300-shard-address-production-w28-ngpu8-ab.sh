#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
RUNS="${RUNS:-1}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
ARCH="${ARCH:-native}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_shard_address_production_w28_ngpu8_ab}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

if (( RUNS < 1 )); then echo "RUNS must be >=1" >&2; exit 2; fi
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < 8 )); then echo "need 8 visible GPUs; got $visible" >&2; exit 2; fi

bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address8-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-mulhi-w28-ngpu8-proof.sh"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

BINS=()
for mode in 1 2; do
  BINS[$mode]="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_shardmode${mode}"
  N=27 SHARD_ADDRESS_MODE="$mode" ARCH="$ARCH" OUT="${BINS[$mode]}" \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-shard-address-mode.sh" \
    >"$LOGDIR/mode_${mode}.build.out" 2>"$LOGDIR/mode_${mode}.build.err"
done

printf 'mode\trun\tresidue\twall_s\tactive_max_s\tactive_sum_s\tprepare_s\n' >"$RESULT"
run_one(){
  local mode="$1" run="$2"
  local out="$LOGDIR/mode_${mode}_run${run}.out" err="$LOGDIR/mode_${mode}_run${run}.err"
  echo "=== full B300 W28x8 shard-address mode=$mode run $run/$RUNS ===" >&2
  GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" \
    "${BINS[$mode]}" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local line residue wall active_max active_sum prepare
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { tail -n 80 "$err" >&2 || true; cat "$out" >&2; exit 3; }
  residue="$(sed -nE 's/.* residue=([^[:space:]]+).*/\1/p' <<<"$line")"
  wall="$(sed -nE 's/.* wall_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  active_max="$(sed -nE 's/.* active_max_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  active_sum="$(sed -nE 's/.* active_sum_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  prepare="$(sed -nE 's/.* prepare_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$residue" && -n "$wall" && -n "$active_max" && -n "$active_sum" && -n "$prepare" ]] || { echo "$line" >&2; exit 4; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$active_max" "$active_sum" "$prepare" >>"$RESULT"
}

for ((r=1;r<=RUNS;++r)); do
  if (( r & 1 )); then order=(1 2); else order=(2 1); fi
  for mode in "${order[@]}"; do run_one "$mode" "$r"; done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
out=[]; residues={}
for mode in ('1','2'):
    rs=[r for r in rows if r['mode']==mode]
    if not rs: raise SystemExit(f'missing mode={mode}')
    rr={r['residue'] for r in rs}
    if len(rr)!=1: raise SystemExit(f'unstable residue mode={mode}: {rr}')
    residues[mode]=next(iter(rr))
    def med(k): return statistics.median(float(r[k]) for r in rs)
    out.append({'mode':mode,'runs':len(rs),'residue':residues[mode],
                'median_wall_s':f'{med("wall_s"):.9f}',
                'median_active_max_s':f'{med("active_max_s"):.9f}',
                'median_active_sum_s':f'{med("active_sum_s"):.9f}',
                'median_prepare_s':f'{med("prepare_s"):.9f}'})
if residues['1']!=residues['2']:
    raise SystemExit(f'residue mismatch compare={residues["1"]} mulhi_mask={residues["2"]}')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['mode']:r for r in out}
a=float(q['1']['median_wall_s']); b=float(q['2']['median_wall_s'])
print(f'b300_shard_address_mulhi_mask_wall_speedup={a/b:.6f}x')
print(f'b300_shard_address_mulhi_mask_wall_delta_pct={(b/a-1)*100:.4f}%')
a=float(q['1']['median_active_max_s']); b=float(q['2']['median_active_max_s'])
print(f'b300_shard_address_mulhi_mask_active_max_speedup={a/b:.6f}x')
print('mode1=three_compare_subtract_stages')
print('mode2=w28x8_mulhi_shift_plus_owner_bit_masked_base')
print(f'residue={residues["1"]}')
print(f'summary={dst}')
PY

echo "b300-shard-address-production-w28-ngpu8-ab OK runs=$RUNS mod=$MOD target_mib=$TARGET_MIB max_window=$MAX_WINDOW result=$RESULT" >&2
