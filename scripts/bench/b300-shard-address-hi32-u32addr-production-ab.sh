#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}"; ARCH="${ARCH:-native}"; PTX_ARCH="${PTX_ARCH:-sm_80}"; RUNS="${RUNS:-1}"
MOD="${MOD:-4294967291}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
command -v "$NVCC" >/dev/null || exit 2; command -v nvidia-smi >/dev/null || exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; ((visible>=8)) || { echo "need 8 visible GPUs" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-u32addr-w28-ngpu8-proof.sh"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-shard-address-hi32-u32addr-production-ptx-proof.sh"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hi32_u32addr_production_ab}"; RESULT="${RESULT:-${PREFIX}.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"; GEN="$ONEESAN_ROOT/scripts/build/gen-b300-shard-address-hi32-u32addr.py"; GENSRC="$LOGDIR/generated.cu"; python3 "$GEN" "$SRC" "$GENSRC" >"$LOGDIR/generate.out"
BINS=()
for m in 1 5 6; do BINS[$m]="$ONEESAN_BUILD_DIR/b300_hi32_u32addr_prod_mode${m}"; TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=0 -DB300_SHARD_ADDRESS_MODE="$m" "$GENSRC" -o "${BINS[$m]}" >"$LOGDIR/m${m}.build.out" 2>"$LOGDIR/m${m}.build.err"; done
printf 'mode\trun\tresidue\twall_s\tactive_max_s\tactive_sum_s\tprepare_s\n' >"$RESULT"
run_one(){ local m="$1" r="$2" o="$LOGDIR/m${m}_r${r}.out" e="$LOGDIR/m${m}_r${r}.err" line; GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" "${BINS[$m]}" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$o" 2>"$e"; line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$o"|tail -n1)"; [[ -n "$line" ]]||{ tail -n80 "$e" >&2; exit 3; }; printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$r" "$(sed -nE 's/.* residue=([^ ]+).*/\1/p'<<<"$line")" "$(sed -nE 's/.* wall_s=([^ ]+).*/\1/p'<<<"$line")" "$(sed -nE 's/.* active_max_s=([^ ]+).*/\1/p'<<<"$line")" "$(sed -nE 's/.* active_sum_s=([^ ]+).*/\1/p'<<<"$line")" "$(sed -nE 's/.* prepare_s=([^ ]+).*/\1/p'<<<"$line")" >>"$RESULT"; }
for((r=1;r<=RUNS;++r));do case $(((r-1)%3)) in 0)order=(1 5 6);;1)order=(5 6 1);;*)order=(6 1 5);;esac;for m in "${order[@]}";do echo "=== B300 production shard mode=$m run $r/$RUNS ===" >&2;run_one "$m" "$r";done;done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
R=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); q={};res={}
for m in ('1','5','6'):
 x=[r for r in R if r['mode']==m]; rr={r['residue'] for r in x}; assert len(rr)==1; res[m]=next(iter(rr)); q[m]={k:statistics.median(float(r[k]) for r in x) for k in ('wall_s','active_max_s','active_sum_s')}
assert len(set(res.values()))==1
for m,n in [('5','hi32_u64corr'),('6','hi32_full_u32')]:
 print(f'b300_{n}_wall_speedup={q["1"]["wall_s"]/q[m]["wall_s"]:.6f}x'); print(f'b300_{n}_active_max_speedup={q["1"]["active_max_s"]/q[m]["active_max_s"]:.6f}x')
print(f'b300_hi32_u32addr_best_mode={min(q,key=lambda m:q[m]["wall_s"])}');print(f'residue={res["1"]}')
PY
