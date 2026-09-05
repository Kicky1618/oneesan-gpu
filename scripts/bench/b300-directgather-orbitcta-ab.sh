#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; REPEATS="${REPEATS:-1}"; PM_ACCUM="${PM_ACCUM:-0}"
LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_ab_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
WARP_BIN="$ONEESAN_BUILD_DIR/b300_orbitcta_ab_warp16_n${N}"
ORBIT16_BIN="$ONEESAN_BUILD_DIR/b300_orbitcta_ab_orbit16_n${N}"
ORBIT64_BIN="$ONEESAN_BUILD_DIR/b300_orbitcta_ab_orbit64_n${N}"

# 16-byte depth-major warpstriped reference.
N="$N" ARCH="$ARCH" OUT="$WARP_BIN" \
  RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 \
  RANKFORMULA_NOMETA_COOPGROUP=1 RANKFORMULA_NOMETA_COOP_UNROLL=0 \
  RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 \
  RANKFORMULA_NOMETA_DIRECTMAP=1 RANKFORMULA_DIRECTGATHER=1 \
  RANKFORMULA_DIRECTGATHER_DEPTHMAJOR=1 RANKFORMULA_DIRECTGATHER_FORCE7=0 \
  RANKFORMULA_MLP_WINDOW4=0 RANKFORMULA_PREFETCH_NEXT=0 RANKFORMULA_PAIR_MLP=0 \
  WARPSTRIPED_COL_ILP=1 RANKFORMULA_ABSTRACT_SELECT8=1 \
  RANKFORMULA_ABSTRACT_DEPTH4=1 RANKFORMULA_ABSTRACT_SRCPACK10=1 \
  RANKFORMULA_GATHER_MLP=1 DEPTHCODE_DECODE_LOAD=ldg RANKSTREAM_LUT_LOAD=ldg \
  PM_ACCUM="$PM_ACCUM" TRANSPOSE_MODE=pipeline PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" \
  >"$LOGDIR/warp16.build.out" 2>"$LOGDIR/warp16.build.err"

N="$N" ARCH="$ARCH" OUT="$ORBIT16_BIN" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
  >"$LOGDIR/orbit16.build.out" 2>"$LOGDIR/orbit16.build.err"
N="$N" ARCH="$ARCH" OUT="$ORBIT64_BIN" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
  >"$LOGDIR/orbit64.build.out" 2>"$LOGDIR/orbit64.build.err"

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$LOGDIR/warp16.build.err" --label warp16 >>"$RESOURCE" || true
python3 "$PARSER" "$LOGDIR/orbit16.build.err" --label orbit16 >>"$RESOURCE" || true
python3 "$PARSER" "$LOGDIR/orbit64.build.err" --label orbit64 >>"$RESOURCE" || true

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
printf 'mode\tdesc_bytes\tthreads\thigh_x\thigh_y\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"

run_case(){
  local mode="$1" bin="$2" desc="$3" threads="$4" gx="$5" gy="$6" orbit="$7"
  for ((rep=1;rep<=REPEATS;++rep)); do
    local so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err"
    if [[ "$orbit" == 1 ]]; then
      BUCKET_THREADS="$threads" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
      BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      BUCKET_ORBITCTA_GRID_Y="$gy" \
        "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    else
      BUCKET_THREADS="$threads" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
      BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      BUCKET_HIGH_GRID_X="$gx" BUCKET_HIGH_GRID_Y="$gy" \
        "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    fi
    local line detail residue
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$mode" "$desc" "$threads" "$gx" "$gy" "$rep" "$residue" "$(field wall_s "$line")" \
      "$(field forward_high_s "$detail")" "$(field reverse_high_s "$detail")" \
      "$(field forward_low_s "$detail")" "$(field reverse_low_s "$detail")" \
      "$(field transpose_s "$detail")" >>"$RESULT"
  done
}

# Scheduler isolation: same 16-byte descriptor.
run_case warp16_16x8 "$WARP_BIN" 16 256 16 8 0
run_case warp16_1x128 "$WARP_BIN" 16 256 1 128 0
run_case orbit16_t128_y128 "$ORBIT16_BIN" 16 128 1 128 1
run_case orbit16_t256_y64 "$ORBIT16_BIN" 16 256 1 64 1
run_case orbit16_t256_y128 "$ORBIT16_BIN" 16 256 1 128 1
run_case orbit16_t512_y64 "$ORBIT16_BIN" 16 512 1 64 1
# Descriptor isolation on the same orbit scheduler.
run_case orbit64_t128_y128 "$ORBIT64_BIN" 8 128 1 128 1
run_case orbit64_t256_y64 "$ORBIT64_BIN" 8 256 1 64 1
run_case orbit64_t256_y128 "$ORBIT64_BIN" 8 256 1 128 1
run_case orbit64_t512_y64 "$ORBIT64_BIN" 8 512 1 64 1

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
by={}
for r in rows: by.setdefault(r['mode'],[]).append(r)
out=[]
for mode,g in by.items():
    z={k:statistics.median(float(r[k]) for r in g) for k in ('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')}
    z['high']=z['forward_high_s']+z['reverse_high_s']
    out.append((z['wall_s'],mode,z))
for _,mode,z in sorted(out):
    print(mode,f"wall={z['wall_s']:.6f}",f"high={z['high']:.6f}",f"fh={z['forward_high_s']:.6f}",f"rh={z['reverse_high_s']:.6f}")
q={m:z for _,m,z in out}
base=q.get('warp16_16x8')
if base:
    for _,m,z in sorted(out):
        print(f"speedup_vs_warp16x8 {m} wall={base['wall_s']/z['wall_s']:.6f} high={base['high']/z['high']:.6f}")
if 'orbit16_t256_y128' in q and 'orbit64_t256_y128' in q:
    a=q['orbit16_t256_y128']; b=q['orbit64_t256_y128']
    print(f"DIRECTGATHER64_t256_y128 wall_speedup={a['wall_s']/b['wall_s']:.6f} high_speedup={a['high']/b['high']:.6f}")
print('BEST',sorted(out)[0][1],f"wall={sorted(out)[0][0]:.6f}")
PY

echo "result=$RESULT ptxas=$RESOURCE depthmajor=1 pm_accum=$PM_ACCUM" >&2
