#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516
  else echo "EXPECT must be set when N/MOD differ" >&2; exit 2; fi
fi
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"; DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RANKCHUNK32_ONESHFL="${RANKCHUNK32_ONESHFL:-1}"; RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-0}"; PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
REPEATS="${REPEATS:-5}"; RUN_SELFTEST="${RUN_SELFTEST:-1}"; RUN_PTXAS="${RUN_PTXAS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_rankchunk32_layout_sweep_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_${RANKSTREAM_LUT_LOAD}_fused${RANKCHUNK32_FUSED16}_pm${PM_ACCUM}_t${BUCKET_THREADS}_gx${BUCKET_GRID_X}_gy${BUCKET_GRID_Y}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo invalid TRANSPOSE_MODE >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo invalid DEPTHCODE_DECODE_LOAD >&2; exit 2;; esac
case "$RANKSTREAM_LUT_LOAD" in constant|ldg|ldg256) ;; *) echo invalid RANKSTREAM_LUT_LOAD >&2; exit 2;; esac
for x in RANKCHUNK32_ONESHFL RANKCHUNK32_FUSED16 PM_ACCUM TERNARY_KEY4 RUN_SELFTEST RUN_PTXAS; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if (( NGPU != 8 || REPEATS < 1 || BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 || BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then
  echo invalid launch/sweep parameters >&2; exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

# label bytepack align block64 backend
CONFIGS=(
  "compact32 0 0 0 normal"
  "bytepack32 1 0 0 normal"
  "compact32_align 0 1 0 normal"
  "compact64 0 0 1 normal"
  "compact64_align 0 1 1 normal"
  "basepair64 1 1 0 basepair"
)

bash "$ONEESAN_ROOT/scripts/bench/cross5-rankstream-projection-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-warpbase-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-bytepack-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-align32-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-block64-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-block64-align-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-basepair64-proof.sh"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

build_one(){
  local label="$1" bytepack="$2" align="$3" block64="$4" backend="$5" bin="$6" highctx
  if [[ "$backend" == basepair ]]; then
    highctx=warpstriped_delta_direct_affine_rankchunk32_basepair64_cross5
  else
    highctx=warpstriped_delta_direct_affine_rankchunk32_cross5
  fi
  N="$N" OUT="$bin" HIGH_CTX="$highctx" \
    DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
    RANKCHUNK32_BYTEPACK="$bytepack" RANKCHUNK32_ALIGN32="$align" RANKCHUNK32_BLOCK64="$block64" \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE="$RUN_PTXAS" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/${label}.build.out" 2>"$LOGDIR/${label}.build.err"
}

run_one(){
  local label="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${label}_r${rep}.out" se="$LOGDIR/${label}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$label missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || { echo "$label residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" >>"$RESULT"
}

printf 'layout\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"
if [[ "$RUN_PTXAS" == 1 ]]; then printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"; fi

for spec in "${CONFIGS[@]}"; do
  read -r label bytepack align block64 backend <<<"$spec"
  if [[ "$RUN_SELFTEST" == 1 ]]; then
    echo "=== selftest $label ===" >&2
    if [[ "$backend" == basepair ]]; then
      RUN_LAYOUT_PROOF=0 RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
        RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" PM_ACCUM="$PM_ACCUM" DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
        bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankchunk32-basepair64-cross5-selftest.sh" \
        >"$LOGDIR/${label}.selftest.out" 2>"$LOGDIR/${label}.selftest.err"
    else
      RUN_LAYOUT_PROOF=0 RANKCHUNK32_BYTEPACK="$bytepack" RANKCHUNK32_ALIGN32="$align" RANKCHUNK32_BLOCK64="$block64" \
        RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" \
        RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" PM_ACCUM="$PM_ACCUM" DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
        bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankchunk32-cross5-selftest.sh" \
        >"$LOGDIR/${label}.selftest.out" 2>"$LOGDIR/${label}.selftest.err"
    fi
  fi
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_rankchunk32_layout_${label}_${RANKSTREAM_LUT_LOAD}_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build $label ===" >&2
  build_one "$label" "$bytepack" "$align" "$block64" "$backend" "$bin"
  if [[ "$RUN_PTXAS" == 1 ]]; then python3 "$PARSER" "$LOGDIR/${label}.build.err" --label "$label" >>"$RESOURCE"; fi
  for ((r=1;r<=REPEATS;++r)); do echo "=== run $label $r/$REPEATS ===" >&2; run_one "$label" "$bin" "$r"; done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RESOURCE" "$RUN_PTXAS" "$LOGDIR" <<'PY'
import csv, pathlib, re, statistics, sys
src,dst,resource,run_ptxas,logdir=sys.argv[1:]
layouts=('compact32','bytepack32','compact32_align','compact64','compact64_align','basepair64')
rows=list(csv.DictReader(open(src),delimiter='\t'))
metrics=('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')
out=[]
for layout in layouts:
    g=[r for r in rows if r['layout']==layout]
    z={'layout':layout,'repeats':str(len(g))}
    for m in metrics:
        xs=[float(r[m]) for r in g if r[m]!='NA']
        z[m]=f'{statistics.median(xs):.9f}' if xs else 'NA'
    if z['forward_high_s']!='NA' and z['reverse_high_s']!='NA':
        z['total_high_s']=f'{float(z["forward_high_s"])+float(z["reverse_high_s"]):.9f}'
    else:
        z['total_high_s']='NA'
    out.append(z)
fields=('layout','repeats',*metrics,'total_high_s')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter='\t'); w.writeheader(); w.writerows(out)
base=next(r for r in out if r['layout']=='compact32')
for r in out:
    if base['wall_s']!='NA' and r['wall_s']!='NA': print(f'{r["layout"]}_wall_speedup_vs_compact32={float(base["wall_s"])/float(r["wall_s"]):.6f}x')
    if base['total_high_s']!='NA' and r['total_high_s']!='NA': print(f'{r["layout"]}_total_high_speedup_vs_compact32={float(base["total_high_s"])/float(r["total_high_s"]):.6f}x')
valid_wall=[r for r in out if r['wall_s']!='NA']; valid_high=[r for r in out if r['total_high_s']!='NA']
if valid_wall: print('rankchunk32_layout_best_wall='+min(valid_wall,key=lambda r:float(r['wall_s']))['layout'])
if valid_high: print('rankchunk32_layout_best_total_high='+min(valid_high,key=lambda r:float(r['total_high_s']))['layout'])
ld=pathlib.Path(logdir)
for layout in layouts:
    p=ld/f'{layout}_r1.err'
    if not p.exists(): continue
    text=p.read_text(errors='replace')
    vals=[tuple(map(int,x)) for x in re.findall(r'p10dc_low_rankchunk32 fixed_owner=\d+ codes=(\d+) blocks=(\d+) l_ranks=(\d+) bytes=(\d+) meta_entries=(\d+) padding=(\d+)',text)]
    if vals:
        table_bytes=sum(v[3] for v in vals)
        if layout=='basepair64':
            pairs=[tuple(map(int,x)) for x in re.findall(r'p10dc_low_rankchunk32_basepair64 fixed_owner=\d+ base32_entries=(\d+) pair_entries=(\d+) pair_bytes=(\d+)',text)]
            if pairs:
                table_bytes -= sum(v[0]*4 for v in pairs)
                table_bytes += sum(v[2] for v in pairs)
                print(f'{layout}_pair_entries_total={sum(v[1] for v in pairs)}')
                print(f'{layout}_pair_bytes_total={sum(v[2] for v in pairs)}')
        print(f'{layout}_runtime_table_mib_total={table_bytes/(1<<20):.6f}')
        print(f'{layout}_parent_blocks_total={sum(v[1] for v in vals)}')
        print(f'{layout}_padding_entries_total={sum(v[5] for v in vals)}')
if run_ptxas=='1':
    rr=list(csv.DictReader(open(resource),delimiter='\t'))
    for layout in layouts:
        g=[r for r in rr if r['backend']==layout and 'high' in r['kernel'].lower()]
        def ivals(k): return [int(r[k]) for r in g if r[k]!='NA']
        regs,ss,sl=ivals('registers'),ivals('spill_store_bytes'),ivals('spill_load_bytes')
        print(f'{layout}_high_max_registers={max(regs) if regs else "NA"}')
        print(f'{layout}_high_spill_store_bytes={sum(ss) if ss else "NA"}')
        print(f'{layout}_high_spill_load_bytes={sum(sl) if sl else "NA"}')
print('layout_compact32=23+9_block32_packed_heights_max2loads')
print('layout_bytepack32=24+8_block32_packed_heights_max2loads')
print('layout_compact32_align=23+9_block32_align32_max1load')
print('layout_compact64=23+9_block64_packed_heights_max2loads')
print('layout_compact64_align=23+9_block64_align64_max1load')
print('layout_basepair64=24+8_block32_align32_base22_delta8_pair64_max1load')
print('basepair64_block_base_bytes_per_code=0.0625')
print(f'summary={dst}')
PY

echo "b300-depthcode-rankchunk32-layout-sweep OK n=$N repeats=$REPEATS lut=$RANKSTREAM_LUT_LOAD decode_load=$DEPTHCODE_DECODE_LOAD fused16=$RANKCHUNK32_FUSED16 transpose=$TRANSPOSE_MODE result=$RESULT resource=$RESOURCE" >&2
