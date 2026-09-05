#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
NGPU="${NGPU:-8}"
if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516
  else echo "EXPECT must be set when N/MOD differ" >&2; exit 2; fi
fi
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"
RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RANKCHUNK32_ONESHFL="${RANKCHUNK32_ONESHFL:-1}"
RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-0}"
RANKCHUNK32_BYTEPACK="${RANKCHUNK32_BYTEPACK:-0}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_GRID_X="${BUCKET_GRID_X:-16}"
BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
REPEATS="${REPEATS:-5}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"
RUN_PTXAS="${RUN_PTXAS:-1}"

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_rankchunk32_align32_ab_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_${RANKSTREAM_LUT_LOAD}_bytepack${RANKCHUNK32_BYTEPACK}_fused${RANKCHUNK32_FUSED16}_pm${PM_ACCUM}_t${BUCKET_THREADS}_gx${BUCKET_GRID_X}_gy${BUCKET_GRID_Y}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
case "$RANKSTREAM_LUT_LOAD" in constant|ldg|ldg256) ;; *) echo "invalid RANKSTREAM_LUT_LOAD" >&2; exit 2;; esac
for x in RANKCHUNK32_ONESHFL RANKCHUNK32_FUSED16 RANKCHUNK32_BYTEPACK PM_ACCUM TERNARY_KEY4 RUN_SELFTEST RUN_PTXAS; do
  v="${!x}"; if [[ "$v" != 0 && "$v" != 1 ]]; then echo "$x must be 0 or 1" >&2; exit 2; fi
done
if (( NGPU != 8 || REPEATS < 1 || BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 || BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then echo "invalid launch/A-B parameters" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/cross5-rankstream-projection-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-warpbase-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-align32-proof.sh"
if [[ "$RANKCHUNK32_BYTEPACK" == 1 ]]; then bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-bytepack-proof.sh"; fi

if [[ "$RUN_SELFTEST" == 1 ]]; then
  for align in 0 1; do
    RUN_LAYOUT_PROOF=0 RANKCHUNK32_ALIGN32="$align" RANKCHUNK32_BYTEPACK="$RANKCHUNK32_BYTEPACK" \
      RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" \
      RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" PM_ACCUM="$PM_ACCUM" DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
      bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankchunk32-cross5-selftest.sh" \
      >"$LOGDIR/selftest_align${align}.out" 2>"$LOGDIR/selftest_align${align}.err"
  done
fi

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
build_one(){
  local align="$1" bin="$2"
  N="$N" OUT="$bin" HIGH_CTX=warpstriped_delta_direct_affine_rankchunk32_cross5 \
    DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
    RANKCHUNK32_BYTEPACK="$RANKCHUNK32_BYTEPACK" RANKCHUNK32_ALIGN32="$align" \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE="$RUN_PTXAS" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/align${align}.build.out" 2>"$LOGDIR/align${align}.build.err"
}
printf 'align32\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"
if [[ "$RUN_PTXAS" == 1 ]]; then printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"; fi
run_one(){
  local align="$1" bin="$2" rep="$3" so="$LOGDIR/align${align}_r${rep}.out" se="$LOGDIR/align${align}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "align32=$align missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "align32=$align residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$align" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" >>"$RESULT"
}

for align in 0 1; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_rankchunk32_align${align}_${RANKSTREAM_LUT_LOAD}_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build rankchunk32 align32=$align ===" >&2; build_one "$align" "$bin"
  if [[ "$RUN_PTXAS" == 1 ]]; then python3 "$PARSER" "$LOGDIR/align${align}.build.err" --label "align${align}" >>"$RESOURCE"; fi
  for ((r=1;r<=REPEATS;++r)); do echo "=== run rankchunk32 align32=$align $r/$REPEATS ===" >&2; run_one "$align" "$bin" "$r"; done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RESOURCE" "$RUN_PTXAS" "$LOGDIR" <<'PY'
import csv,re,statistics,sys,pathlib
src,dst,resource,run_ptxas,logdir=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t')); metrics=('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')
out=[]
for mode in ('0','1'):
    g=[r for r in rows if r['align32']==mode]; z={'align32':mode,'repeats':str(len(g))}
    for m in metrics:
        xs=[float(r[m]) for r in g if r[m]!='NA']; z[m]=f'{statistics.median(xs):.9f}' if xs else 'NA'
    out.append(z)
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=('align32','repeats',*metrics),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['align32']:r for r in out}
for m in ('wall_s','forward_high_s','reverse_high_s'):
    if q['0'][m]!='NA' and q['1'][m]!='NA': print(f'rankchunk32_align32_{m}_speedup={float(q["0"][m])/float(q["1"][m]):.6f}x')
if all(q[x][m]!='NA' for x in ('0','1') for m in ('forward_high_s','reverse_high_s')):
    a=float(q['0']['forward_high_s'])+float(q['0']['reverse_high_s']); b=float(q['1']['forward_high_s'])+float(q['1']['reverse_high_s'])
    print(f'rankchunk32_align32_total_high_speedup={a/b:.6f}x')
if run_ptxas=='1':
    rr=list(csv.DictReader(open(resource),delimiter='\t'))
    for mode in ('0','1'):
        g=[r for r in rr if r['backend']==f'align{mode}' and 'high' in r['kernel'].lower()]
        def vals(k): return [int(r[k]) for r in g if r[k]!='NA']
        regs=vals('registers'); ss=vals('spill_store_bytes'); sl=vals('spill_load_bytes')
        print(f'rankchunk32_align{mode}_high_max_registers={max(regs) if regs else "NA"}')
        print(f'rankchunk32_align{mode}_high_spill_store_bytes={sum(ss) if ss else "NA"}')
        print(f'rankchunk32_align{mode}_high_spill_load_bytes={sum(sl) if sl else "NA"}')
ld=pathlib.Path(logdir)
for mode in ('0','1'):
    p=ld/f'align{mode}_r1.err'
    if p.exists():
        vals=[tuple(map(int,x)) for x in re.findall(r'p10dc_low_rankchunk32 fixed_owner=\d+ codes=(\d+) blocks=(\d+) l_ranks=(\d+) bytes=(\d+) meta_entries=(\d+) padding=(\d+)',p.read_text(errors='replace'))]
        if vals:
            print(f'rankchunk32_align{mode}_runtime_metadata_mib_total={sum(v[3] for v in vals)/(1<<20):.6f}')
            print(f'rankchunk32_align{mode}_padding_entries_total={sum(v[5] for v in vals)}')
            print(f'rankchunk32_align{mode}_meta_entries_total={sum(v[4] for v in vals)}')
print('rankchunk32_align0_block_base_loads_per_warp_max=2')
print('rankchunk32_align1_block_base_loads_per_warp_max=1')
print('rankchunk32_align1_max_padding_bytes_per_owner_bound=3720')
print('rankchunk32_rankstream_payload_unchanged=1')
print(f'summary={dst}')
PY

echo "b300-depthcode-rankchunk32-align32-ab OK n=$N repeats=$REPEATS lut=$RANKSTREAM_LUT_LOAD decode_load=$DEPTHCODE_DECODE_LOAD bytepack=$RANKCHUNK32_BYTEPACK fused16=$RANKCHUNK32_FUSED16 transpose=$TRANSPOSE_MODE result=$RESULT resource=$RESOURCE" >&2
