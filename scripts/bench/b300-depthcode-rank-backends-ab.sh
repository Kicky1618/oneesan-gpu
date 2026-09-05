#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516; else echo "EXPECT must be set when N/MOD differ" >&2; exit 2; fi
fi
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"; DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"
RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-constant}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
REPEATS="${REPEATS:-3}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_rank_backends_ab_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_ranklut${RANKSTREAM_LUT_LOAD}_pm${PM_ACCUM}_t${BUCKET_THREADS}_gx${BUCKET_GRID_X}_gy${BUCKET_GRID_Y}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
case "$RANKSTREAM_LUT_LOAD" in constant|ldg|ldg256) ;; *) echo "invalid RANKSTREAM_LUT_LOAD" >&2; exit 2;; esac
if (( NGPU != 8 || REPEATS < 1 || BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 || BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then echo "invalid launch/A-B parameters" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/cross5-automaton-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/low-rank16-plan.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankstream32-warpbase-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-warpbase-proof.sh"
field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
build_one(){ local ctx="$1" bin="$2"; N="$N" OUT="$bin" HIGH_CTX="$ctx" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" >"$LOGDIR/${ctx}.build.out" 2>"$LOGDIR/${ctx}.build.err"; }
printf 'high_ctx\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"
run_one(){
  local ctx="$1" bin="$2" rep="$3" so="$LOGDIR/${ctx}_r${rep}.out" se="$LOGDIR/${ctx}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$ctx missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$ctx residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ctx" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" >>"$RESULT"
}

contexts=(
  warpstriped_delta_direct_affine_prekey_cross5
  warpstriped_delta_direct_affine_prekey_rank16_cross5
  warpstriped_delta_direct_affine_prekey_rankstream_cross5
  warpstriped_delta_direct_affine_rankstream32_cross5
  warpstriped_delta_direct_affine_rankchunk32_cross5
)
for ctx in "${contexts[@]}"; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_${ctx}_${DEPTHCODE_DECODE_LOAD}_${RANKSTREAM_LUT_LOAD}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build $ctx ===" >&2; build_one "$ctx" "$bin"
  for ((r=1;r<=REPEATS;++r)); do echo "=== run $ctx $r/$REPEATS ===" >&2; run_one "$ctx" "$bin" "$r"; done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$LOGDIR" <<'PY'
import csv,re,statistics,sys,pathlib
src,dst,logdir=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
metrics=('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')
base='warpstriped_delta_direct_affine_prekey_cross5'
r16='warpstriped_delta_direct_affine_prekey_rank16_cross5'
rstream='warpstriped_delta_direct_affine_prekey_rankstream_cross5'
r32='warpstriped_delta_direct_affine_rankstream32_cross5'
rc32='warpstriped_delta_direct_affine_rankchunk32_cross5'
order=(base,r16,rstream,r32,rc32); out=[]
for ctx in order:
    g=[r for r in rows if r['high_ctx']==ctx]
    z={'high_ctx':ctx,'repeats':str(len(g))}
    for m in metrics:
        xs=[float(r[m]) for r in g if r[m]!='NA']; z[m]=f'{statistics.median(xs):.9f}' if xs else 'NA'
    out.append(z)
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=('high_ctx','repeats',*metrics),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['high_ctx']:r for r in out}
def ratio(a,b,m,label):
    if q[a][m]!='NA' and q[b][m]!='NA': print(f'{label}_{m}_speedup={float(q[a][m])/float(q[b][m]):.6f}x')
for m in ('wall_s','forward_high_s','reverse_high_s'):
    ratio(base,r16,m,'rank16'); ratio(base,rstream,m,'rankstream'); ratio(base,r32,m,'rankstream32'); ratio(base,rc32,m,'rankchunk32')
    ratio(r16,r32,m,'rankstream32_vs_rank16'); ratio(rstream,r32,m,'rankstream32_vs_rankstream'); ratio(r32,rc32,m,'rankchunk32_vs_rankstream32'); ratio(rstream,rc32,m,'rankchunk32_vs_rankstream')
for ctx,label in ((r16,'rank16'),(rstream,'rankstream'),(r32,'rankstream32'),(rc32,'rankchunk32')):
    if all(q[x][m]!='NA' for x in (base,ctx) for m in ('forward_high_s','reverse_high_s')):
        b=float(q[base]['forward_high_s'])+float(q[base]['reverse_high_s']); o=float(q[ctx]['forward_high_s'])+float(q[ctx]['reverse_high_s']); print(f'{label}_total_high_speedup={b/o:.6f}x')
if all(q[x][m]!='NA' for x in (r32,rc32) for m in ('forward_high_s','reverse_high_s')):
    a=float(q[r32]['forward_high_s'])+float(q[r32]['reverse_high_s']); b=float(q[rc32]['forward_high_s'])+float(q[rc32]['reverse_high_s']); print(f'rankchunk32_vs_rankstream32_total_high_speedup={a/b:.6f}x')
ld=pathlib.Path(logdir)
rs=list(ld.glob('*prekey_rankstream_cross5_r1.err'))
if rs:
    vals=[tuple(map(int,m)) for m in re.findall(r'p10dc_low_rankstream fixed_owner=\d+ codes=(\d+) l_ranks=(\d+)',rs[0].read_text(errors='replace'))]
    if vals:
        codes=sum(x for x,_ in vals); ranks=sum(y for _,y in vals)
        print(f'rankstream_offset_rank_mib_total={(codes*4+ranks*2)/(1<<20):.6f}'); print(f'rankstream_runtime_metadata_mib_total={(codes*8+ranks*2)/(1<<20):.6f}')
r32logs=list(ld.glob('*affine_rankstream32_cross5_r1.err'))
if r32logs:
    vals=[tuple(map(int,m)) for m in re.findall(r'p10dc_low_rankstream32 fixed_owner=\d+ codes=(\d+) blocks=(\d+) l_ranks=(\d+) bytes=(\d+)',r32logs[0].read_text(errors='replace'))]
    if vals:
        print(f'rankstream32_runtime_metadata_mib_total={sum(v[3] for v in vals)/(1<<20):.6f}'); print(f'rankstream32_codes_total={sum(v[0] for v in vals)}'); print(f'rankstream32_blocks_total={sum(v[1] for v in vals)}')
rc32logs=list(ld.glob('*affine_rankchunk32_cross5_r1.err'))
if rc32logs:
    text=rc32logs[0].read_text(errors='replace')
    vals=[tuple(map(int,m)) for m in re.findall(r'p10dc_low_rankchunk32 fixed_owner=\d+ codes=(\d+) blocks=(\d+) l_ranks=(\d+) bytes=(\d+) meta_entries=(\d+) padding=(\d+)',text)]
    if vals:
        print(f'rankchunk32_runtime_metadata_mib_total={sum(v[3] for v in vals)/(1<<20):.6f}'); print(f'rankchunk32_codes_total={sum(v[0] for v in vals)}'); print(f'rankchunk32_blocks_total={sum(v[1] for v in vals)}'); print(f'rankchunk32_meta_entries_total={sum(v[4] for v in vals)}'); print(f'rankchunk32_padding_entries_total={sum(v[5] for v in vals)}')
print('rankstream_model=prekey32+offset32+rank16_per_L')
print('rankstream32_model=key23+prefix9_per_code+blockbase32+rank16_per_L')
print('rankstream32_block_base_loads_per_warp_max=2')
print('rankstream32_cross_runtime_divmod=1')
print('rankchunk32_model=chunk23+prefix9_per_code+blockbase32+rank16_per_L')
print('rankchunk32_height_padding_entries=0')
print('rankchunk32_third_chunk_bits=7')
print('rankchunk32_block_base_loads_per_warp_max=2')
print('rankchunk32_cross_runtime_divmod=0')
print('rankchunk32_cross_runtime_direct_lookup=0')
print('fallback_structurally_unreachable=1')
print(f'summary={dst}')
PY

echo "depthcode-rank-backends-ab OK n=$N repeats=$REPEATS threads=$BUCKET_THREADS gx=$BUCKET_GRID_X gy=$BUCKET_GRID_Y decode_load=$DEPTHCODE_DECODE_LOAD rankstream_lut_load=$RANKSTREAM_LUT_LOAD transpose=$TRANSPOSE_MODE result=$RESULT" >&2
