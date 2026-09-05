#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
NGPU="${NGPU:-8}"
if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516
  else echo "EXPECT must be set when N/MOD differ" >&2; exit 2
  fi
fi

ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"
RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RANKCHUNK32_RANKPLANE="${RANKCHUNK32_RANKPLANE:-1}"
RANKCHUNK32_BLOCK64="${RANKCHUNK32_BLOCK64:-0}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_GRID_X="${BUCKET_GRID_X:-16}"
BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
REPEATS="${REPEATS:-2}"
ILP_MODES="${ILP_MODES:-1 2 4}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"
SAMPLE_GPU_UTIL="${SAMPLE_GPU_UTIL:-1}"
UTIL_SAMPLE_S="${UTIL_SAMPLE_S:-0.25}"

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_rankchunk32_ilp_ab_n${N}_${TRANSPOSE_MODE}_rankplane${RANKCHUNK32_RANKPLANE}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
case "$RANKSTREAM_LUT_LOAD" in constant|ldg|ldg256) ;; *) echo "invalid RANKSTREAM_LUT_LOAD" >&2; exit 2;; esac
for x in RANKCHUNK32_RANKPLANE RANKCHUNK32_BLOCK64 PM_ACCUM TERNARY_KEY4 PTXAS_VERBOSE RUN_SELFTEST SAMPLE_GPU_UTIL; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
for ilp in $ILP_MODES; do case "$ilp" in 1|2|4) ;; *) echo "ILP_MODES entries must be 1, 2, or 4" >&2; exit 2;; esac; done
if (( NGPU != 8 || REPEATS < 1 || BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 || BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then
  echo "invalid launch/A-B parameters" >&2; exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2; exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-directmask-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-warpbase-proof.sh"

if [[ "$RUN_SELFTEST" == 1 ]]; then
  for ilp in $ILP_MODES; do
    echo "=== selftest rankchunk32 directmask rankplane=$RANKCHUNK32_RANKPLANE ILP=$ilp ===" >&2
    W=10 ARCH="$ARCH" RUN_LAYOUT_PROOF=0 \
      RANKCHUNK32_DIRECTMASK=1 RANKCHUNK32_RANKPLANE="$RANKCHUNK32_RANKPLANE" \
      RANKCHUNK32_ILP="$ilp" RANKCHUNK32_ALIGN32=1 \
      RANKCHUNK32_ONESHFL=1 RANKCHUNK32_DIRECT3=0 RANKCHUNK32_FUSED16=0 \
      RANKCHUNK32_BYTEPACK=0 RANKCHUNK32_BLOCK64="$RANKCHUNK32_BLOCK64" \
      RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" PM_ACCUM="$PM_ACCUM" \
      DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
      bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankchunk32-cross5-selftest.sh" \
      >"$LOGDIR/ilp_${ilp}.selftest.out" 2>"$LOGDIR/ilp_${ilp}.selftest.err"
  done
fi

field() {
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

build_one() {
  local ilp="$1" bin="$2"
  N="$N" ARCH="$ARCH" OUT="$bin" HIGH_CTX=warpstriped_delta_direct_affine_rankchunk32_cross5 \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
    RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" RANKCHUNK32_DIRECTMASK=1 \
    RANKCHUNK32_RANKPLANE="$RANKCHUNK32_RANKPLANE" RANKCHUNK32_ILP="$ilp" \
    RANKCHUNK32_ALIGN32=1 RANKCHUNK32_ONESHFL=1 RANKCHUNK32_DIRECT3=0 \
    RANKCHUNK32_FUSED16=0 RANKCHUNK32_BYTEPACK=0 \
    RANKCHUNK32_BLOCK64="$RANKCHUNK32_BLOCK64" PM_ACCUM="$PM_ACCUM" \
    TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/ilp_${ilp}.build.out" 2>"$LOGDIR/ilp_${ilp}.build.err"
}

printf 'ilp\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\tsm_active_avg_pct\tmem_active_avg_pct\tmem_max_pct\n' >"$RESULT"

run_one() {
  local ilp="$1" bin="$2" rep="$3"
  local so="$LOGDIR/ilp_${ilp}_r${rep}.out" se="$LOGDIR/ilp_${ilp}_r${rep}.err"
  local util="$LOGDIR/ilp_${ilp}_r${rep}.util.tsv"
  : >"$util"

  if [[ "$SAMPLE_GPU_UTIL" == 1 ]]; then
    BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{gsub(/[[:space:]]/,"",$1); gsub(/[[:space:]]/,"",$2); if($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/) print $1 "\t" $2}' \
        >>"$util" || true
      sleep "$UTIL_SAMPLE_S"
    done
    wait "$pid"
  else
    BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  fi

  local line detail residue wall fh rh fl rl ts smu memu memmax
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "ilp=$ilp missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || { echo "ilp=$ilp residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; ts="$(field transpose_s "$detail")"

  smu=NA; memu=NA; memmax=NA
  if [[ -s "$util" ]]; then
    read -r smu memu memmax < <(awk '
      BEGIN {sa=0; ma=0; na=0; s_all=0; m_all=0; n_all=0; mx=0}
      $1 ~ /^[0-9]+([.][0-9]+)?$/ && $2 ~ /^[0-9]+([.][0-9]+)?$/ {
        s_all += $1; m_all += $2; ++n_all; if ($2 > mx) mx=$2;
        if ($1 >= 20) {sa += $1; ma += $2; ++na}
      }
      END {
        if (na) printf "%.3f %.3f %.3f\n", sa/na, ma/na, mx;
        else if (n_all) printf "%.3f %.3f %.3f\n", s_all/n_all, m_all/n_all, mx;
        else print "NA NA NA";
      }' "$util")
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ilp" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" \
    "$smu" "$memu" "$memmax" >>"$RESULT"
}

for ilp in $ILP_MODES; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_rankchunk32_rankplane${RANKCHUNK32_RANKPLANE}_ilp_${ilp}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build rankchunk32 rankplane=$RANKCHUNK32_RANKPLANE ILP=$ilp ===" >&2
  build_one "$ilp" "$bin"
  for ((r=1; r<=REPEATS; ++r)); do
    echo "=== run rankchunk32 rankplane=$RANKCHUNK32_RANKPLANE ILP=$ilp $r/$REPEATS ===" >&2
    run_one "$ilp" "$bin" "$r"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RANKCHUNK32_RANKPLANE" <<'PY'
import csv, statistics, sys
src,dst,rankplane=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
metrics=('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s',
         'sm_active_avg_pct','mem_active_avg_pct','mem_max_pct')
modes=[]
for mode in sorted({r['ilp'] for r in rows}, key=int):
    group=[r for r in rows if r['ilp']==mode]
    z={'ilp':mode,'repeats':str(len(group))}
    for m in metrics:
        xs=[float(r[m]) for r in group if r[m]!='NA']
        z[m]=f'{statistics.median(xs):.9f}' if xs else 'NA'
    modes.append(z)
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=('ilp','repeats',*metrics),delimiter='\t'); w.writeheader(); w.writerows(modes)
q={r['ilp']:r for r in modes}
if '1' in q:
    for mode in ('2','4'):
        if mode not in q: continue
        for m in ('wall_s','forward_high_s','reverse_high_s'):
            if q['1'][m]!='NA' and q[mode][m]!='NA':
                print(f'rankchunk32_ilp1_to_{mode}_{m}_speedup={float(q["1"][m])/float(q[mode][m]):.6f}x')
        if all(q[x][m]!='NA' for x in ('1',mode) for m in ('forward_high_s','reverse_high_s')):
            a=float(q['1']['forward_high_s'])+float(q['1']['reverse_high_s'])
            b=float(q[mode]['forward_high_s'])+float(q[mode]['reverse_high_s'])
            print(f'rankchunk32_ilp1_to_{mode}_total_high_speedup={a/b:.6f}x')
for r in modes:
    print(f'ilp{r["ilp"]}_sm_active_avg_pct={r["sm_active_avg_pct"]}')
    print(f'ilp{r["ilp"]}_mem_active_avg_pct={r["mem_active_avg_pct"]}')
    print(f'ilp{r["ilp"]}_mem_max_pct={r["mem_max_pct"]}')
print('ilp_model=independent_lr_chains_batched_before_consumption')
print('rankplane='+rankplane)
print('directmask_runtime_path=' + ('mask8_then_rank16plane_then_source32' if rankplane=='1' else 'mask8_then_offset32_then_rank16_then_source32'))
print('rankchunk_meta_runtime=0')
print('blockbase_shuffle_runtime=0')
print('ilp2_outstanding_chains_per_lane=2')
print('ilp4_outstanding_chains_per_lane=4')
print('directmask=1')
print('align32=1')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
python3 - "$LOGDIR" $ILP_MODES <<'PY'
import pathlib, re, sys
logdir=pathlib.Path(sys.argv[1])
for mode in sys.argv[2:]:
    path=logdir/f'ilp_{mode}.build.err'
    text=path.read_text(errors='replace') if path.exists() else ''
    regs=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers', text)]
    stores=[int(x) for x in re.findall(r'(\d+)\s+bytes spill stores', text)]
    loads=[int(x) for x in re.findall(r'(\d+)\s+bytes spill loads', text)]
    blocks=re.split(r"(?=ptxas info\s*: Compiling entry function )", text)
    hot=[]
    for block in blocks:
        if 'rankchunk32_cross5_kernel' not in block:
            continue
        r=re.search(r'Used\s+(\d+)\s+registers', block)
        ss=re.search(r'(\d+)\s+bytes spill stores', block)
        sl=re.search(r'(\d+)\s+bytes spill loads', block)
        hot.append((int(r.group(1)) if r else -1,
                    int(ss.group(1)) if ss else 0,
                    int(sl.group(1)) if sl else 0))
    print(f'ptxas_ilp{mode}_max_registers_all={max(regs) if regs else "NA"}')
    print(f'ptxas_ilp{mode}_max_spill_store_bytes_all={max(stores) if stores else 0}')
    print(f'ptxas_ilp{mode}_max_spill_load_bytes_all={max(loads) if loads else 0}')
    if hot:
        print(f'ptxas_ilp{mode}_hot_max_registers={max(x[0] for x in hot)}')
        print(f'ptxas_ilp{mode}_hot_max_spill_store_bytes={max(x[1] for x in hot)}')
        print(f'ptxas_ilp{mode}_hot_max_spill_load_bytes={max(x[2] for x in hot)}')
        print(f'ptxas_ilp{mode}_hot_spill_free={int(all(x[1]==0 and x[2]==0 for x in hot))}')
    else:
        print(f'ptxas_ilp{mode}_hot_kernel_parse=NA')
PY
fi

echo "b300-depthcode-rankchunk32-ilp-ab OK n=$N repeats=$REPEATS modes='$ILP_MODES' rankplane=$RANKCHUNK32_RANKPLANE selftest=$RUN_SELFTEST sample_gpu_util=$SAMPLE_GPU_UTIL result=$RESULT logs=$LOGDIR" >&2
