#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"; REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pattern10_depthcode_ab_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}_key4${TERNARY_KEY4}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
if (( NGPU != 8 )); then echo "depthcode A/B currently requires NGPU=8" >&2; exit 2; fi
if (( REPEATS < 1 )); then echo "REPEATS must be >=1" >&2; exit 2; fi
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
field(){ local k="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

build_one(){
  local tag="$1" bin="$2"
  if [[ "$tag" == depth4_lut ]]; then
    N="$N" OUT="$bin" P10_DECODE=lut TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
      bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depth4-graph-batch.sh"
  else
    N="$N" OUT="$bin" DEPTHCODE_DECODE=pair TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
      bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh"
  fi
}
run_one(){
  local tag="$1" bin="$2" rep="$3" so="$LOGDIR/${tag}_r${rep}.out" se="$LOGDIR/${tag}_r${rep}.err"
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line residue wall detail fh fl rl rh ts meta
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$tag missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$tag residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; fh="$(field forward_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; rh="$(field reverse_high_s "$detail")"; ts="$(field transpose_s "$detail")"
  meta="$(grep -E 'pattern10_depthcode|bucket_(forward|reverse)_pattern10_depth4' "$se" | tr '\n' ';' || true)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$rep" "$residue" "$wall" "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" "$bin" "${meta:-NA}" >>"$RESULT"
}

printf 'backend\trepeat\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\tbinary\tmetadata_log\n' >"$RESULT"
for tag in depth4_lut depthcode; do
  bin="$ONEESAN_BUILD_DIR/ab_${tag}_graph_n${N}"
  echo "=== build $tag ===" >&2; build_one "$tag" "$bin"
  for ((r=1;r<=REPEATS;++r)); do echo "=== run $tag $r/$REPEATS ===" >&2; run_one "$tag" "$bin" "$r"; done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
by={}
for r in rows: by.setdefault(r['backend'],[]).append(r)
metrics=['wall_s','forward_high_s','forward_low_s','reverse_low_s','reverse_high_s','transpose_s']
out={}
for k,g in by.items():
    out[k]={m:statistics.median(float(r[m]) for r in g if r[m]!='NA') for m in metrics if any(r[m]!='NA' for r in g)}
with open(dst,'w',newline='') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['backend',*metrics])
    for k in ('depth4_lut','depthcode'):w.writerow([k,*[f"{out[k].get(m,float('nan')):.9f}" for m in metrics]])
a=out['depth4_lut']['wall_s'];b=out['depthcode']['wall_s']
print(f'depthcode_speedup_vs_depth4_lut={a/b:.6f}x')
print(f'depthcode_wall_delta_s={b-a:.6f}')
print(f'summary={dst}')
PY
echo "depthcode-ab OK n=$N expected=$EXPECT repeats=$REPEATS result=$RESULT" >&2
