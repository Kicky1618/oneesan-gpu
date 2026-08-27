#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"; PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pattern10_depth8_depth4_graph_ab_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}_key4${TERNARY_KEY4}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
if (( NGPU != 8 )); then echo "Graph A/B currently requires NGPU=8" >&2; exit 2; fi
field(){ local k="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
run_one(){
  local tag="$1" script="$2" bin="$3"; shift 3
  echo "=== build $tag ===" >&2
  env N="$N" OUT="$bin" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" "$@" bash "$ONEESAN_ROOT/$script"
  echo "=== run $tag ===" >&2
  local so="$LOGDIR/${tag}.out" se="$LOGDIR/${tag}.err" line detail residue wall fh fl rl rh ts meta
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  line="$(grep '^residue=' "$so" | tail -n1)"; [[ -n "$line" ]] || { echo "$tag missing residue line" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$tag residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_batch modulus=' "$se" | tail -n1 || true)"; fh="$(field forward_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; rh="$(field reverse_high_s "$detail")"; ts="$(field transpose_s "$detail")"
  meta="$(grep -E 'bucket_(forward|reverse)_pattern10_depth[48] bytes=' "$se" | tr '\n' ';' || true)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$residue" "$wall" "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" "$bin" "${meta:-NA}" >>"$RESULT"
}
printf 'backend\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\tbinary\tmetadata_log\n' >"$RESULT"
run_one depth8 scripts/build/b300-bucket-snake-pattern10-depth8-graph-batch.sh "$ONEESAN_BUILD_DIR/ab_depth8_graph_n${N}" P10_DECODE=unrank HIGH_CTX=thread
run_one depth4 scripts/build/b300-bucket-snake-pattern10-depth4-graph-batch.sh "$ONEESAN_BUILD_DIR/ab_depth4_graph_n${N}"
cat "$RESULT"
w8="$(awk -F '\t' '$1=="depth8"{print $3}' "$RESULT")"; w4="$(awk -F '\t' '$1=="depth4"{print $3}' "$RESULT")"
python3 - "$w8" "$w4" <<'PY'
import sys
w8,w4=map(float,sys.argv[1:])
print(f"depth4_speedup_vs_depth8={w8/w4:.6f}x")
print(f"depth4_wall_delta_s={w4-w8:.6f}")
PY
echo "depth8-depth4-graph-ab OK n=$N transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM ternary_key4=$TERNARY_KEY4 result=$RESULT" >&2
