#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"
NGPU="${NGPU:-8}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-sync}"; PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
RESULT="${RESULT:-$ONEESAN_ROOT/work/b300_pattern10_decode_ab_n${N}.tsv}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300_pattern10_decode_ab_n${N}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
if (( NGPU != 8 )); then echo "decode A/B requires NGPU=8" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc/nvidia-smi required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; if (( visible < NGPU )); then echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; fi

field(){ local k="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
run_one(){
  local decode="$1" ctx="$2" tag="${decode}_${ctx}"
  local bin="$ONEESAN_BUILD_DIR/ab_p10_${tag}_${TRANSPOSE_MODE}_n${N}"
  echo "=== $tag ===" >&2
  N="$N" OUT="$bin" P10_DECODE="$decode" HIGH_CTX="$ctx" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depth8-batch.sh"
  local so="$LOGDIR/${tag}.out" se="$LOGDIR/${tag}.err"
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh fl rl rh ts
  line="$(grep '^residue=' "$so" | tail -n1)"; [[ -n "$line" ]] || { echo "$tag missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || { echo "$tag residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; rh="$(field reverse_high_s "$detail")"; ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$decode" "$ctx" "$residue" "$wall" "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" "$bin" >>"$RESULT"
}

printf 'pattern_decode\thigh_ctx\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\tbinary\n' >"$RESULT"
run_one unrank thread
run_one lut thread
run_one unrank shared
run_one lut shared
cat "$RESULT"
echo "pattern10-decode-ab OK n=$N modulus=$MOD expected=$EXPECT result=$RESULT" >&2
