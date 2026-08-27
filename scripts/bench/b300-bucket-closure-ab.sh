#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"
NGPU="${NGPU:-8}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-sync}"; PM_ACCUM="${PM_ACCUM:-0}"
RESULT="${RESULT:-$ONEESAN_ROOT/work/b300_bucket_closure_ab_n${N}.tsv}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300_bucket_closure_ab_n${N}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
if (( NGPU != 8 )); then echo "closure A/B currently requires NGPU=8" >&2; exit 2; fi
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; if (( visible < NGPU )); then echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; fi

field(){ local k="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
run_one(){
  local tag="$1" build_script="$2" bin="$3" high_ctx="${4:-thread}"
  echo "=== build $tag ===" >&2
  N="$N" OUT="$bin" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" HIGH_CTX="$high_ctx" bash "$ONEESAN_ROOT/$build_script"
  echo "=== run $tag ===" >&2
  local so="$LOGDIR/${tag}.out" se="$LOGDIR/${tag}.err"
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh fl rl rh ts
  line="$(grep '^residue=' "$so" | tail -n1)"; [[ -n "$line" ]] || { echo "$tag missing residue line" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  if [[ "$residue" != "$EXPECT" ]]; then echo "$tag residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; fi
  detail="$(grep 'snake_onepass_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; rh="$(field reverse_high_s "$detail")"; ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$residue" "$wall" "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" "$bin" >>"$RESULT"
}

printf 'backend\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\tbinary\n' >"$RESULT"
SUFFIX="_${TRANSPOSE_MODE}"; if [[ "$PM_ACCUM" == 1 ]]; then SUFFIX="${SUFFIX}_pm"; fi
run_one inline8 \
  scripts/build/b300-bucket-snake-hybrid18-inline8-batch.sh \
  "$ONEESAN_BUILD_DIR/ab_inline8${SUFFIX}_n${N}"
run_one zero_scan \
  scripts/build/b300-bucket-snake-zero-batch.sh \
  "$ONEESAN_BUILD_DIR/ab_zero_scan${SUFFIX}_n${N}"
run_one pattern10 \
  scripts/build/b300-bucket-snake-pattern10-batch.sh \
  "$ONEESAN_BUILD_DIR/ab_pattern10${SUFFIX}_n${N}"
run_one pattern10_depth8_thread \
  scripts/build/b300-bucket-snake-pattern10-depth8-batch.sh \
  "$ONEESAN_BUILD_DIR/ab_pattern10_depth8_thread${SUFFIX}_n${N}" thread
run_one pattern10_depth8_shared \
  scripts/build/b300-bucket-snake-pattern10-depth8-batch.sh \
  "$ONEESAN_BUILD_DIR/ab_pattern10_depth8_shared${SUFFIX}_n${N}" shared

cat "$RESULT"
echo "closure-ab OK n=$N modulus=$MOD expected=$EXPECT transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM result=$RESULT" >&2
