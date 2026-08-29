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
RANKCHUNK32_ONESHFL="${RANKCHUNK32_ONESHFL:-1}"
RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-0}"
RANKCHUNK32_DIRECT3="${RANKCHUNK32_DIRECT3:-0}"
PROFILE_LOG2="${PROFILE_LOG2:-8}"
RANKCHUNK32_BYTEPACK="${RANKCHUNK32_BYTEPACK:-0}"
RANKCHUNK32_ALIGN32="${RANKCHUNK32_ALIGN32:-0}"
RANKCHUNK32_BLOCK64="${RANKCHUNK32_BLOCK64:-0}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_GRID_X="${BUCKET_GRID_X:-16}"
BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_rankchunk32_rankmask_profile_n${N}_${TRANSPOSE_MODE}_${RANKSTREAM_LUT_LOAD}_s${PROFILE_LOG2}}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_depthcode_rankchunk32_rankmask_profile_n${N}_s${PROFILE_LOG2}}"
BUILD_OUT="${BUILD_OUT:-${PREFIX}.build.out}"
BUILD_ERR="${BUILD_ERR:-${PREFIX}.build.err}"
RUN_OUT="${RUN_OUT:-${PREFIX}.out}"
RUN_ERR="${RUN_ERR:-${PREFIX}.err}"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
case "$RANKSTREAM_LUT_LOAD" in constant|ldg|ldg256) ;; *) echo "invalid RANKSTREAM_LUT_LOAD" >&2; exit 2;; esac
for x in RANKCHUNK32_ONESHFL RANKCHUNK32_FUSED16 RANKCHUNK32_DIRECT3 RANKCHUNK32_BYTEPACK RANKCHUNK32_ALIGN32 RANKCHUNK32_BLOCK64 PM_ACCUM TERNARY_KEY4; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if ! [[ "$PROFILE_LOG2" =~ ^[0-9]+$ ]] || (( PROFILE_LOG2 > 16 )); then
  echo "PROFILE_LOG2 must be an integer in [0,16]" >&2; exit 2
fi
if [[ "$RANKCHUNK32_BLOCK64" == 1 && "$RANKCHUNK32_BYTEPACK" == 1 ]]; then
  echo "RANKCHUNK32_BLOCK64 requires BYTEPACK=0" >&2; exit 2
fi
if (( NGPU != 8 || BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 || BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then
  echo "invalid launch/profile parameters" >&2; exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2; exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$BIN")" "$(dirname "$RUN_OUT")"

bash "$ONEESAN_ROOT/scripts/bench/cross5-rankmask-shape-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/cross5-rankstream-projection-proof.sh"

echo "=== build rankchunk32 sampled real-traffic rankmask profiler 1/$((1 << PROFILE_LOG2)) warps ===" >&2
N="$N" ARCH="$ARCH" OUT="$BIN" HIGH_CTX=warpstriped_delta_direct_affine_rankchunk32_cross5 \
  DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
  RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
  RANKCHUNK32_DIRECT3="$RANKCHUNK32_DIRECT3" RANKCHUNK32_RANKMASK_PROFILE=1 \
  RANKCHUNK32_RANKMASK_PROFILE_LOG2="$PROFILE_LOG2" \
  RANKCHUNK32_BYTEPACK="$RANKCHUNK32_BYTEPACK" RANKCHUNK32_ALIGN32="$RANKCHUNK32_ALIGN32" \
  RANKCHUNK32_BLOCK64="$RANKCHUNK32_BLOCK64" TRANSPOSE_MODE="$TRANSPOSE_MODE" \
  PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
  >"$BUILD_OUT" 2>"$BUILD_ERR"

echo "=== run one real production-shaped residue with sampled profiling enabled ===" >&2
BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
  "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$RUN_OUT" 2>"$RUN_ERR"

field() {
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

residue_line="$(grep '^residue=' "$RUN_OUT" | tail -n1 || true)"
[[ -n "$residue_line" ]] || { echo "missing residue output" >&2; exit 3; }
residue="$(field residue "$residue_line")"
[[ "$residue" == "$EXPECT" ]] || {
  echo "residue mismatch got=$residue expected=$EXPECT" >&2
  exit 4
}

profile_line="$(grep '^rankchunk32_rankmask_profile ' "$RUN_ERR" | tail -n1 || true)"
[[ -n "$profile_line" ]] || { echo "missing rankchunk32_rankmask_profile output" >&2; exit 5; }
printf '%s\n' "$profile_line"

sample_log2="$(field sample_log2 "$profile_line")"
sample_one_in="$(field sample_one_in "$profile_line")"
total="$(field total "$profile_line")"
m0="$(field m0 "$profile_line")"
m4="$(field m4 "$profile_line")"
m6="$(field m6 "$profile_line")"
other="$(field other "$profile_line")"
disallowed="$(field disallowed "$profile_line")"
zero_frac="$(field zero_frac "$profile_line")"
nonzero_frac="$(field nonzero_frac "$profile_line")"
avg_popcount="$(field avg_popcount "$profile_line")"
warp_events="$(field warp_events "$profile_line")"
warp_all_zero="$(field warp_all_zero "$profile_line")"
warp_all_nonzero="$(field warp_all_nonzero "$profile_line")"
warp_mixed="$(field warp_mixed "$profile_line")"
warp_all_zero_frac="$(field warp_all_zero_frac "$profile_line")"
warp_all_nonzero_frac="$(field warp_all_nonzero_frac "$profile_line")"
warp_mixed_frac="$(field warp_mixed_frac "$profile_line")"
avg_active_lanes="$(field avg_active_lanes "$profile_line")"
for x in "$sample_log2" "$sample_one_in" "$total" "$m0" "$m4" "$m6" "$other" "$disallowed" "$zero_frac" "$nonzero_frac" "$avg_popcount" \
         "$warp_events" "$warp_all_zero" "$warp_all_nonzero" "$warp_mixed" \
         "$warp_all_zero_frac" "$warp_all_nonzero_frac" "$warp_mixed_frac" "$avg_active_lanes"; do
  [[ -n "$x" ]] || { echo "failed to parse profile output" >&2; exit 6; }
done
[[ "$sample_log2" == "$PROFILE_LOG2" && "$sample_one_in" == "$((1 << PROFILE_LOG2))" ]] || {
  echo "profile sampling mismatch log2=$sample_log2 one_in=$sample_one_in expected_log2=$PROFILE_LOG2" >&2; exit 7
}
(( total > 0 && warp_events > 0 )) || { echo "rankmask profile sampled no traffic; lower PROFILE_LOG2" >&2; exit 8; }
[[ "$m4" == 0 && "$m6" == 0 && "$other" == 0 && "$disallowed" == 0 ]] || {
  echo "runtime rankmask shape violation m4=$m4 m6=$m6 other=$other disallowed=$disallowed" >&2
  exit 9
}
if (( warp_all_zero + warp_all_nonzero + warp_mixed != warp_events )); then
  echo "warp profile partition mismatch events=$warp_events zero=$warp_all_zero nonzero=$warp_all_nonzero mixed=$warp_mixed" >&2
  exit 10
fi

table_zero_frac="$(awk 'BEGIN { printf "%.9f", 5855.0/6075.0 }')"
zero_ratio="$(awk -v a="$zero_frac" -v b="$table_zero_frac" 'BEGIN { if (b == 0) print "inf"; else printf "%.6f", a / b }')"
printf 'rankchunk32-rankmask-profile OK n=%s modulus=%s residue=%s sampled_chunks=%s sample_log2=%s sample_one_in=%s\n' \
  "$N" "$MOD" "$residue" "$total" "$sample_log2" "$sample_one_in"
printf 'actual_zero_frac=%s actual_nonzero_frac=%s actual_avg_popcount=%s\n' "$zero_frac" "$nonzero_frac" "$avg_popcount"
printf 'warp_events=%s warp_all_zero_frac=%s warp_all_nonzero_frac=%s warp_mixed_frac=%s avg_active_lanes=%s\n' \
  "$warp_events" "$warp_all_zero_frac" "$warp_all_nonzero_frac" "$warp_mixed_frac" "$avg_active_lanes"
printf 'table_space_zero_frac=%s actual_to_table_zero_ratio=%s\n' "$table_zero_frac" "$zero_ratio"
printf 'rankmask_allowed=0,1,2,3,5,7 sampled_runtime_disallowed=0 warp_aggregated=1\n'
printf 'profile_estimate=deterministic_data_blind_warp_sample timing_valid_for_ab=0 note=profiling_instrumentation_intentionally_distorts_kernel_time\n'
printf 'run_stdout=%s run_stderr=%s\n' "$RUN_OUT" "$RUN_ERR"
