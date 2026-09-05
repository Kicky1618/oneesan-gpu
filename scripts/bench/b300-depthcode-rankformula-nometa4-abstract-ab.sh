#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-3}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"; RUN_SELFTEST="${RUN_SELFTEST:-1}"; RUN_PTXAS="${RUN_PTXAS:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }; fi
for x in PM_ACCUM TERNARY_KEY4 RUN_SELFTEST RUN_PTXAS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankformula_nometa4_abstract_ab_n${N}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-nometa-blocks-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankformula-abstract-lut-proof.sh"

if [[ "$RUN_SELFTEST" == 1 ]]; then
  RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" PM_ACCUM="$PM_ACCUM" \
    bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-selftest.sh" >"$LOGDIR/ballot.selftest.out" 2>"$LOGDIR/ballot.selftest.err"
  RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" PM_ACCUM="$PM_ACCUM" \
    bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-abstract-selftest.sh" >"$LOGDIR/abstract.selftest.out" 2>"$LOGDIR/abstract.selftest.err"
fi

ballot_bin="$ONEESAN_BUILD_DIR/ab_rankformula_nometa4_ballot_n${N}"
abstract_bin="$ONEESAN_BUILD_DIR/ab_rankformula_nometa4_abstract_n${N}"
N="$N" ARCH="$ARCH" OUT="$ballot_bin" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
  TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE="$RUN_PTXAS" \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4.sh" >"$LOGDIR/ballot.build.out" 2>"$LOGDIR/ballot.build.err"
N="$N" ARCH="$ARCH" OUT="$abstract_bin" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
  TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE="$RUN_PTXAS" \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" >"$LOGDIR/abstract.build.out" 2>"$LOGDIR/abstract.build.err"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
for mode in ballot abstract; do
  [[ "$mode" == ballot ]] && bin="$ballot_bin" || bin="$abstract_bin"
  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${mode}_r${r}.out"; se="$LOGDIR/${mode}_r${r}.err"
    BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue mismatch" >&2; exit 4; }
    wall="$(field wall_s "$line")"; detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$r" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" >>"$RESULT"
  done
done
cat "$RESULT"
echo "b300-depthcode-rankformula-nometa4-abstract-ab OK n=$N repeats=$REPEATS result=$RESULT logs=$LOGDIR" >&2
