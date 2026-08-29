#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
SYMBOL_AUDIT="${SYMBOL_AUDIT:-1}"
NVCC="${NVCC:-nvcc}"
CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
[[ "$SYMBOL_AUDIT" == 0 || "$SYMBOL_AUDIT" == 1 ]] || { echo "SYMBOL_AUDIT must be 0 or 1" >&2; exit 2; }
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
if [[ "$SYMBOL_AUDIT" == 1 ]]; then command -v "$CUOBJDUMP" >/dev/null || { echo "$CUOBJDUMP not found; set SYMBOL_AUDIT=0 to skip ELF symbol audit" >&2; exit 2; }; fi
SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_codec_table_physical_compile_probe.cu"
[[ -f "$SRC" ]] || { echo "missing physical compile probe: $SRC" >&2; exit 2; }
PREFIX="${PREFIX:-$ONEESAN_BUILD_DIR/gridfp_codec_table_physical_compile}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/gridfp_codec_table_physical_compile_matrix_logs}"
mkdir -p "$LOGDIR" "$(dirname "$PREFIX")"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")

expected_choose_symbol() {
  case "$1" in
    1) echo RP_CODEC_PHYSICAL_CHOOSE_SYM_U32;;
    2) echo RP_CODEC_PHYSICAL_CHOOSE_TRI_U32;;
    3) echo RP_CODEC_PHYSICAL_CHOOSE_FULL_U32;;
    *) return 2;;
  esac
}
expected_primitive_symbol() {
  case "$1" in
    1) echo RP_CODEC_PHYSICAL_PRIMITIVE_SYM_U32;;
    2) echo RP_CODEC_PHYSICAL_PRIMITIVE_FULL_U32;;
    *) return 2;;
  esac
}

audit_symbols() {
  local choose="$1" primitive="$2" bin="$3" symbols="$4" serr="$5"
  "$CUOBJDUMP" --dump-elf-symbols "$bin" >"$symbols" 2>"$serr"
  [[ -s "$symbols" ]] || { echo "empty cuobjdump symbol output c=$choose p=$primitive" >&2; cat "$serr" >&2 || true; exit 7; }

  if [[ "$choose" == 0 ]]; then
    grep -Fq 'RP_CHOOSE' "$symbols" || { echo "baseline choose symbol missing c=$choose p=$primitive" >&2; exit 8; }
  else
    if grep -Fq 'RP_CHOOSE' "$symbols"; then
      echo "legacy RP_CHOOSE symbol survived physical mode c=$choose p=$primitive" >&2
      grep -F 'RP_CHOOSE' "$symbols" >&2 || true
      exit 9
    fi
    local csym; csym="$(expected_choose_symbol "$choose")"
    grep -Fq "$csym" "$symbols" || { echo "candidate choose symbol missing: $csym c=$choose p=$primitive" >&2; exit 10; }
  fi

  if [[ "$primitive" == 0 ]]; then
    grep -Fq 'RP_PRIMITIVE' "$symbols" || { echo "baseline primitive symbol missing c=$choose p=$primitive" >&2; exit 11; }
  else
    if grep -Fq 'RP_PRIMITIVE' "$symbols"; then
      echo "legacy RP_PRIMITIVE symbol survived physical mode c=$choose p=$primitive" >&2
      grep -F 'RP_PRIMITIVE' "$symbols" >&2 || true
      exit 12
    fi
    local psym; psym="$(expected_primitive_symbol "$primitive")"
    grep -Fq "$psym" "$symbols" || { echo "candidate primitive symbol missing: $psym c=$choose p=$primitive" >&2; exit 13; }
  fi
}

compiled=0; symbol_audited=0
for choose in 0 1 2 3; do
  for primitive in 0 1 2; do
    out="${PREFIX}_c${choose}_p${primitive}"
    bout="$LOGDIR/c${choose}_p${primitive}.out"
    berr="$LOGDIR/c${choose}_p${primitive}.err"
    echo "=== physical codec compile choose=$choose primitive=$primitive ===" >&2
    TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" \
      "${PTXAS_FLAGS[@]}" \
      -DRP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE="$choose" \
      -DRP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE="$primitive" \
      "$SRC" -o "$out" >"$bout" 2>"$berr"
    [[ -x "$out" ]] || { echo "physical codec compile produced no binary c=$choose p=$primitive" >&2; exit 3; }
    ((compiled += 1))
    if [[ "$SYMBOL_AUDIT" == 1 ]]; then
      audit_symbols "$choose" "$primitive" "$out" "$LOGDIR/c${choose}_p${primitive}.symbols" "$LOGDIR/c${choose}_p${primitive}.symbols.err"
      ((symbol_audited += 1))
    fi
    if [[ "$PTXAS_VERBOSE" == 1 ]]; then
      echo "--- ptxas physical codec c=$choose p=$primitive ---" >&2
      grep -E 'Used .* registers|bytes smem|bytes cmem' "$berr" >&2 || true
    fi
  done
done
[[ "$compiled" == 12 ]] || { echo "unexpected physical compile count=$compiled" >&2; exit 4; }
if [[ "$SYMBOL_AUDIT" == 1 && "$symbol_audited" != 12 ]]; then echo "unexpected physical symbol audit count=$symbol_audited" >&2; exit 4; fi

# Nonzero physical layout + old preinclude remap must be rejected at compile
# time so a benchmark cannot accidentally retain the legacy constant table.
PRE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_codec_tables_sym_u32_preinclude.cuh"
conflict_out="${PREFIX}_conflict_should_not_exist"
if TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" -O2 -std=c++17 -arch="$ARCH" \
    -include "$PRE" \
    -DRP_RUNTIME_CODEC_CHOOSE_U32_MODE=1 \
    -DRP_RUNTIME_CODEC_PRIMITIVE_U32_MODE=1 \
    -DRP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE=1 \
    -DRP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE=1 \
    "$SRC" -o "$conflict_out" >"$LOGDIR/conflict.out" 2>"$LOGDIR/conflict.err"; then
  echo "physical codec compile unexpectedly accepted preinclude conflict" >&2
  exit 5
fi
grep -Eq 'physical (choose|primitive) layout cannot be combined' "$LOGDIR/conflict.err" || {
  echo "physical codec conflict failed for an unexpected reason" >&2
  cat "$LOGDIR/conflict.err" >&2
  exit 6
}

echo "gridfp-codec-table-physical-compile-matrix OK combinations=$compiled symbol_audit=$SYMBOL_AUDIT symbol_audited=$symbol_audited preinclude_conflict_rejected=1 arch=$ARCH exact=1"
