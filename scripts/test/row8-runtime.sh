#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=9
MOD=4294967291
EXPECTED=2674633373
ARCH="${ARCH:-native}"
BIN="$ONEESAN_BUILD_DIR/row8_runtime_regression_n${N}"
CACHE_DIR="$ONEESAN_BUILD_DIR/row8-runtime-test-cache"
LOG_DIR="$ONEESAN_BUILD_DIR/row8-runtime-test-logs"
mkdir -p "$LOG_DIR"
rm -rf "$CACHE_DIR"

ROW8_TENSOR=1 N="$N" ARCH="$ARCH" OUT="$BIN" \
  "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >/dev/null

run_one() {
  local tag="$1"
  GRIDFP_ROW8_CACHE_DIR="$CACHE_DIR" \
  GRIDFP_ROW8_VALIDATE_BUILD=1 \
  GRIDFP_BOUNDED_PREFIX_K=8 \
  GRIDFP_DIRECT_ROW8_TENSOR=1 \
  GRIDFP_VRAM_RESERVE_MIB=128 \
    "$BIN" "$N" 256 8 1 "$MOD" \
    >"$LOG_DIR/$tag.out" 2>"$LOG_DIR/$tag.err"
  local got
  got="$(sed -n 's/.* residue=\([0-9][0-9]*\) modulus=.*/\1/p' "$LOG_DIR/$tag.out" | tail -1)"
  if [[ "$got" != "$EXPECTED" ]]; then
    echo "row8 $tag residue mismatch: got=${got:-<none>} expected=$EXPECTED" >&2
    cat "$LOG_DIR/$tag.out" >&2
    cat "$LOG_DIR/$tag.err" >&2
    exit 1
  fi
}

run_one cold
if ! grep -q 'row8 runtime mod build' "$LOG_DIR/cold.err"; then
  echo 'row8 cold run did not build a runtime modular table' >&2
  exit 1
fi

mapfile -t caches < <(find "$CACHE_DIR" -maxdepth 1 -type f -name 'row8_mod_*.bin' -print)
if (( ${#caches[@]} != 1 )); then
  echo "expected one row8 cache, found ${#caches[@]}" >&2
  exit 1
fi
cache="${caches[0]}"

run_one warm
if ! grep -q 'row8 mod cache hit' "$LOG_DIR/warm.err"; then
  echo 'row8 warm run did not hit modular cache' >&2
  exit 1
fi

python3 - "$cache" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
b=bytearray(p.read_bytes())
if len(b) < 1024:
    raise SystemExit('row8 cache unexpectedly small')
b[-257] ^= 0x5A
p.write_bytes(b)
PY
run_one corrupt
if grep -q 'row8 mod cache hit' "$LOG_DIR/corrupt.err"; then
  echo 'row8 corrupt cache was incorrectly accepted' >&2
  exit 1
fi
if ! grep -q 'row8 runtime mod build' "$LOG_DIR/corrupt.err"; then
  echo 'row8 corrupt cache was not rebuilt' >&2
  exit 1
fi

echo "row8 runtime regression: PASS n=$N mod=$MOD residue=$EXPECTED"
