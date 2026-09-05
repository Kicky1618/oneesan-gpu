#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-sm_80}"
SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/rankmask5_decode_microbench.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/rankmask5_decode_ptxas_${ARCH}}"
LOG="${LOG:-$ONEESAN_BUILD_DIR/rankmask5_decode_ptxas_${ARCH}.log}"
mkdir -p "$(dirname "$BIN")" "$(dirname "$LOG")"

if ! command -v nvcc >/dev/null; then
  echo "nvcc is required" >&2
  exit 2
fi

# One compile instantiates all four template kernels. ptxas names preserve the
# template integer (Li0E..Li3E), letting us compare register/spill pressure
# without four separate builds or any GPU execution.
set +e
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -Xptxas=-v "$SRC" -o "$BIN" >"$LOG.out" 2>"$LOG"
rc=$?
set -e
if (( rc != 0 )); then
  cat "$LOG" >&2
  exit "$rc"
fi

cat "$LOG"

python3 - "$LOG" <<'PY'
import re
import sys

path = sys.argv[1]
lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
mode_name = {0: 'ffs', 1: 'unrolled5', 2: 'direct3', 3: 'direct3_guard'}
rows = {m: {'mode': mode_name[m]} for m in mode_name}
current = None

for line in lines:
    if 'rankmask5_kernel' in line:
        m = re.search(r'Li([0-3])E', line)
        if m:
            current = int(m.group(1))
    if current is None:
        continue
    m = re.search(r'Used\s+(\d+)\s+registers', line)
    if m:
        rows[current]['registers'] = int(m.group(1))
    m = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', line)
    if m:
        rows[current]['spill_stores'] = int(m.group(1))
        rows[current]['spill_loads'] = int(m.group(2))

missing = [mode_name[m] for m, r in rows.items() if 'registers' not in r]
if missing:
    raise SystemExit('failed to parse ptxas registers for: ' + ','.join(missing))

print('mode\tregisters\tspill_stores\tspill_loads')
for m in range(4):
    r = rows[m]
    print(f"{r['mode']}\t{r['registers']}\t{r.get('spill_stores', 0)}\t{r.get('spill_loads', 0)}")

base = rows[2]['registers']
guard = rows[3]['registers']
print(f'direct3_guard_register_delta={guard-base:+d}')
print(f'direct3_guard_same_registers={int(guard == base)}')
print(f'all_modes_spill_free={int(all(r.get("spill_stores",0)==0 and r.get("spill_loads",0)==0 for r in rows.values()))}')
PY

echo "rankmask5-decode-ptxas OK arch=$ARCH log=$LOG" >&2
