#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

[[ "${ORBITCTA_FLAT:-1}" == 1 ]] || { echo 'pipe2 requires ORBITCTA_FLAT=1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT_DYNAMIC:-1}" == 1 ]] || { echo 'pipe2 requires ORBITCTA_FLAT_DYNAMIC=1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT_CHUNK:-1}" == 1 ]] || { echo 'pipe2 requires ORBITCTA_FLAT_CHUNK=1' >&2; exit 2; }
# The old lease/first-prepare fusion and pipe2 both optimize the same boundary.
# Isolate them for the first B300 A/B instead of silently stacking both.
[[ "${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}" == 0 ]] || {
  echo 'pipe2 A/B requires ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0' >&2; exit 2;
}

base="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh"
tmp="$ONEESAN_ROOT/scripts/build/.b300-directgather-orbitcta-pipe2.$$.sh"
trap 'rm -f "$tmp"' EXIT
python3 - "$base" "$tmp" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text()
needle='  -DP10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP" \\\n'
if src.count(needle)!=1:
    raise SystemExit(f'pipe2 build macro anchor expected one match got {src.count(needle)}')
src=src.replace(needle,needle+'  -DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2=1 \\\n',1)
# Keep the base script authoritative; only append a marker to its existing build
# summary so stale binaries/configurations are easy to reject in A/B scripts.
marker='echo "built $BIN backend=$BACKEND_TAG '
idx=src.find(marker)
if idx<0:
    raise SystemExit('pipe2 build summary anchor missing')
line_end=src.find('\n',idx)
src=src[:line_end]+r' dynamic_pipe2=1'+src[line_end:]
Path(sys.argv[2]).write_text(src)
PY
chmod +x "$tmp"
exec env ORBITCTA_FLAT=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_CHUNK=1 \
  ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 bash "$tmp"
