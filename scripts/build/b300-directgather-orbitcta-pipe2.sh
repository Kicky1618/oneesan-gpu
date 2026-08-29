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
lines=src.splitlines(keepends=True)
needle='-DP10DC_ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP"'
hits=[i for i,line in enumerate(lines) if needle in line]
if len(hits)!=1:
    raise SystemExit(f'pipe2 build macro anchor expected one match got {len(hits)}')
i=hits[0]
if not lines[i].rstrip().endswith('\\'):
    raise SystemExit('pipe2 build macro anchor is no longer a continued nvcc line')
lines.insert(i+1,'  -DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2=1 \\\n')
src=''.join(lines)
marker='echo "built $BIN backend=$BACKEND_TAG '
idx=src.find(marker)
if idx<0:
    raise SystemExit('pipe2 build summary anchor missing')
line_end=src.find('\n',idx)
if line_end<0:
    raise SystemExit('pipe2 build summary line end missing')
line=src[idx:line_end]
if 'dynamic_pipe2=' not in line:
    # Add inside the quoted summary when possible; otherwise append an argument.
    q=line.rfind('"')
    if q>0:
        line=line[:q]+' dynamic_pipe2=1'+line[q:]
    else:
        line=line+' dynamic_pipe2=1'
    src=src[:idx]+line+src[line_end:]
Path(sys.argv[2]).write_text(src)
PY
chmod +x "$tmp"
exec env ORBITCTA_FLAT=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_CHUNK=1 \
  ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 bash "$tmp"
