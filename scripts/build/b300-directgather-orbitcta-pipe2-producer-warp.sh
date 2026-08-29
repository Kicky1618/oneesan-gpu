#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PATCH_ONLY="${PIPE2_PRODUCER_PATCH_ONLY:-0}"
[[ "$PATCH_ONLY" == 0 || "$PATCH_ONLY" == 1 ]] || { echo 'PIPE2_PRODUCER_PATCH_ONLY must be 0/1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT:-1}" == 1 && "${ORBITCTA_FLAT_DYNAMIC:-1}" == 1 && "${ORBITCTA_FLAT_DYNAMIC_PIPE2:-1}" == 1 ]] || {
  echo 'producer-warp requires flat dynamic pipe2' >&2; exit 2;
}
[[ "${ORBITCTA_FLAT_CHUNK:-1}" == 1 ]] || { echo 'producer-warp requires flat chunk=1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}" == 0 ]] || { echo 'producer-warp pipe2 requires fuse-lease-prep=0' >&2; exit 2; }

base="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh"
tmp="$ONEESAN_ROOT/scripts/build/.b300-directgather-orbitcta-pipe2-producer.$$.sh"
trap 'rm -f "$tmp"' EXIT
python3 - "$base" "$tmp" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(); lines=src.splitlines(keepends=True)
needle='-DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2="$ORBITCTA_FLAT_DYNAMIC_PIPE2"'
hits=[i for i,l in enumerate(lines) if needle in l]
if len(hits)!=1: raise SystemExit(f'producer macro anchor expected one match got {len(hits)}')
i=hits[0]
if not lines[i].rstrip().endswith('\\'): raise SystemExit('producer macro anchor not continued nvcc line')
lines.insert(i+1,'  -DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP=1 \\\n')
src=''.join(lines)
marker='echo "built $BIN backend=$BACKEND_TAG '
i=src.find(marker)
if i<0: raise SystemExit('build summary anchor missing')
e=src.find('\n',i); line=src[i:e]; q=line.rfind('"')
if q<0: raise SystemExit('build summary quote missing')
line=line[:q]+' pipe2_producer_warp=1'+line[q:]
src=src[:i]+line+src[e:]
Path(sys.argv[2]).write_text(src)
PY
chmod +x "$tmp"
if [[ "$PATCH_ONLY" == 1 ]]; then
  bash -n "$tmp"
  grep -Fq -- '-DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP=1' "$tmp" || exit 3
  grep -Fq 'pipe2_producer_warp=1' "$tmp" || exit 3
  echo 'b300_pipe2_producer_warp_patch=OK gpu_work=0'
  exit 0
fi
exec env ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC=1 \
  ORBITCTA_FLAT_DYNAMIC_PIPE2=1 ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 bash "$tmp"
