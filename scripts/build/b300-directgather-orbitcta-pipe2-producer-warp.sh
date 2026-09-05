#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PATCH_ONLY="${PIPE2_PRODUCER_PATCH_ONLY:-0}"
QUAD_MLP="${QUAD_MLP:-0}"
COL_ILP="${ORBITCTA_COL_ILP:-4}"
PRODUCER_PRECTX_WARPCOOP="${PRODUCER_PRECTX_WARPCOOP:-0}"
PRODUCER_WORKER_WEIGHT="${PRODUCER_WORKER_WEIGHT:-0}"
PRODUCER_ADAPTIVE_COLS="${PRODUCER_ADAPTIVE_COLS:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"
PRECTX_REVERSE="${PRECTX_REVERSE:-0}"
PRECTX_COMPACT="${PRECTX_COMPACT:-0}"
PRECTX_FLAT_BID="${PRECTX_FLAT_BID:-0}"
PRECTX_FLAT_BID_FUSED="${PRECTX_FLAT_BID_FUSED:-0}"
[[ "$PATCH_ONLY" == 0 || "$PATCH_ONLY" == 1 ]] || { echo 'PIPE2_PRODUCER_PATCH_ONLY must be 0/1' >&2; exit 2; }
[[ "$QUAD_MLP" == 0 || "$QUAD_MLP" == 1 ]] || { echo 'QUAD_MLP must be 0/1' >&2; exit 2; }
[[ "$PRODUCER_PRECTX_WARPCOOP" == 0 || "$PRODUCER_PRECTX_WARPCOOP" == 1 ]] || { echo 'PRODUCER_PRECTX_WARPCOOP must be 0/1' >&2; exit 2; }
case "$PRODUCER_WORKER_WEIGHT" in 0|1|2|3|4) ;; *) echo 'PRODUCER_WORKER_WEIGHT must be 0..4' >&2; exit 2;; esac
[[ "$PRODUCER_ADAPTIVE_COLS" =~ ^[0-9]+$ ]] || { echo 'PRODUCER_ADAPTIVE_COLS must be non-negative integer' >&2; exit 2; }
[[ "$PRECTX_FLAT_BID" == 0 || "$PRECTX_FLAT_BID" == 1 ]] || { echo 'PRECTX_FLAT_BID must be 0/1' >&2; exit 2; }
[[ "$PRECTX_FLAT_BID_FUSED" == 0 || "$PRECTX_FLAT_BID_FUSED" == 1 ]] || { echo 'PRECTX_FLAT_BID_FUSED must be 0/1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT:-1}" == 1 && "${ORBITCTA_FLAT_DYNAMIC:-1}" == 1 && "${ORBITCTA_FLAT_DYNAMIC_PIPE2:-1}" == 1 ]] || { echo 'producer-warp requires flat dynamic pipe2' >&2; exit 2; }
[[ "${ORBITCTA_FLAT_CHUNK:-1}" == 1 ]] || { echo 'producer-warp requires flat chunk=1' >&2; exit 2; }
[[ "${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}" == 0 ]] || { echo 'producer-warp pipe2 requires fuse-lease-prep=0' >&2; exit 2; }
if [[ "$QUAD_MLP" == 1 && "$COL_ILP" != 4 ]]; then echo 'producer-warp native QUAD_MLP=1 requires ORBITCTA_COL_ILP=4' >&2; exit 2; fi
if [[ "$PRODUCER_PRECTX_WARPCOOP" == 1 ]]; then
  [[ "$PRECTX_FORWARD" == 1 && "$PRECTX_REVERSE" == 1 && "$PRECTX_COMPACT" == 1 ]] || { echo 'PRODUCER_PRECTX_WARPCOOP=1 requires compact prectx' >&2; exit 2; }
  [[ "$PRECTX_FLAT_BID_FUSED" == 0 ]] || { echo 'PRODUCER_PRECTX_WARPCOOP=1 currently requires PRECTX_FLAT_BID_FUSED=0' >&2; exit 2; }
fi

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
lines.insert(i+2,'  -DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT="$PRODUCER_WORKER_WEIGHT" \\\n')
lines.insert(i+3,'  -DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS="$PRODUCER_ADAPTIVE_COLS" \\\n')
lines.insert(i+4,'  -DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP="$PRODUCER_PRECTX_WARPCOOP" \\\n')
src=''.join(lines)
marker='echo "built $BIN backend=$BACKEND_TAG '
i=src.find(marker)
if i<0: raise SystemExit('build summary anchor missing')
e=src.find('\n',i); line=src[i:e]; q=line.rfind('"')
line=line[:q]+' pipe2_producer_warp=1 pipe2_producer_worker_weight=$PRODUCER_WORKER_WEIGHT pipe2_producer_prectx_warpcoop=$PRODUCER_PRECTX_WARPCOOP pipe2_producer_adaptive_cols=$PRODUCER_ADAPTIVE_COLS'+line[q:]
src=src[:i]+line+src[e:]
Path(sys.argv[2]).write_text(src)
PY
chmod +x "$tmp"
if [[ "$PATCH_ONLY" == 1 ]]; then
  bash -n "$tmp"
  grep -Fq -- '-DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP=1' "$tmp" || exit 3
  grep -Fq -- '-DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT="$PRODUCER_WORKER_WEIGHT"' "$tmp" || exit 3
  grep -Fq -- '-DP10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS="$PRODUCER_ADAPTIVE_COLS"' "$tmp" || exit 3
  grep -Fq 'pipe2_producer_warp=1' "$tmp" || exit 3
  echo "b300_pipe2_producer_warp_patch=OK quad_mlp=$QUAD_MLP col_ilp=$COL_ILP producer_worker_weight=$PRODUCER_WORKER_WEIGHT producer_prectx_warpcoop=$PRODUCER_PRECTX_WARPCOOP producer_adaptive_cols=$PRODUCER_ADAPTIVE_COLS gpu_work=0"
  exit 0
fi
exec env ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC=1 \
  ORBITCTA_COL_ILP="$COL_ILP" QUAD_MLP="$QUAD_MLP" \
  PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" PRECTX_COMPACT="$PRECTX_COMPACT" \
  PRECTX_FLAT_BID="$PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$PRECTX_FLAT_BID_FUSED" \
  PRODUCER_PRECTX_WARPCOOP="$PRODUCER_PRECTX_WARPCOOP" PRODUCER_WORKER_WEIGHT="$PRODUCER_WORKER_WEIGHT" PRODUCER_ADAPTIVE_COLS="$PRODUCER_ADAPTIVE_COLS" \
  ORBITCTA_FLAT_DYNAMIC_PIPE2=1 ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 bash "$tmp"
