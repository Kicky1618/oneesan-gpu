#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

N="18"
MODE="residue"
MOD="4294967291"
EXTRA=()

if [[ $# -gt 0 && "$1" != --* ]]; then
  N="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exact)
      MODE="exact"
      shift
      ;;
    --mod)
      MOD="${2:?--mod requires a value}"
      shift 2
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

export ARCH="${ARCH:-native}"
export NGPU="${NGPU:-1}"
export TARGET_MIB="${TARGET_MIB:-512}"
export GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-1024}"

if (( NGPU != 1 )); then
  echo "warning: local.sh is intended for one GPU; NGPU=$NGPU" >&2
fi

case "$MODE" in
  residue)
    if (( ${#EXTRA[@]} != 0 )); then
      echo "unexpected arguments for residue mode: ${EXTRA[*]}" >&2
      echo "use --exact before exact-run options such as --max-runs" >&2
      exit 2
    fi
    exec "$SCRIPT_DIR/b300x8.sh" "$N" "$MOD"
    ;;
  exact)
    exec "$SCRIPT_DIR/b300x8-exact.sh" "$N" "${EXTRA[@]}"
    ;;
esac
