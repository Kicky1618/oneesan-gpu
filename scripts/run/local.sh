#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/common.sh"

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

for spec in "N:$N" "MOD:$MOD" "NGPU:$NGPU" "TARGET_MIB:$TARGET_MIB" "GRIDFP_VRAM_RESERVE_MIB:$GRIDFP_VRAM_RESERVE_MIB"; do
  require_uint "${spec%%:*}" "${spec#*:}" || exit 2
done

if (( NGPU != 1 )); then
  echo "warning: local.sh is intended for one GPU; NGPU=$NGPU" >&2
fi

run_with_gpu_energy() {
  local power_log power_pid start_ns end_ns rc gpu_ids
  power_log="$(mktemp)"
  power_pid=""

  cleanup_power_meter() {
    if [[ -n "$power_pid" ]]; then
      kill "$power_pid" 2>/dev/null || true
      wait "$power_pid" 2>/dev/null || true
    fi
    rm -f "$power_log"
  }
  trap cleanup_power_meter EXIT

  if command -v nvidia-smi >/dev/null; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      gpu_ids="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | cut -d, -f1-"$NGPU")"
    else
      gpu_ids="$(seq -s, 0 $((NGPU - 1)))"
    fi

    nvidia-smi \
      --id="$gpu_ids" \
      --query-gpu=power.draw \
      --format=csv,noheader,nounits \
      -lms 200 >"$power_log" 2>/dev/null &
    power_pid=$!
  fi

  start_ns="$(date +%s%N)"
  set +e
  "$@"
  rc=$?
  set -e
  end_ns="$(date +%s%N)"

  if [[ -n "$power_pid" ]]; then
    kill "$power_pid" 2>/dev/null || true
    wait "$power_pid" 2>/dev/null || true
    power_pid=""
  fi

  awk -v elapsed_ns="$((end_ns - start_ns))" -v gpu_count="$NGPU" '
    /^[[:space:]]*[0-9]+([.][0-9]+)?[[:space:]]*$/ {
      sum += $1
      count++
    }
    END {
      if (count > 0) {
        seconds = elapsed_ns / 1000000000.0
        avg_w = (sum / count) * gpu_count
        wh = avg_w * seconds / 3600.0
        printf "GPU energy: %.3f Wh (avg %.1f W, %.3f s)\n", wh, avg_w, seconds
      } else {
        print "GPU energy: unavailable (no power samples)" > "/dev/stderr"
      }
    }
  ' "$power_log"

  trap - EXIT
  rm -f "$power_log"
  return "$rc"
}

case "$MODE" in
  residue)
    if (( ${#EXTRA[@]} != 0 )); then
      echo "unexpected arguments for residue mode: ${EXTRA[*]}" >&2
      echo "use --exact before exact-run options such as --max-runs" >&2
      exit 2
    fi
    run_with_gpu_energy "$SCRIPT_DIR/b300x8.sh" "$N" "$MOD"
    ;;
  exact)
    run_with_gpu_energy "$SCRIPT_DIR/b300x8-exact.sh" "$N" "${EXTRA[@]}"
    ;;
esac
