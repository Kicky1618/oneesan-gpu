#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; ARCH="${ARCH:-sm_103}"
RUN_PLAN="${RUN_PLAN:-0}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
for x in RUN_PLAN; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }

bash -n "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh"
bash -n "$ONEESAN_ROOT/scripts/build/b300-rankformula-hbm-max.sh"
bash -n "$ONEESAN_ROOT/scripts/bench/b300-cpasync-local-pair-ab.sh"
bash -n "$ONEESAN_ROOT/scripts/bench/b300-cpasync-local-pipeline-ab.sh"

mkdir -p "$ONEESAN_ROOT/work/b300_cpasync_local_preflight"
for mode in cross local overlap; do
  case "$mode" in
    cross) local_cpa=0; overlap_cpa=0 ;;
    local) local_cpa=1; overlap_cpa=0 ;;
    overlap) local_cpa=0; overlap_cpa=1 ;;
  esac
  bin="$ONEESAN_BUILD_DIR/b300_cpasync_local_preflight_${mode}_n${N}"
  echo "=== compile $mode ===" >&2
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP=2 PM_ACCUM=1 \
    DEPTHMAJOR=1 PAIR_MLP=1 QUAD_MLP=0 MLP_WINDOW4=1 CPASYNC_PAIR=1 \
    CPASYNC_LOCAL_PAIR="$local_cpa" CPASYNC_OVERLAP_LOCAL_PAIR="$overlap_cpa" \
    DIRECTGATHER64=1 DIRECTGATHER_SPARSE64=1 SORTED=1 \
    FORCE7=0 PREFETCH_NEXT=0 PRECTX_FORWARD=0 PRECTX_REVERSE=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
    >"$ONEESAN_ROOT/work/b300_cpasync_local_preflight/${mode}.build.out" \
    2>"$ONEESAN_ROOT/work/b300_cpasync_local_preflight/${mode}.build.err"
  [[ -x "$bin" ]] || { echo "$mode binary missing: $bin" >&2; exit 3; }
  if [[ "$RUN_PLAN" == 1 ]]; then
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" 8 --plan-only \
      >"$ONEESAN_ROOT/work/b300_cpasync_local_preflight/${mode}.plan.out" \
      2>"$ONEESAN_ROOT/work/b300_cpasync_local_preflight/${mode}.plan.err"
  fi
done

echo "b300-cpasync-local-pipeline-preflight OK n=$N arch=$ARCH run_plan=$RUN_PLAN" >&2
