#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GUARD="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-prefetch-guard-staged-calibrate.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-load-policy-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-load-policy-staged-calibrate.sh"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-mate-load-policy.sh"
TRANSFORM_PREFLIGHT="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-mate-load-policy-preflight.sh"
for f in "$GUARD" "$SWEEP" "$STAGED" "$BUILDER" "$TRANSFORM_PREFLIGHT"; do
  [[ -f "$f" ]] || { echo "missing Stage-M dependency=$f" >&2; exit 2; }
  bash -n "$f"
done

need(){
  local file="$1" marker="$2" label="$3"
  grep -Fq "$marker" "$file" || { echo "$label marker missing: $marker" >&2; exit 3; }
}

# Stage L remains exclusively the prefetch-guard stage.
for s in \
  'B300_STAGEL_FINAL_SELF_GUARD=' \
  'B300_STAGEL_FINAL_MATE_GUARD=' \
  'B300_STAGEL_FINAL_BIN=' \
  'B300_STAGEL_FINAL_STAGE_RESIDUE='; do
  need "$GUARD" "$s" stage-l-guard
 done
if grep -Fq 'B300_STAGEM_' "$GUARD"; then
  echo 'Stage-L guard script writes Stage-M namespace' >&2
  exit 3
fi

# Stage M reads Stage L as immutable upstream state and writes only STAGEM.
for f in "$SWEEP" "$STAGED"; do
  need "$f" 'STAGEL_GUARD_ENV=' stage-m-upstream
  need "$f" 'B300_STAGEL_FINAL_SELF_GUARD' stage-m-upstream
  need "$f" 'B300_STAGEL_FINAL_MATE_GUARD' stage-m-upstream
  need "$f" 'B300_STAGEM_' stage-m-namespace
  if grep -Eq "printf '[^']*B300_STAGEL_|'B300_STAGEL_[A-Z0-9_]+':" "$f"; then
    echo "Stage-M script writes Stage-L namespace: $f" >&2
    exit 3
  fi
done

for s in \
  'SELF_GUARD="$SG" MATE_GUARD="$MG"' \
  'self_geometry width=$SW distance=$SD evict=$SE guard=$SG' \
  'mate_geometry width=$MW distance=$MD evict=$ME guard=$MG' \
  'guard_policy_preserved=1' \
  'b300_stagem_exact_match=1' \
  'b300_stagem_ngpu=' \
  "'B300_STAGEM_SELF_GUARD':sg" \
  "'B300_STAGEM_MATE_GUARD':mg"; do
  need "$SWEEP" "$s" stage-m-sweep
 done

for s in \
  'check_configuration' \
  'FATAL Stage-M upstream configuration drift' \
  'FATAL Stage-M/Stage-L residue mismatch' \
  'B300_STAGEM_STAGED_VALIDATED=' \
  'B300_STAGEM_FINAL_ENABLED=' \
  'B300_STAGEM_INPUT_STAGEL_GUARD_ENV=' \
  'B300_STAGEM_FINAL_STAGE_RESIDUE=' \
  'b300_stagem_exact_match=1'; do
  need "$STAGED" "$s" stage-m-staged
 done

for s in \
  'SELF_GUARD="${SELF_GUARD:-branch}"' \
  'MATE_GUARD="${MATE_GUARD:-branch}"' \
  'SELF_GUARD="$SELF_GUARD" MATE_GUARD="$MATE_GUARD"' \
  'guard_policy_preserved=1'; do
  need "$BUILDER" "$s" stage-m-builder
 done

bash "$TRANSFORM_PREFLIGHT" >/dev/null

echo 'b300_stagem_preflight=OK stage_l_namespace=guard_only stage_m_namespace=mate_load_only upstream_guard_preserved=1 geometry_preserved=1 eviction_preserved=1 staged_residue_gate=1 spill_gate=1 gpu_work=0'
