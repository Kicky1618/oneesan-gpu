#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GUARD="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-prefetch-guard-staged-calibrate.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-load-policy-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-mate-load-policy-staged-calibrate.sh"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-mate-load-policy.sh"
TRANSFORM_PREFLIGHT="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-mate-load-policy-preflight.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh"
for f in "$GUARD" "$SWEEP" "$STAGED" "$BUILDER" "$TRANSFORM_PREFLIGHT" "$PROMOTE"; do
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

# Promotion must bind to the exact Stage-L prepared modulus, GPU count and
# manifest. It may change only the load policy and must use Stage L as control.
for s in \
  'B300_STAGEL_PREPARED_MOD' \
  'B300_STAGEL_PREPARED_NGPU' \
  'B300_STAGEL_PREPARED_MANIFEST' \
  'Stage-M/Stage-L modulus mismatch' \
  'Stage-M/Stage-L GPU count mismatch' \
  'sha256sum -c "$B300_STAGEL_PREPARED_MANIFEST"' \
  'Stage-M changed Stage-L geometry/eviction/guard policy' \
  'Stage-M control is not exact Stage-L prepared binary' \
  'B300_STAGEM_PROMOTION_VALIDATED=1' \
  'B300_STAGEM_PROMOTION_STAGEL_MANIFEST_SHA256=' \
  'B300_STAGEM_PREPARED=1' \
  'B300_STAGEM_PREPARED_POLICY=' \
  'B300_STAGEM_PREPARED_MANIFEST=' \
  'B300_STAGEM_PREPARED_FINAL_STAGE_RESIDUE=' \
  'Stage-M promotion fingerprint mismatch'; do
  need "$PROMOTE" "$s" stage-m-promotion
done

python3 - "$SWEEP" "$STAGED" "$PROMOTE" <<'PY'
from pathlib import Path
import sys
sweep=Path(sys.argv[1]).read_text(); staged=Path(sys.argv[2]).read_text(); promote=Path(sys.argv[3]).read_text()
if 'Stage L owns the exact geometry' not in sweep or 'Stage M is allowed to change only' not in sweep:
    raise SystemExit('Stage-M upstream ownership contract missing')
if "printf 'default\\t%s\\t-\\n' \"$CONTROL_BIN\"" not in sweep:
    raise SystemExit('Stage-M default control must be exact Stage-L binary')
if 'run_stage "$SEARCH_ROWS" "$POLICY_LIST"' not in staged:
    raise SystemExit('Stage-M broad search missing')
if 'run_stage "$rows" "default $SELECTED_POLICY"' not in staged:
    raise SystemExit('Stage-M validation must lock one selected policy')
for marker in ('[[ "$rows" == "$L_ROWS"', '[[ "$rows" == "$F_ROWS"', '[[ "$rows" == "$E_ROWS"'):
    if marker not in staged: raise SystemExit('row-scoped residue gate missing: '+marker)
if staged.count('check_configuration') < 3:
    raise SystemExit('Stage-M configuration-lock checks missing')
if 'PREPARE_ONLY=1' not in promote and 'PREPARE_ONLY="${PREPARE_ONLY:-0}"' not in promote:
    raise SystemExit('Stage-M prepare-only path missing')
print('stagem_contract_structure=OK')
PY

bash "$TRANSFORM_PREFLIGHT" >/dev/null

echo 'b300_stagem_preflight=OK stage_l_namespace=guard_only stage_m_namespace=mate_load_only upstream_guard_preserved=1 geometry_preserved=1 eviction_preserved=1 staged_residue_gate=1 spill_gate=1 policy_lock=1 promotion_manifest=1 modulus_bound=1 gpu_count_bound=1 gpu_work=0'
