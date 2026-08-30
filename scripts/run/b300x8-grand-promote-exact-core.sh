#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'grand exact promotion currently targets n=27' >&2; exit 2; }

FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
ALLOW_HEAD_DRIFT="${ALLOW_HEAD_DRIFT:-0}"
ALLOW_DIRTY_FIRSTPASS="${ALLOW_DIRTY_FIRSTPASS:-0}"
ALLOW_WORKTREE_DIRTY="${ALLOW_WORKTREE_DIRTY:-0}"
DRY_RUN="${DRY_RUN:-0}"
for x in ALLOW_HEAD_DRIFT ALLOW_DIRTY_FIRSTPASS ALLOW_WORKTREE_DIRTY DRY_RUN; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done

[[ -s "$SELECTED_ENV" ]] || { echo "missing first-pass selection contract: $SELECTED_ENV" >&2; exit 2; }
# shellcheck disable=SC1090
source "$SELECTED_ENV"
required=(
  B300_GRAND_SELECTED_SCHEMA B300_GRAND_SELECTED_VALIDATED B300_GRAND_SELECTED_N
  B300_GRAND_SELECTED_HEAD_SHA B300_GRAND_SELECTED_HEAD_DIRTY
  B300_GRAND_SELECTED_PROFILE_FILE B300_GRAND_SELECTED_PROFILE_SHA256
  B300_GRAND_SELECTED_BACKEND B300_GRAND_SELECTED_PROFILE
  B300_GRAND_SELECTED_BINARY B300_GRAND_SELECTED_BINARY_SHA256
  B300_GRAND_SELECTED_RESIDUE B300_GRAND_SELECTED_WALL_S B300_GRAND_SELECTED_SMOKE_PRIME
  B300_GRAND_SELECTED_RUNTIME_KIND B300_GRAND_SELECTED_THREADS B300_GRAND_SELECTED_TARGET_MIB
  B300_GRAND_SELECTED_MAX_WINDOW B300_GRAND_SELECTED_WORK_DIR B300_GRAND_SELECTED_CHECKPOINT
  B300_GRAND_SELECTED_RACE_RESULT B300_GRAND_SELECTED_RACE_RESULT_SHA256
  B300_GRAND_SELECTED_FIRSTPASS_META
)
for key in "${required[@]}"; do
  [[ -n "${!key+x}" ]] || { echo "selection contract missing $key" >&2; exit 3; }
done
[[ "$B300_GRAND_SELECTED_SCHEMA" =~ ^[1-9][0-9]*$ ]] || { echo "bad selection schema=$B300_GRAND_SELECTED_SCHEMA" >&2; exit 3; }
SELECTION_SCHEMA="$B300_GRAND_SELECTED_SCHEMA"
(( SELECTION_SCHEMA >= 1 && SELECTION_SCHEMA <= 3 )) || { echo "unsupported grand selection schema=$SELECTION_SCHEMA" >&2; exit 3; }
[[ "$B300_GRAND_SELECTED_VALIDATED" == 1 && "$B300_GRAND_SELECTED_N" == 27 ]] || {
  echo 'invalid grand first-pass selection contract' >&2; exit 3;
}

# Schema 2 adds a hashed grand-summary snapshot. Schema 3 additionally proves
# that Stage I/J/K were staged before exactly one complete-prime race.
GRAND_SUMMARY_ENV=""
GRAND_SUMMARY_SHA=""
if (( SELECTION_SCHEMA >= 2 )); then
  for key in B300_GRAND_SELECTED_GRAND_SUMMARY_ENV B300_GRAND_SELECTED_GRAND_SUMMARY_SHA256; do
    [[ -n "${!key+x}" ]] || { echo "selection schema $SELECTION_SCHEMA missing $key" >&2; exit 3; }
  done
  GRAND_SUMMARY_ENV="$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV"
  GRAND_SUMMARY_SHA="$B300_GRAND_SELECTED_GRAND_SUMMARY_SHA256"
  [[ -s "$GRAND_SUMMARY_ENV" ]] || { echo "grand summary missing: $GRAND_SUMMARY_ENV" >&2; exit 3; }
fi
if (( SELECTION_SCHEMA >= 3 )); then
  [[ "${B300_GRAND_SELECTED_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {
    echo 'schema-3 selection does not prove exactly one complete-prime race' >&2; exit 3;
  }
  for key in B300_GRAND_SELECTED_STAGEI_ACCEPTED B300_GRAND_SELECTED_STAGEJ_ACCEPTED B300_GRAND_SELECTED_STAGEK_ACCEPTED; do
    [[ -n "${!key+x}" ]] || { echo "schema-3 selection missing $key" >&2; exit 3; }
    [[ "${!key}" == 0 || "${!key}" == 1 ]] || { echo "bad $key=${!key}" >&2; exit 3; }
  done
fi

case "$B300_GRAND_SELECTED_RUNTIME_KIND" in forced|warp|orbit) ;; *) echo "bad runtime kind=$B300_GRAND_SELECTED_RUNTIME_KIND" >&2; exit 3;; esac
for x in B300_GRAND_SELECTED_TARGET_MIB B300_GRAND_SELECTED_MAX_WINDOW B300_GRAND_SELECTED_SMOKE_PRIME; do
  v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "bad $x=$v" >&2; exit 3; }
done
[[ -x "$B300_GRAND_SELECTED_BINARY" ]] || { echo "selected binary missing: $B300_GRAND_SELECTED_BINARY" >&2; exit 3; }
[[ -s "$B300_GRAND_SELECTED_PROFILE_FILE" ]] || { echo "selected profile missing: $B300_GRAND_SELECTED_PROFILE_FILE" >&2; exit 3; }
[[ -s "$B300_GRAND_SELECTED_RACE_RESULT" ]] || { echo "selected race result missing: $B300_GRAND_SELECTED_RACE_RESULT" >&2; exit 3; }
[[ -s "$B300_GRAND_SELECTED_CHECKPOINT" ]] || { echo "selected checkpoint missing: $B300_GRAND_SELECTED_CHECKPOINT" >&2; exit 3; }
[[ -s "$B300_GRAND_SELECTED_FIRSTPASS_META" ]] || { echo "first-pass meta missing: $B300_GRAND_SELECTED_FIRSTPASS_META" >&2; exit 3; }

command -v git >/dev/null || { echo 'git required' >&2; exit 2; }
command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }
CURRENT_HEAD="$(git -C "$ONEESAN_ROOT" rev-parse HEAD)"
CURRENT_DIRTY=0
[[ -z "$(git -C "$ONEESAN_ROOT" status --porcelain=v1 --untracked-files=normal)" ]] || CURRENT_DIRTY=1
if [[ "$B300_GRAND_SELECTED_HEAD_DIRTY" != 0 && "$ALLOW_DIRTY_FIRSTPASS" != 1 ]]; then
  echo 'first-pass provenance was dirty; set ALLOW_DIRTY_FIRSTPASS=1 only after auditing local changes' >&2
  exit 3
fi
if [[ "$CURRENT_HEAD" != "$B300_GRAND_SELECTED_HEAD_SHA" && "$ALLOW_HEAD_DRIFT" != 1 ]]; then
  echo "repository HEAD drifted since first-pass: selected=$B300_GRAND_SELECTED_HEAD_SHA current=$CURRENT_HEAD" >&2
  echo 'rerun first-pass, or set ALLOW_HEAD_DRIFT=1 only when the selected binary/profile and solver changes were audited' >&2
  exit 3
fi
if (( CURRENT_DIRTY )) && [[ "$ALLOW_WORKTREE_DIRTY" != 1 ]]; then
  echo 'current worktree is dirty; refusing exact promotion without ALLOW_WORKTREE_DIRTY=1' >&2
  exit 3
fi

BIN_SHA="$(sha256sum "$B300_GRAND_SELECTED_BINARY" | awk '{print $1}')"
PROFILE_SHA="$(sha256sum "$B300_GRAND_SELECTED_PROFILE_FILE" | awk '{print $1}')"
RACE_SHA="$(sha256sum "$B300_GRAND_SELECTED_RACE_RESULT" | awk '{print $1}')"
[[ "$BIN_SHA" == "$B300_GRAND_SELECTED_BINARY_SHA256" ]] || { echo 'selected binary fingerprint mismatch' >&2; exit 4; }
[[ "$PROFILE_SHA" == "$B300_GRAND_SELECTED_PROFILE_SHA256" ]] || { echo 'selected profile fingerprint mismatch' >&2; exit 4; }
[[ "$RACE_SHA" == "$B300_GRAND_SELECTED_RACE_RESULT_SHA256" ]] || { echo 'single-pass TSV fingerprint mismatch' >&2; exit 4; }
if (( SELECTION_SCHEMA >= 2 )); then
  ACTUAL_GRAND_SUMMARY_SHA="$(sha256sum "$GRAND_SUMMARY_ENV" | awk '{print $1}')"
  [[ "$ACTUAL_GRAND_SUMMARY_SHA" == "$GRAND_SUMMARY_SHA" ]] || { echo 'grand summary fingerprint mismatch' >&2; exit 4; }
  # shellcheck disable=SC1090
  source "$GRAND_SUMMARY_ENV"
  [[ "${B300_GRAND_PREPARED:-0}" == 1 ]] || { echo 'grand summary prepared marker missing' >&2; exit 4; }
  if (( SELECTION_SCHEMA >= 3 )); then
    [[ "${B300_GRAND_STAGEI_NAMESPACE_ISOLATED:-0}" == 1 && "${B300_GRAND_STAGEJ_INTEGRATED:-0}" == 1 && \
       "${B300_GRAND_STAGEK_INTEGRATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || {
      echo 'schema-3 grand summary Stage-I/J/K single-race proof missing' >&2; exit 4;
    }
  fi
fi

python3 - "$B300_GRAND_SELECTED_RACE_RESULT" "$B300_GRAND_SELECTED_BACKEND" "$B300_GRAND_SELECTED_PROFILE" \
  "$B300_GRAND_SELECTED_BINARY" "$B300_GRAND_SELECTED_RESIDUE" "$B300_GRAND_SELECTED_WALL_S" <<'PY'
import csv,sys
path,backend,profile,binary,residue,wall=sys.argv[1:]
r=list(csv.DictReader(open(path,encoding='utf-8'),delimiter='\t'))
ok=[x for x in r if x.get('status')=='ok']
if not ok: raise SystemExit('race TSV has no successful candidates')
if len({x['residue'] for x in ok}) != 1: raise SystemExit('race TSV residues no longer agree')
b=min(ok,key=lambda x:float(x['wall_s']))
expected=(backend,profile,binary,residue,wall)
actual=(b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s'])
if actual != expected: raise SystemExit(f'race winner contract mismatch: actual={actual!r} expected={expected!r}')
PY

python3 - "$B300_GRAND_SELECTED_CHECKPOINT" "$B300_GRAND_SELECTED_BINARY_SHA256" \
  "$B300_GRAND_SELECTED_PROFILE_SHA256" "$B300_GRAND_SELECTED_SMOKE_PRIME" "$B300_GRAND_SELECTED_RESIDUE" <<'PY'
import json,sys
cp,bsha,psha,prime,residue=sys.argv[1:]
d=json.load(open(cp))
if int(d.get('n',-1)) != 27: raise SystemExit('checkpoint n mismatch')
fp=d.get('solver_fingerprint',{})
if fp != {'schema':3,'binary_sha256':bsha,'profile_sha256':psha}:
    raise SystemExit(f'checkpoint fingerprint mismatch: {fp!r}')
r=d.get('residues',{}).get(str(int(prime)))
if not r or int(r.get('residue',-1)) != int(residue):
    raise SystemExit('checkpoint does not contain the selected smoke residue')
PY

unset GRIDFP_THREADS BUCKET_THREADS BUCKET_GRID_X BUCKET_GRID_Y BUCKET_LOW_GRID_X BUCKET_LOW_GRID_Y \
  BUCKET_ORBITCTA_GRID_Y BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM || true

# shellcheck disable=SC1090
source "$B300_GRAND_SELECTED_PROFILE_FILE"
case "$B300_GRAND_SELECTED_RUNTIME_KIND" in
  forced)
    [[ "$B300_GRAND_SELECTED_THREADS" =~ ^[0-9]+$ ]] && \
      ((B300_GRAND_SELECTED_THREADS>=32 && B300_GRAND_SELECTED_THREADS<=1024 && B300_GRAND_SELECTED_THREADS%32==0)) || {
        echo "bad forced threads=$B300_GRAND_SELECTED_THREADS" >&2; exit 4;
      }
    export GRIDFP_THREADS="$B300_GRAND_SELECTED_THREADS"
    ;;
  warp)
    THREADS="${BUCKET_THREADS:-256}"; WARP_GX="${WARP_GX:-32}"; WARP_GY="${WARP_GY:-8}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
    export BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" \
      BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"
    ;;
  orbit)
    THREADS="${BUCKET_THREADS:-256}"; ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
    ORBITCTA_FLAT="${ORBITCTA_FLAT:-0}"; ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"
    export BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" \
      BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"
    unset BUCKET_ORBITCTA_FLAT_BLOCKS || true
    if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" != 0 ]]; then
      export BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM"
    else
      unset BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM || true
    fi
    ;;
esac

PROMOTION_META="$B300_GRAND_SELECTED_WORK_DIR/promotion.meta"
{
  printf 'schema=2\n'
  printf 'selection_schema=%s\n' "$SELECTION_SCHEMA"
  printf 'selected_env=%s\n' "$SELECTED_ENV"
  printf 'selected_backend=%s\n' "$B300_GRAND_SELECTED_BACKEND"
  printf 'selected_profile=%s\n' "$B300_GRAND_SELECTED_PROFILE"
  printf 'selected_binary=%s\n' "$B300_GRAND_SELECTED_BINARY"
  printf 'selected_binary_sha256=%s\n' "$BIN_SHA"
  printf 'selected_profile_sha256=%s\n' "$PROFILE_SHA"
  printf 'selected_race_result_sha256=%s\n' "$RACE_SHA"
  printf 'selected_grand_summary=%s\n' "$GRAND_SUMMARY_ENV"
  printf 'selected_grand_summary_sha256=%s\n' "$GRAND_SUMMARY_SHA"
  printf 'selected_runtime_kind=%s\n' "$B300_GRAND_SELECTED_RUNTIME_KIND"
  printf 'selected_target_mib=%s\n' "$B300_GRAND_SELECTED_TARGET_MIB"
  printf 'selected_max_window=%s\n' "$B300_GRAND_SELECTED_MAX_WINDOW"
  printf 'selected_smoke_prime=%s\n' "$B300_GRAND_SELECTED_SMOKE_PRIME"
  printf 'selected_smoke_residue=%s\n' "$B300_GRAND_SELECTED_RESIDUE"
  printf 'firstpass_head_sha=%s\n' "$B300_GRAND_SELECTED_HEAD_SHA"
  printf 'promotion_head_sha=%s\n' "$CURRENT_HEAD"
  printf 'allow_head_drift=%s\n' "$ALLOW_HEAD_DRIFT"
  printf 'allow_dirty_firstpass=%s\n' "$ALLOW_DIRTY_FIRSTPASS"
  printf 'allow_worktree_dirty=%s\n' "$ALLOW_WORKTREE_DIRTY"
  printf 'dry_run=%s\n' "$DRY_RUN"
} >"$PROMOTION_META"

cmd=(python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" 27
  --binary "$B300_GRAND_SELECTED_BINARY"
  --target-mib "$B300_GRAND_SELECTED_TARGET_MIB"
  --max-window "$B300_GRAND_SELECTED_MAX_WINDOW"
  --gpus 8
  --work-dir "$B300_GRAND_SELECTED_WORK_DIR")
cmd+=("$@")

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'B300 GRAND EXACT PROMOTION VALIDATED schema=%s command=' "$SELECTION_SCHEMA" >&2
  printf '%q ' "${cmd[@]}" >&2
  printf '\n' >&2
  echo "promotion_meta=$PROMOTION_META checkpoint=$B300_GRAND_SELECTED_CHECKPOINT" >&2
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
((GPU_COUNT>=8)) || { echo "need 8 visible GPUs; got $GPU_COUNT" >&2; exit 2; }

echo "=== B300 GRAND EXACT PROMOTION schema=$SELECTION_SCHEMA backend=$B300_GRAND_SELECTED_BACKEND profile=$B300_GRAND_SELECTED_PROFILE sha=${BIN_SHA:0:12} cached_smoke_prime=$B300_GRAND_SELECTED_SMOKE_PRIME ===" >&2
echo "work_dir=$B300_GRAND_SELECTED_WORK_DIR target_mib=$B300_GRAND_SELECTED_TARGET_MIB max_window=$B300_GRAND_SELECTED_MAX_WINDOW" >&2
exec "${cmd[@]}"