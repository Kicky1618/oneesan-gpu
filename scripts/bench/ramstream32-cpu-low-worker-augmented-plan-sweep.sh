#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$ROOT" ]] || ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

N="${N:-27}"
ARCH="${ARCH:-native}"
CONFIGS="${CONFIGS:-32:16 64:32 96:48 128:64}"
MAX_RUN="${MAX_RUN:-4}"
MAX_SWAP="${MAX_SWAP:-4}"
BUILD="${BUILD:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_worker_augmented}"

if (( N < 2 || N > 27 )); then
  echo "N must be in 2..27" >&2
  exit 2
fi
if [[ "$PREFLIGHT_ONLY" != 0 && "$PREFLIGHT_ONLY" != 1 ]]; then
  echo "PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 2
fi

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-worker-augmented-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_worker_augmented_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing augmented plan binary: $bin" >&2; exit 3; }
mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
line=""
field() {
  local key="$1" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s\n' "${token#*=}"
      return 0
    fi
  done
  return 1
}

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  out="$OUT_DIR/workspace-preflight-n${N}-${ts}.txt"
  meta="$OUT_DIR/workspace-preflight-n${N}-${ts}.meta"
  echo "workspace preflight n=$N" >&2
  line="$("$bin" "$N" --workspace-only 2> >(tee /dev/stderr) | tee "$out" | tail -n1)"
  [[ "$(field objective)" == exact-workspace-audit-v5.36-preflight ]] || exit 4
  [[ "$(field workspace_builder)" == streaming-two-pass ]] || exit 4
  [[ "$(field workspace_audit_ok)" == 1 ]] || exit 4
  [[ "$(field workspace_audited_jobs)" =~ ^[1-9][0-9]*$ ]] || exit 4
  [[ "$(field workspace_audited_cells)" =~ ^[1-9][0-9]*$ ]] || exit 4
  cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
n=$N
objective=exact-workspace-audit-v5.36-preflight
workspace_builder=$(field workspace_builder)
authoritative_main_states=$(field authoritative_main_states)
authoritative_blocked_states=$(field authoritative_blocked_states)
workspace_audit_ok=$(field workspace_audit_ok)
workspace_audited_jobs=$(field workspace_audited_jobs)
workspace_audited_cells=$(field workspace_audited_cells)
workspace_mib=$(field workspace_mib)
workspace_reserved_mib=$(field workspace_reserved_mib)
workspace_mask_index_mib=$(field workspace_mask_index_mib)
workspace_dense_index_mib=$(field workspace_dense_index_mib)
workspace_dense_reserved_mib=$(field workspace_dense_reserved_mib)
workspace_transition_mib=$(field workspace_transition_mib)
workspace_audit_s=$(field workspace_audit_s)
workspace_build_s=$(field workspace_build_s)
preflight_total_s=$(field preflight_total_s)
EOF
  echo "preflight_ok=1 n=$N builder=$(field workspace_builder) main_states=$(field authoritative_main_states) blocked_states=$(field authoritative_blocked_states) workspace_jobs=$(field workspace_audited_jobs) workspace_cells=$(field workspace_audited_cells) workspace_mib=$(field workspace_mib) workspace_reserved_mib=$(field workspace_reserved_mib) workspace_audit_s=$(field workspace_audit_s) workspace_build_s=$(field workspace_build_s) total_s=$(field preflight_total_s)"
  echo "result=$out"
  echo "metadata=$meta"
  exit 0
fi

for x in MAX_RUN MAX_SWAP; do
  v="${!x}"
  [[ "$v" =~ ^[1-9][0-9]*$ ]] && (( v <= 64 )) || exit 2
done
read -r -a configs <<<"$CONFIGS"
for cfg in "${configs[@]}"; do
  [[ "$cfg" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]] || exit 2
  (( BASH_REMATCH[2] <= BASH_REMATCH[1] )) || exit 2
done

out="$OUT_DIR/augmented-n${N}-${ts}.tsv"
meta="$OUT_DIR/augmented-n${N}-${ts}.meta"
cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
n=$N
configs=$CONFIGS
max_run=$MAX_RUN
max_swap=$MAX_SWAP
objective=exact-neutral-augmented-v5.36-plan
workspace_audit=required
workspace_builder=streaming-two-pass
EOF
printf 'workers\tdomain_size\tbaseline_pages_2m\tbaseline_pages_4k\tbaseline_transitions\tbaseline_max_worker_cells\taugmented_pages_2m\taugmented_pages_4k\taugmented_transitions\taugmented_max_worker_cells\tpages_2m_delta\tpages_4k_delta\ttransition_delta\taugmented_rounds\texact_schedule_changes\texact_primary_improvements\texact_profile_improvements\tneutral_moves\tneutral_candidates\taugmented_build_s\tworkspace_audited_jobs\tworkspace_audited_cells\tworkspace_mib\tworkspace_audit_s\tworkspace_build_s\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"
  domain="${cfg#*:}"
  echo "augmented n=$N workers=$workers domain=$domain" >&2
  line="$("$bin" "$N" "$workers" "$domain" "$MAX_RUN" "$MAX_SWAP" 2> >(tee /dev/stderr) | tail -n1)"
  [[ "$(field objective)" == exact-neutral-augmented-v5.36-plan ]] || exit 4
  [[ "$(field workspace_builder)" == streaming-two-pass ]] || exit 4
  [[ "$(field limits_clear)" == 1 ]] || exit 4
  [[ "$(field workspace_audit_ok)" == 1 ]] || exit 4
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$workers" "$domain" \
    "$(field baseline_pages_2m)" "$(field baseline_pages_4k)" "$(field baseline_transitions)" "$(field baseline_max_worker_cells)" \
    "$(field augmented_pages_2m)" "$(field augmented_pages_4k)" "$(field augmented_transitions)" "$(field augmented_max_worker_cells)" \
    "$(field pages_2m_delta)" "$(field pages_4k_delta)" "$(field transition_delta)" "$(field augmented_rounds)" \
    "$(field exact_schedule_changes)" "$(field exact_primary_improvements)" "$(field exact_profile_improvements)" \
    "$(field neutral_moves)" "$(field neutral_candidates)" "$(field augmented_build_s)" \
    "$(field workspace_audited_jobs)" "$(field workspace_audited_cells)" "$(field workspace_mib)" "$(field workspace_audit_s)" "$(field workspace_build_s)" \
    "$line" >>"$out"
done

python3 - "$out" <<'PY'
import csv, sys
for r in csv.DictReader(open(sys.argv[1], newline=''), delimiter='\t'):
    b = (int(r['baseline_pages_2m']), int(r['baseline_pages_4k']), int(r['baseline_transitions']))
    a = (int(r['augmented_pages_2m']), int(r['augmented_pages_4k']), int(r['augmented_transitions']))
    bm = int(r['baseline_max_worker_cells'])
    am = int(r['augmented_max_worker_cells'])
    moves = int(r['neutral_moves'])
    rounds = int(r['augmented_rounds'])
    if a > b or am > bm:
        raise SystemExit('augmented regression')
    if int(r['workspace_audited_jobs']) <= 0 or int(r['workspace_audited_cells']) <= 0:
        raise SystemExit('workspace audit unexpectedly empty')
    if a < b and moves:
        cls = 'augmented_plateau_escape'
    elif a < b:
        cls = 'augmented_exact_improvement'
    elif am < bm:
        cls = 'augmented_load_improvement'
    elif moves:
        cls = 'neutral_fixedpoint_change'
    else:
        cls = 'no_change'
    print(
        f"comparison workers={r['workers']} domain_size={r['domain_size']} classification={cls} "
        f"rounds={rounds} neutral_moves={moves} exact_primary_improvements={r['exact_primary_improvements']} "
        f"pages_2m_delta={a[0]-b[0]} pages_4k_delta={a[1]-b[1]} transition_delta={a[2]-b[2]} "
        f"max_worker_delta={am-bm} workspace_jobs={r['workspace_audited_jobs']} "
        f"workspace_cells={r['workspace_audited_cells']} workspace_mib={float(r['workspace_mib']):.6f} "
        f"workspace_audit_s={float(r['workspace_audit_s']):.9f} build_s={float(r['augmented_build_s']):.9f}"
    )
PY

echo "results=$out"
echo "metadata=$meta"
