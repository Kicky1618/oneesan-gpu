#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

N="${N:-27}"
ARCH="${ARCH:-native}"
CONFIGS="${CONFIGS:-32:16 64:32 96:48 128:64}"
MAX_RUN="${MAX_RUN:-4}"
MAX_SWAP="${MAX_SWAP:-4}"
BUILD="${BUILD:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_worker_fixedpoint}"

if (( N < 2 || N > 27 )); then echo "N must be in 2..27" >&2; exit 2; fi
for x in MAX_RUN MAX_SWAP; do
  v="${!x}"
  [[ "$v" =~ ^[1-9][0-9]*$ ]] && (( v <= 64 )) || { echo "$x must be 1..64" >&2; exit 2; }
done
read -r -a configs <<<"$CONFIGS"
if ((${#configs[@]} == 0)); then echo "CONFIGS must be non-empty" >&2; exit 2; fi
for cfg in "${configs[@]}"; do
  [[ "$cfg" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]] || { echo "invalid CONFIGS entry: $cfg" >&2; exit 2; }
  (( BASH_REMATCH[2] <= BASH_REMATCH[1] )) || { echo "domain size exceeds workers: $cfg" >&2; exit 2; }
done

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-worker-fixedpoint-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_worker_fixedpoint_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/fixedpoint-n${N}-${ts}.tsv"
meta="$OUT_DIR/fixedpoint-n${N}-${ts}.meta"
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
max_run=$MAX_RUN
max_swap=$MAX_SWAP
objective=alternating-run-swap-v5.34-plan
EOF

printf 'workers\tdomain_size\tmultistart_source\tparent_pages_2m\tparent_pages_4k\tparent_transitions\trs_pages_2m\trs_pages_4k\trs_transitions\trs_rounds\trs_run_accepted\trs_swap_accepted\trs_build_s\tsr_pages_2m\tsr_pages_4k\tsr_transitions\tsr_rounds\tsr_run_accepted\tsr_swap_accepted\tsr_build_s\tselected_order\tselected_pages_2m\tselected_pages_4k\tselected_transitions\tparent_max_worker_cells\tselected_max_worker_cells\tworkspace_build_s\tworkspace_mib\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"; domain="${cfg#*:}"
  echo "fixedpoint n=$N workers=$workers domain_size=$domain" >&2
  line="$($bin "$N" "$workers" "$domain" "$MAX_RUN" "$MAX_SWAP" 2> >(tee /dev/stderr) | tail -n1)"
  [[ "$(field "$line" objective)" == alternating-run-swap-v5.34-plan ]] || exit 4
  [[ "$(field "$line" identical_parent_score)" == 1 ]] || exit 4
  [[ "$(field "$line" limits_clear)" == 1 ]] || exit 4
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$workers" "$domain" "$(field "$line" multistart_source)" \
    "$(field "$line" parent_pages_2m)" "$(field "$line" parent_pages_4k)" "$(field "$line" parent_transitions)" \
    "$(field "$line" rs_pages_2m)" "$(field "$line" rs_pages_4k)" "$(field "$line" rs_transitions)" \
    "$(field "$line" rs_rounds)" "$(field "$line" rs_run_accepted)" "$(field "$line" rs_swap_accepted)" "$(field "$line" rs_build_s)" \
    "$(field "$line" sr_pages_2m)" "$(field "$line" sr_pages_4k)" "$(field "$line" sr_transitions)" \
    "$(field "$line" sr_rounds)" "$(field "$line" sr_run_accepted)" "$(field "$line" sr_swap_accepted)" "$(field "$line" sr_build_s)" \
    "$(field "$line" selected_order)" "$(field "$line" selected_pages_2m)" "$(field "$line" selected_pages_4k)" "$(field "$line" selected_transitions)" \
    "$(field "$line" parent_max_worker_cells)" "$(field "$line" selected_max_worker_cells)" \
    "$(field "$line" workspace_build_s)" "$(field "$line" workspace_mib)" "$line" >>"$out"
done

python3 - "$out" <<'PY'
import csv,sys
for r in csv.DictReader(open(sys.argv[1],newline=''),delimiter='\t'):
    p=(int(r['parent_pages_2m']),int(r['parent_pages_4k']),int(r['parent_transitions']))
    a=(int(r['rs_pages_2m']),int(r['rs_pages_4k']),int(r['rs_transitions']))
    b=(int(r['sr_pages_2m']),int(r['sr_pages_4k']),int(r['sr_transitions']))
    s=(int(r['selected_pages_2m']),int(r['selected_pages_4k']),int(r['selected_transitions']))
    if s > p or s > a or s > b: raise SystemExit('fixedpoint dominance regression')
    if a < b: basin='run_swap_wins'
    elif b < a: basin='swap_run_wins'
    else: basin='exact_tie'
    cls='fixedpoint_improvement' if s<p else 'no_change'
    print(
      f"comparison workers={r['workers']} domain_size={r['domain_size']} "
      f"classification={cls} basin={basin} selected_order={r['selected_order']} "
      f"selected_2m_delta={s[0]-p[0]} selected_4k_delta={s[1]-p[1]} "
      f"selected_transition_delta={s[2]-p[2]} "
      f"rs_rounds={r['rs_rounds']} sr_rounds={r['sr_rounds']} "
      f"rs_build_s={float(r['rs_build_s']):.9f} sr_build_s={float(r['sr_build_s']):.9f}"
    )
PY

echo "results=$out"
echo "metadata=$meta"
