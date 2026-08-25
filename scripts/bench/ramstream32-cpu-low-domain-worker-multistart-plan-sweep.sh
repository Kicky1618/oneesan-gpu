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
BUILD="${BUILD:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_domain_worker_multistart_plan}"

if (( N < 2 || N > 27 )); then echo "N must be in 2..27" >&2; exit 2; fi
read -r -a configs <<<"$CONFIGS"
if ((${#configs[@]} == 0)); then echo "CONFIGS must contain worker:domain-size pairs" >&2; exit 2; fi
for cfg in "${configs[@]}"; do
  if [[ ! "$cfg" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]]; then
    echo "invalid CONFIGS entry: $cfg" >&2; exit 2
  fi
  workers="${BASH_REMATCH[1]}"; domain="${BASH_REMATCH[2]}"
  if (( domain > workers )); then echo "domain size must not exceed workers: $cfg" >&2; exit 2; fi
done

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-domain-worker-multistart-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_domain_worker_multistart_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/domain-worker-multistart-plan-n${N}-${ts}.tsv"
meta="$OUT_DIR/domain-worker-multistart-plan-n${N}-${ts}.meta"

file_sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
objective=multistart-global-unique-worker-v5.28-plan
exact_implementation=flat-page-delta-v5.29
direct=v5.25-to-v5.27
hybrid=v5.25-to-v5.26-to-v5.27
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'workers\tdomain_size\tdomains\tselected_source\tparent_max_worker_cells\tdirect_max_worker_cells\traw_v526_max_worker_cells\thybrid_max_worker_cells\tselected_max_worker_cells\tparent_pages_2m\tparent_pages_4k\tparent_transitions\tdirect_pages_2m\tdirect_pages_4k\tdirect_transitions\traw_v526_pages_2m\traw_v526_pages_4k\traw_v526_transitions\thybrid_pages_2m\thybrid_pages_4k\thybrid_transitions\tselected_pages_2m\tselected_pages_4k\tselected_transitions\tdirect_accepted_moves\thybrid_v526_accepted_moves\thybrid_v527_accepted_moves\tdirect_candidate_evaluations\thybrid_candidate_evaluations\tdirect_flat_delta_normalizations\thybrid_flat_delta_normalizations\tdirect_flat_delta_peak_entries\thybrid_flat_delta_peak_entries\tdirect_mask_index_build_s\thybrid_mask_index_build_s\tdirect_build_s\thybrid_v526_build_s\thybrid_v527_build_s\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"; domain="${cfg#*:}"
  echo "worker-multistart-plan n=$N workers=$workers domain_size=$domain" >&2
  line="$($bin "$N" "$workers" "$domain" 2> >(tee /dev/stderr) | tail -n1)"
  [[ "$(field "$line" objective)" == multistart-global-unique-worker-v5.28-plan ]] || {
    echo "objective provenance mismatch cfg=$cfg" >&2; exit 4;
  }
  [[ "$(field "$line" exact_implementation)" == flat-page-delta-v5.29 ]] || {
    echo "exact implementation provenance mismatch cfg=$cfg" >&2; exit 4;
  }
  [[ "$(field "$line" workers)" == "$workers" ]] || { echo "worker provenance mismatch cfg=$cfg" >&2; exit 4; }
  [[ "$(field "$line" domain_size)" == "$domain" ]] || { echo "domain provenance mismatch cfg=$cfg" >&2; exit 4; }
  src="$(field "$line" selected_source)"
  [[ "$src" == direct || "$src" == hybrid ]] || { echo "invalid selected source cfg=$cfg source=$src" >&2; exit 4; }

  python3 - \
    "$(field "$line" parent_max_worker_cells)" \
    "$(field "$line" direct_max_worker_cells)" \
    "$(field "$line" hybrid_raw_v526_max_worker_cells)" \
    "$(field "$line" hybrid_max_worker_cells)" \
    "$(field "$line" selected_max_worker_cells)" \
    "$(field "$line" parent_pages_2m)" "$(field "$line" parent_pages_4k)" "$(field "$line" parent_transitions)" \
    "$(field "$line" direct_pages_2m)" "$(field "$line" direct_pages_4k)" "$(field "$line" direct_transitions)" \
    "$(field "$line" raw_v526_pages_2m)" "$(field "$line" raw_v526_pages_4k)" "$(field "$line" raw_v526_transitions)" \
    "$(field "$line" hybrid_pages_2m)" "$(field "$line" hybrid_pages_4k)" "$(field "$line" hybrid_transitions)" \
    "$(field "$line" selected_pages_2m)" "$(field "$line" selected_pages_4k)" "$(field "$line" selected_transitions)" <<'PY'
import sys
v=list(map(int,sys.argv[1:]))
parent_max,direct_max,raw_max,hybrid_max,selected_max=v[:5]
p=tuple(v[5:8]); d=tuple(v[8:11]); r=tuple(v[11:14]); h=tuple(v[14:17]); s=tuple(v[17:20])
if direct_max > parent_max or raw_max > parent_max or hybrid_max > raw_max or selected_max > parent_max:
    raise SystemExit('multistart max-worker regression')
if d > p:
    raise SystemExit('direct exact branch regression')
if h > r:
    raise SystemExit('hybrid exact cleanup regression')
if s > d or s > h or s > r or s > p:
    raise SystemExit('selected branch failed dominance contract')
PY

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$workers" "$domain" "$(field "$line" domains)" "$src" \
    "$(field "$line" parent_max_worker_cells)" \
    "$(field "$line" direct_max_worker_cells)" \
    "$(field "$line" hybrid_raw_v526_max_worker_cells)" \
    "$(field "$line" hybrid_max_worker_cells)" \
    "$(field "$line" selected_max_worker_cells)" \
    "$(field "$line" parent_pages_2m)" "$(field "$line" parent_pages_4k)" "$(field "$line" parent_transitions)" \
    "$(field "$line" direct_pages_2m)" "$(field "$line" direct_pages_4k)" "$(field "$line" direct_transitions)" \
    "$(field "$line" raw_v526_pages_2m)" "$(field "$line" raw_v526_pages_4k)" "$(field "$line" raw_v526_transitions)" \
    "$(field "$line" hybrid_pages_2m)" "$(field "$line" hybrid_pages_4k)" "$(field "$line" hybrid_transitions)" \
    "$(field "$line" selected_pages_2m)" "$(field "$line" selected_pages_4k)" "$(field "$line" selected_transitions)" \
    "$(field "$line" direct_accepted_moves)" \
    "$(field "$line" hybrid_v526_accepted_moves)" \
    "$(field "$line" hybrid_v527_accepted_moves)" \
    "$(field "$line" direct_candidate_evaluations)" \
    "$(field "$line" hybrid_candidate_evaluations)" \
    "$(field "$line" direct_flat_delta_normalizations)" \
    "$(field "$line" hybrid_flat_delta_normalizations)" \
    "$(field "$line" direct_flat_delta_peak_entries)" \
    "$(field "$line" hybrid_flat_delta_peak_entries)" \
    "$(field "$line" direct_mask_index_build_s)" \
    "$(field "$line" hybrid_mask_index_build_s)" \
    "$(field "$line" direct_build_s)" \
    "$(field "$line" hybrid_v526_build_s)" \
    "$(field "$line" hybrid_v527_build_s)" \
    "$line" >>"$out"
done

python3 - "$out" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        p=(int(r['parent_pages_2m']),int(r['parent_pages_4k']),int(r['parent_transitions']))
        d=(int(r['direct_pages_2m']),int(r['direct_pages_4k']),int(r['direct_transitions']))
        raw=(int(r['raw_v526_pages_2m']),int(r['raw_v526_pages_4k']),int(r['raw_v526_transitions']))
        h=(int(r['hybrid_pages_2m']),int(r['hybrid_pages_4k']),int(r['hybrid_transitions']))
        s=(int(r['selected_pages_2m']),int(r['selected_pages_4k']),int(r['selected_transitions']))
        if s < p: cls='multistart_improvement'
        else: cls='multistart_no_change'
        if h < d: basin='hybrid_wins'
        elif d < h: basin='direct_wins'
        else: basin='exact_tie'
        de=int(r['direct_candidate_evaluations'])
        he=int(r['hybrid_candidate_evaluations'])
        dn=int(r['direct_flat_delta_normalizations'])
        hn=int(r['hybrid_flat_delta_normalizations'])
        print(
            f"comparison workers={r['workers']} domain_size={r['domain_size']} "
            f"classification={cls} basin={basin} selected_source={r['selected_source']} "
            f"selected_2m_delta={s[0]-p[0]} selected_4k_delta={s[1]-p[1]} "
            f"selected_transition_delta={s[2]-p[2]} "
            f"hybrid_cleanup_vs_raw_v526_2m_delta={h[0]-raw[0]} "
            f"hybrid_cleanup_vs_raw_v526_4k_delta={h[1]-raw[1]} "
            f"direct_candidate_evaluations={de} hybrid_candidate_evaluations={he} "
            f"direct_normalizations_per_candidate={(dn/de if de else 0.0):.6f} "
            f"hybrid_normalizations_per_candidate={(hn/he if he else 0.0):.6f} "
            f"direct_flat_delta_peak_entries={r['direct_flat_delta_peak_entries']} "
            f"hybrid_flat_delta_peak_entries={r['hybrid_flat_delta_peak_entries']} "
            f"direct_build_s={float(r['direct_build_s']):.9f} "
            f"hybrid_total_build_s={float(r['hybrid_v526_build_s'])+float(r['hybrid_v527_build_s']):.9f}"
        )
PY

echo "results=$out"
echo "metadata=$meta"
