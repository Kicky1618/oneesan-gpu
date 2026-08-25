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
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_domain_page_plan_ab}"

if (( N < 2 || N > 27 )); then echo "N must be in 2..27" >&2; exit 2; fi
read -r -a configs <<<"$CONFIGS"
if ((${#configs[@]} == 0)); then echo "CONFIGS must contain worker:domain-size pairs" >&2; exit 2; fi
for cfg in "${configs[@]}"; do
  if [[ ! "$cfg" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]]; then
    echo "invalid CONFIGS entry: $cfg (expected workers:domain-size)" >&2; exit 2
  fi
  workers="${BASH_REMATCH[1]}"; domain="${BASH_REMATCH[2]}"
  if (( domain > workers )); then echo "domain size must not exceed workers: $cfg" >&2; exit 2; fi
done

if [[ "$BUILD" != 0 ]]; then N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh; fi
bin="$ROOT/build/ramstream32_cpu_low_schedule_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/domain-page-plan-ab-n${N}-${ts}.tsv"
meta="$OUT_DIR/domain-page-plan-ab-n${N}-${ts}.meta"

file_sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
cpu_low_domain_refine=1
variants=page0,page1
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'workers\tdomain_size\tpage_tiebreak\timbalance\tmax_worker_cells\tcross_domain_4k\tcross_domain_2m\tcross_worker_4k\tcross_worker_2m\tpage_boundary_moves\tpage_moved_jobs\tpage_penalty_2m_before\tpage_penalty_2m_after\tpage_penalty_4k_before\tpage_penalty_4k_after\tdomain_build_s\tpage_build_s\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"; domain="${cfg#*:}"
  for page in 0 1; do
    echo "domain-page-plan n=$N workers=$workers domain_size=$domain page=$page" >&2
    line="$(CPU_LOW_DOMAIN_REFINE=1 CPU_LOW_DOMAIN_PAGE_TIEBREAK="$page" \
      "$bin" "$N" "$workers" --domain-size "$domain" | head -n1)"
    [[ "$(field "$line" workers)" == "$workers" ]] || { echo "worker provenance mismatch cfg=$cfg page=$page" >&2; exit 4; }
    [[ "$(field "$line" domain_size)" == "$domain" ]] || { echo "domain provenance mismatch cfg=$cfg page=$page" >&2; exit 4; }
    [[ "$(field "$line" hybrid_domain_refine)" == 1 ]] || { echo "refine provenance mismatch cfg=$cfg page=$page" >&2; exit 4; }
    [[ "$(field "$line" hybrid_domain_page_tiebreak)" == "$page" ]] || { echo "page provenance mismatch cfg=$cfg page=$page" >&2; exit 4; }

    if [[ "$page" == 1 ]]; then
      p20="$(field "$line" hybrid_domain_page_penalty_2m_before)"
      p21="$(field "$line" hybrid_domain_page_penalty_2m_after)"
      p40="$(field "$line" hybrid_domain_page_penalty_4k_before)"
      p41="$(field "$line" hybrid_domain_page_penalty_4k_after)"
      python3 - "$p20" "$p21" "$p40" "$p41" <<'PY'
import sys
p20,p21,p40,p41=map(int,sys.argv[1:])
if (p21,p41) > (p20,p40):
    raise SystemExit('local page penalty regression')
PY
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$workers" "$domain" "$page" \
      "$(field "$line" hybrid_domain_imbalance)" \
      "$(field "$line" hybrid_domain_max_worker_cells)" \
      "$(field "$line" hybrid_domain_cross_domain_pages_4k)" \
      "$(field "$line" hybrid_domain_cross_domain_pages_2m)" \
      "$(field "$line" hybrid_domain_cross_worker_pages_4k)" \
      "$(field "$line" hybrid_domain_cross_worker_pages_2m)" \
      "$(field "$line" hybrid_domain_page_boundary_moves)" \
      "$(field "$line" hybrid_domain_page_moved_jobs)" \
      "$(field "$line" hybrid_domain_page_penalty_2m_before)" \
      "$(field "$line" hybrid_domain_page_penalty_2m_after)" \
      "$(field "$line" hybrid_domain_page_penalty_4k_before)" \
      "$(field "$line" hybrid_domain_page_penalty_4k_after)" \
      "$(field "$line" domain_build_s)" \
      "$(field "$line" hybrid_domain_page_build_s)" \
      "$line" >>"$out"
  done
done

python3 - "$out" <<'PY'
import csv, sys
from collections import defaultdict
rows=defaultdict(dict)
with open(sys.argv[1], newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        rows[(int(r['workers']),int(r['domain_size']))][int(r['page_tiebreak'])]=r
for key in sorted(rows):
    if 0 not in rows[key] or 1 not in rows[key]:
        raise SystemExit(f'missing pair {key}')
    a,b=rows[key][0],rows[key][1]
    m0,m1=int(a['max_worker_cells']),int(b['max_worker_cells'])
    if m1 > m0:
        raise SystemExit(f'page tie-break max-worker regression workers={key[0]} domain_size={key[1]} before={m0} after={m1}')
    c20,c21=int(a['cross_domain_2m']),int(b['cross_domain_2m'])
    c40,c41=int(a['cross_domain_4k']),int(b['cross_domain_4k'])
    if (c21,c41) < (c20,c40): cls='global_page_improvement'
    elif (c21,c41) == (c20,c40): cls='global_no_change'
    elif c21 <= c20 and c41 <= c40: cls='global_nonworse'
    else: cls='global_page_tradeoff'
    print(
        f'comparison workers={key[0]} domain_size={key[1]} '
        f'page0_max_worker_cells={m0} page1_max_worker_cells={m1} '
        f'max_worker_cells_saved={m0-m1} '
        f'cross_domain_2m_delta={c21-c20} cross_domain_4k_delta={c41-c40} '
        f'page_boundary_moves={b["page_boundary_moves"]} page_moved_jobs={b["page_moved_jobs"]} '
        f'page_extra_build_s={float(b["domain_build_s"])-float(a["domain_build_s"]):.9f} '
        f'classification={cls}'
    )
PY

echo "results=$out"
echo "metadata=$meta"
