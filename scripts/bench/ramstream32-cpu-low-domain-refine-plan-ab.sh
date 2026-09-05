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
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_domain_refine_plan_ab}"

if (( N < 2 || N > 27 )); then
  echo "N must be in 2..27" >&2
  exit 2
fi

read -r -a configs <<<"$CONFIGS"
if ((${#configs[@]} == 0)); then
  echo "CONFIGS must contain worker:domain-size pairs" >&2
  exit 2
fi
for cfg in "${configs[@]}"; do
  if [[ ! "$cfg" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]]; then
    echo "invalid CONFIGS entry: $cfg (expected workers:domain-size)" >&2
    exit 2
  fi
  workers="${BASH_REMATCH[1]}"
  domain="${BASH_REMATCH[2]}"
  if (( domain > workers )); then
    echo "domain size must not exceed workers: $cfg" >&2
    exit 2
  fi
done

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-schedule-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_schedule_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/domain-refine-plan-ab-n${N}-${ts}.tsv"
meta="$OUT_DIR/domain-refine-plan-ab-n${N}-${ts}.meta"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}
field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s\n' "${token#*=}"
      return 0
    fi
  done
  return 1
}

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
cpu_low_domain_page_tiebreak=0
variants=refine0,refine1
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'workers\tdomain_size\trefine\tdomains\timbalance\tmax_worker_cells\tcross_domain_4k\tcross_domain_2m\tcross_worker_4k\tcross_worker_2m\touter_normalized_cap\tactive_domains\trefined_boundaries\trefined_job_moves\tdomain_build_s\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"
  domain="${cfg#*:}"
  for refine in 0 1; do
    echo "domain-refine-plan n=$N workers=$workers domain_size=$domain refine=$refine page_tiebreak=0" >&2
    line="$(CPU_LOW_DOMAIN_REFINE="$refine" CPU_LOW_DOMAIN_PAGE_TIEBREAK=0 \
      "$bin" "$N" "$workers" --domain-size "$domain" | head -n1)"
    [[ "$(field "$line" workers)" == "$workers" ]] || {
      echo "worker provenance mismatch for $cfg refine=$refine" >&2; exit 4;
    }
    [[ "$(field "$line" domain_size)" == "$domain" ]] || {
      echo "domain provenance mismatch for $cfg refine=$refine" >&2; exit 4;
    }
    [[ "$(field "$line" hybrid_domain_page_tiebreak)" == 0 ]] || {
      echo "page tie-break leaked into refine preflight cfg=$cfg refine=$refine" >&2; exit 4;
    }
    boundaries="$(field "$line" hybrid_domain_refined_boundaries)"
    moves="$(field "$line" hybrid_domain_refined_job_moves)"
    if [[ "$refine" == 0 && ( "$boundaries" != 0 || "$moves" != 0 ) ]]; then
      echo "unrefined preflight moved boundaries cfg=$cfg boundaries=$boundaries moves=$moves" >&2
      exit 5
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$workers" "$domain" "$refine" "$(field "$line" domains)" \
      "$(field "$line" hybrid_domain_imbalance)" \
      "$(field "$line" hybrid_domain_max_worker_cells)" \
      "$(field "$line" hybrid_domain_cross_domain_pages_4k)" \
      "$(field "$line" hybrid_domain_cross_domain_pages_2m)" \
      "$(field "$line" hybrid_domain_cross_worker_pages_4k)" \
      "$(field "$line" hybrid_domain_cross_worker_pages_2m)" \
      "$(field "$line" hybrid_domain_outer_normalized_cap)" \
      "$(field "$line" hybrid_domain_active_domains)" \
      "$boundaries" "$moves" "$(field "$line" domain_build_s)" \
      "$line" >>"$out"
  done
done

python3 - "$out" <<'PY'
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
rows = defaultdict(dict)
with open(path, newline='') as f:
    for row in csv.DictReader(f, delimiter='\t'):
        key = (int(row['workers']), int(row['domain_size']))
        rows[key][int(row['refine'])] = row

for key in sorted(rows):
    variants = rows[key]
    if 0 not in variants or 1 not in variants:
        raise SystemExit(f'missing refine pair for workers={key[0]} domain_size={key[1]}')
    a, b = variants[0], variants[1]
    i0, i1 = float(a['imbalance']), float(b['imbalance'])
    m0, m1 = int(a['max_worker_cells']), int(b['max_worker_cells'])
    c40, c41 = int(a['cross_domain_4k']), int(b['cross_domain_4k'])
    c20, c21 = int(a['cross_domain_2m']), int(b['cross_domain_2m'])
    b0, b1 = float(a['domain_build_s']), float(b['domain_build_s'])

    if m1 > m0:
        raise SystemExit(
            f'refinement max-worker regression workers={key[0]} '
            f'domain_size={key[1]} refine0={m0} refine1={m1}'
        )

    balance_better = m1 < m0
    pages4_better = c41 < c40
    pages2_better = c21 < c20
    pages_worse = c41 > c40 or c21 > c20
    pages_equal = c41 == c40 and c21 == c20

    if not balance_better and pages_equal:
        classification = 'no_change'
    elif balance_better and not pages_worse:
        classification = 'dominates'
    elif balance_better and pages_worse:
        classification = 'balance_page_tradeoff'
    elif not balance_better and (pages4_better or pages2_better) and not pages_worse:
        classification = 'page_only_improvement'
    else:
        classification = 'page_regression_without_balance_gain'

    print(
        f'comparison workers={key[0]} domain_size={key[1]} '
        f'classification={classification} '
        f'refine0_imbalance={i0:.9f} refine1_imbalance={i1:.9f} '
        f'imbalance_speedup={(i0/i1 if i1 else 0.0):.9f}x '
        f'max_worker_cells_saved={m0-m1} '
        f'cross_domain_4k_delta={c41-c40} '
        f'cross_domain_2m_delta={c21-c20} '
        f'refined_boundaries={b["refined_boundaries"]} '
        f'refined_job_moves={b["refined_job_moves"]} '
        f'refine_extra_build_s={b1-b0:.9f}'
    )
PY

echo "results=$out"
echo "metadata=$meta"
