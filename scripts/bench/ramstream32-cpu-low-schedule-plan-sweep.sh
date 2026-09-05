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
ANALYZE="${ANALYZE:-1}"
MAX_IMBALANCE="${MAX_IMBALANCE:-}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_schedule_plan_sweep}"

# The generic topology/Pareto sweep is the standard refined-domain baseline.
# Page-aware locality is intentionally isolated in domain-page-plan-ab.sh.
CPU_LOW_DOMAIN_REFINE=1
CPU_LOW_DOMAIN_PAGE_TIEBREAK=0

if (( N < 2 || N > 27 )); then
  echo "N must be in 2..27" >&2
  exit 2
fi
if [[ "$ANALYZE" != 0 && "$ANALYZE" != 1 ]]; then
  echo "ANALYZE must be 0 or 1" >&2
  exit 2
fi
if [[ -n "$MAX_IMBALANCE" && ! "$MAX_IMBALANCE" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "MAX_IMBALANCE must be a non-negative number" >&2
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
out="$OUT_DIR/low-schedule-plan-sweep-n${N}-${ts}.tsv"
meta="$OUT_DIR/low-schedule-plan-sweep-n${N}-${ts}.meta"
analysis="$OUT_DIR/low-schedule-plan-sweep-n${N}-${ts}.analysis.txt"

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
cpu_low_domain_refine=$CPU_LOW_DOMAIN_REFINE
cpu_low_domain_page_tiebreak=$CPU_LOW_DOMAIN_PAGE_TIEBREAK
analyze=$ANALYZE
max_imbalance=${MAX_IMBALANCE:-none}
binary_sha256=$(file_sha256 "$bin")
EOF

# Raw probe names retain hybrid_domain_* for compatibility; persisted sweep
# columns use the production domain terminology. This sweep is page=0 baseline.
printf 'workers\tdomain_size\tdomains\tlpt_imbalance\tlpt_cross_domain_4k\tlpt_cross_domain_2m\tcontiguous_imbalance\tcontiguous_cross_domain_4k\tcontiguous_cross_domain_2m\tdomain_imbalance\tdomain_cross_domain_4k\tdomain_cross_domain_2m\tdomain_cross_worker_4k\tdomain_cross_worker_2m\tdomain_outer_normalized_cap\tdomain_active_domains\tdomain_refined_boundaries\tdomain_refined_job_moves\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"
  domain="${cfg#*:}"
  echo "schedule-plan n=$N workers=$workers domain_size=$domain refine=1 page_tiebreak=0" >&2
  line="$(CPU_LOW_DOMAIN_REFINE="$CPU_LOW_DOMAIN_REFINE" \
    CPU_LOW_DOMAIN_PAGE_TIEBREAK="$CPU_LOW_DOMAIN_PAGE_TIEBREAK" \
    "$bin" "$N" "$workers" --domain-size "$domain" | head -n1)"
  [[ "$(field "$line" workers)" == "$workers" ]] || {
    echo "worker provenance mismatch for $cfg" >&2; exit 4;
  }
  [[ "$(field "$line" domain_size)" == "$domain" ]] || {
    echo "domain provenance mismatch for $cfg" >&2; exit 4;
  }
  [[ "$(field "$line" hybrid_domain_refine)" == "$CPU_LOW_DOMAIN_REFINE" ]] || {
    echo "refine provenance mismatch for $cfg" >&2; exit 4;
  }
  [[ "$(field "$line" hybrid_domain_page_tiebreak)" == "$CPU_LOW_DOMAIN_PAGE_TIEBREAK" ]] || {
    echo "page provenance mismatch for $cfg" >&2; exit 4;
  }

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$workers" "$domain" "$(field "$line" domains)" \
    "$(field "$line" imbalance)" \
    "$(field "$line" cross_domain_pages_4k)" \
    "$(field "$line" cross_domain_pages_2m)" \
    "$(field "$line" contiguous_imbalance)" \
    "$(field "$line" contiguous_cross_domain_pages_4k)" \
    "$(field "$line" contiguous_cross_domain_pages_2m)" \
    "$(field "$line" hybrid_domain_imbalance)" \
    "$(field "$line" hybrid_domain_cross_domain_pages_4k)" \
    "$(field "$line" hybrid_domain_cross_domain_pages_2m)" \
    "$(field "$line" hybrid_domain_cross_worker_pages_4k)" \
    "$(field "$line" hybrid_domain_cross_worker_pages_2m)" \
    "$(field "$line" hybrid_domain_outer_normalized_cap)" \
    "$(field "$line" hybrid_domain_active_domains)" \
    "$(field "$line" hybrid_domain_refined_boundaries)" \
    "$(field "$line" hybrid_domain_refined_job_moves)" \
    "$line" >>"$out"
done

awk -F '\t' '
  NR==1 {next}
  {
    printf("summary workers=%s domain_size=%s domains=%s lpt_imbalance=%s contiguous_imbalance=%s domain_imbalance=%s lpt_cross_domain_4k=%s contiguous_cross_domain_4k=%s domain_cross_domain_4k=%s lpt_cross_domain_2m=%s contiguous_cross_domain_2m=%s domain_cross_domain_2m=%s refined_boundaries=%s refined_job_moves=%s\n",
           $1,$2,$3,$4,$7,$10,$5,$8,$11,$6,$9,$12,$17,$18)
  }
' "$out"

if [[ "$ANALYZE" == 1 ]]; then
  args=(scripts/tools/analyze_cpu_low_schedule_plan_sweep.py "$out")
  if [[ -n "$MAX_IMBALANCE" ]]; then
    args+=(--max-imbalance "$MAX_IMBALANCE")
  fi
  python3 "${args[@]}" | tee "$analysis"
fi

echo "results=$out"
echo "metadata=$meta"
if [[ "$ANALYZE" == 1 ]]; then echo "analysis=$analysis"; fi
