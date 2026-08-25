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
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_domain_worker_locality_plan}"

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
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-domain-worker-locality-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_domain_worker_locality_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/domain-worker-locality-plan-n${N}-${ts}.tsv"
meta="$OUT_DIR/domain-worker-locality-plan-n${N}-${ts}.meta"

file_sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
objective=contiguous-under-lpt-cap-v5.25-plan
baseline=refined-domain-plus-v5.23-page
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'workers\tdomain_size\tdomains\tconverted_domains\tfallback_domains\tconverted_jobs\tbaseline_max_worker_cells\tlocality_max_worker_cells\tbaseline_cross_worker_2m\tlocality_cross_worker_2m\tbaseline_cross_worker_4k\tlocality_cross_worker_4k\tbaseline_owner_transitions\tlocality_owner_transitions\tcross_domain_2m\tcross_domain_4k\tworker_locality_build_s\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"; domain="${cfg#*:}"
  echo "worker-locality-plan n=$N workers=$workers domain_size=$domain" >&2
  line="$($bin "$N" "$workers" "$domain" 2> >(tee /dev/stderr) | tail -n1)"
  [[ "$(field "$line" objective)" == contiguous-under-lpt-cap-v5.25-plan ]] || {
    echo "objective provenance mismatch cfg=$cfg" >&2; exit 4;
  }
  [[ "$(field "$line" workers)" == "$workers" ]] || { echo "worker provenance mismatch cfg=$cfg" >&2; exit 4; }
  [[ "$(field "$line" domain_size)" == "$domain" ]] || { echo "domain provenance mismatch cfg=$cfg" >&2; exit 4; }

  python3 - \
    "$(field "$line" baseline_max_worker_cells)" \
    "$(field "$line" locality_max_worker_cells)" <<'PY'
import sys
base,loc=map(int,sys.argv[1:])
if loc > base:
    raise SystemExit('worker-locality max-worker regression')
PY

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$workers" "$domain" "$(field "$line" domains)" \
    "$(field "$line" converted_domains)" \
    "$(field "$line" fallback_domains)" \
    "$(field "$line" converted_jobs)" \
    "$(field "$line" baseline_max_worker_cells)" \
    "$(field "$line" locality_max_worker_cells)" \
    "$(field "$line" baseline_cross_worker_pages_2m)" \
    "$(field "$line" locality_cross_worker_pages_2m)" \
    "$(field "$line" baseline_cross_worker_pages_4k)" \
    "$(field "$line" locality_cross_worker_pages_4k)" \
    "$(field "$line" baseline_worker_owner_transitions)" \
    "$(field "$line" locality_worker_owner_transitions)" \
    "$(field "$line" cross_domain_pages_2m)" \
    "$(field "$line" cross_domain_pages_4k)" \
    "$(field "$line" worker_locality_build_s)" \
    "$line" >>"$out"
done

python3 - "$out" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        b2,l2=int(r['baseline_cross_worker_2m']),int(r['locality_cross_worker_2m'])
        b4,l4=int(r['baseline_cross_worker_4k']),int(r['locality_cross_worker_4k'])
        bt,lt=int(r['baseline_owner_transitions']),int(r['locality_owner_transitions'])
        converted=int(r['converted_domains'])
        fallback=int(r['fallback_domains'])
        if (l2,l4) < (b2,b4): cls='page_improvement'
        elif (l2,l4) == (b2,b4) and lt < bt: cls='transition_only_improvement'
        elif (l2,l4) == (b2,b4) and lt == bt: cls='no_change'
        else: cls='page_tradeoff'
        print(
            f"comparison workers={r['workers']} domain_size={r['domain_size']} "
            f"classification={cls} converted_domains={converted} fallback_domains={fallback} "
            f"cross_worker_2m_delta={l2-b2} cross_worker_4k_delta={l4-b4} "
            f"owner_transition_delta={lt-bt} "
            f"max_worker_cells_saved={int(r['baseline_max_worker_cells'])-int(r['locality_max_worker_cells'])} "
            f"worker_locality_build_s={float(r['worker_locality_build_s']):.9f}"
        )
PY

echo "results=$out"
echo "metadata=$meta"
