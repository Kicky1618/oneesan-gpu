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
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_worker_shared_multistart}"

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
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-worker-shared-multistart-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_worker_shared_multistart_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/shared-multistart-n${N}-${ts}.tsv"
meta="$OUT_DIR/shared-multistart-n${N}-${ts}.meta"

file_sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
objective=shared-dense-multistart-v5.31-plan
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'workers\tdomain_size\tselected_source\tdirect_pages_2m\tdirect_pages_4k\thybrid_pages_2m\thybrid_pages_4k\tlegacy_total_build_s\tworkspace_build_s\tworkspace_mib\tshared_direct_search_s\tshared_v526_build_s\tshared_hybrid_search_s\tshared_total_build_s\tshared_vs_legacy_speedup\tdirect_candidate_evaluations\thybrid_candidate_evaluations\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"; domain="${cfg#*:}"
  echo "shared-multistart n=$N workers=$workers domain_size=$domain" >&2
  line="$($bin "$N" "$workers" "$domain" 2> >(tee /dev/stderr) | tail -n1)"
  [[ "$(field "$line" objective)" == shared-dense-multistart-v5.31-plan ]] || {
    echo "objective provenance mismatch cfg=$cfg" >&2; exit 4;
  }
  [[ "$(field "$line" identical_direct_schedule)" == 1 ]] || exit 4
  [[ "$(field "$line" identical_hybrid_schedule)" == 1 ]] || exit 4
  [[ "$(field "$line" identical_selector)" == 1 ]] || exit 4
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$workers" "$domain" "$(field "$line" selected_source)" \
    "$(field "$line" direct_pages_2m)" "$(field "$line" direct_pages_4k)" \
    "$(field "$line" hybrid_pages_2m)" "$(field "$line" hybrid_pages_4k)" \
    "$(field "$line" legacy_total_build_s)" \
    "$(field "$line" workspace_build_s)" \
    "$(field "$line" workspace_mib)" \
    "$(field "$line" shared_direct_search_s)" \
    "$(field "$line" shared_v526_build_s)" \
    "$(field "$line" shared_hybrid_search_s)" \
    "$(field "$line" shared_total_build_s)" \
    "$(field "$line" shared_vs_legacy_speedup)" \
    "$(field "$line" direct_candidate_evaluations)" \
    "$(field "$line" hybrid_candidate_evaluations)" \
    "$line" >>"$out"
done

python3 - "$out" <<'PY'
import csv, statistics, sys
rows=list(csv.DictReader(open(sys.argv[1], newline=''), delimiter='\t'))
for r in rows:
    speed=float(r['shared_vs_legacy_speedup'])
    cls='shared_faster' if speed > 1.02 else ('shared_slower' if speed < 0.98 else 'near_tie')
    print(
        f"comparison workers={r['workers']} domain_size={r['domain_size']} "
        f"classification={cls} speedup={speed:.6f} selected_source={r['selected_source']} "
        f"legacy_total_build_s={float(r['legacy_total_build_s']):.9f} "
        f"shared_total_build_s={float(r['shared_total_build_s']):.9f} "
        f"workspace_build_s={float(r['workspace_build_s']):.9f} "
        f"workspace_mib={float(r['workspace_mib']):.6f}"
    )
if rows:
    print('summary median_shared_vs_legacy_speedup=' +
          f"{statistics.median(float(r['shared_vs_legacy_speedup']) for r in rows):.6f}")
PY

echo "results=$out"
echo "metadata=$meta"
