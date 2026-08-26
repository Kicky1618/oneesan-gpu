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
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_worker_dense_compare}"

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
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-worker-dense-compare-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_worker_dense_compare_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/dense-compare-n${N}-${ts}.tsv"
meta="$OUT_DIR/dense-compare-n${N}-${ts}.meta"

file_sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
objective=flat-vs-dense-exact-v5.30-plan
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'workers\tdomain_size\texact_pages_2m\texact_pages_4k\texact_transitions\tcandidate_evaluations\taccepted_moves\tparent_max_worker_cells\tfinal_max_worker_cells\tflat_build_s\tdense_build_s\tdense_vs_flat_speedup\tdense_index_mib\tdense_index_build_s\tflat_delta_peak_entries\tdense_delta_peak_entries\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"; domain="${cfg#*:}"
  echo "dense-compare n=$N workers=$workers domain_size=$domain" >&2
  line="$($bin "$N" "$workers" "$domain" 2> >(tee /dev/stderr) | tail -n1)"
  [[ "$(field "$line" objective)" == flat-vs-dense-exact-v5.30-plan ]] || {
    echo "objective provenance mismatch cfg=$cfg" >&2; exit 4;
  }
  [[ "$(field "$line" identical_schedule)" == 1 ]] || {
    echo "dense final schedule mismatch cfg=$cfg" >&2; exit 4;
  }
  [[ "$(field "$line" identical_trace)" == 1 ]] || {
    echo "dense search trace mismatch cfg=$cfg" >&2; exit 4;
  }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$workers" "$domain" \
    "$(field "$line" exact_pages_2m)" \
    "$(field "$line" exact_pages_4k)" \
    "$(field "$line" exact_transitions)" \
    "$(field "$line" candidate_evaluations)" \
    "$(field "$line" accepted_moves)" \
    "$(field "$line" parent_max_worker_cells)" \
    "$(field "$line" final_max_worker_cells)" \
    "$(field "$line" flat_build_s)" \
    "$(field "$line" dense_build_s)" \
    "$(field "$line" dense_vs_flat_speedup)" \
    "$(field "$line" dense_index_mib)" \
    "$(field "$line" dense_index_build_s)" \
    "$(field "$line" flat_delta_peak_entries)" \
    "$(field "$line" dense_delta_peak_entries)" \
    "$line" >>"$out"
done

python3 - "$out" <<'PY'
import csv, statistics, sys
rows=list(csv.DictReader(open(sys.argv[1], newline=''), delimiter='\t'))
for r in rows:
    speed=float(r['dense_vs_flat_speedup'])
    cls='dense_faster' if speed > 1.02 else ('dense_slower' if speed < 0.98 else 'near_tie')
    print(
        f"comparison workers={r['workers']} domain_size={r['domain_size']} "
        f"classification={cls} speedup={speed:.6f} "
        f"flat_build_s={float(r['flat_build_s']):.9f} "
        f"dense_build_s={float(r['dense_build_s']):.9f} "
        f"dense_index_mib={float(r['dense_index_mib']):.6f} "
        f"dense_index_build_s={float(r['dense_index_build_s']):.9f}"
    )
if rows:
    speeds=[float(r['dense_vs_flat_speedup']) for r in rows]
    print(f"summary median_dense_vs_flat_speedup={statistics.median(speeds):.6f}")
PY

echo "results=$out"
echo "metadata=$meta"
