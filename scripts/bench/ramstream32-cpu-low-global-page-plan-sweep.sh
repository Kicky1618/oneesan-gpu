#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

N="${N:-27}"
ARCH="${ARCH:-native}"
# Include 2-domain controls and 3/4-domain cases.  With exactly two domains
# there is only one boundary, so local and global-unique page objectives are
# mathematically identical and v5.24 cannot improve on v5.23.
CONFIGS="${CONFIGS:-32:16 48:16 64:16 64:32 96:32 128:32}"
BUILD="${BUILD:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_global_page_plan}"

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
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-global-page-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_global_page_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/global-page-plan-n${N}-${ts}.tsv"
meta="$OUT_DIR/global-page-plan-n${N}-${ts}.meta"

file_sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
objective=global-unique-max-guard-page-sum-v5.24-plan
note=two-domain rows are controls because one boundary makes local and global page objectives identical
binary_sha256=$(file_sha256 "$bin")
EOF

printf 'workers\tdomain_size\tdomains\trefined_max_worker_cells\tlocal_max_worker_cells\tglobal_max_worker_cells\trefined_pages_2m\trefined_pages_4k\tlocal_pages_2m\tlocal_pages_4k\tglobal_pages_2m\tglobal_pages_4k\tlocal_boundary_moves\tlocal_candidate_evaluations\tlocal_max_guard_rejections\tlocal_relaxed_moves\tglobal_boundary_moves\tglobal_candidate_evaluations\tglobal_max_guard_rejections\tglobal_page_improving_moves\tglobal_page_tie_load_moves\tglobal_relaxed_moves\tglobal_build_s\tglobal_total_build_s\tscore_mask_index_mib\tscore_mask_index_build_s\tlocal_mask_index_build_s\tglobal_mask_index_build_s\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"; domain="${cfg#*:}"
  echo "global-page-plan n=$N workers=$workers domain_size=$domain" >&2
  line="$($bin "$N" "$workers" "$domain" 2> >(tee /dev/stderr) | tail -n1)"
  [[ "$(field "$line" objective)" == global-unique-max-guard-page-sum-v5.24-plan ]] || {
    echo "objective provenance mismatch cfg=$cfg" >&2; exit 4;
  }
  [[ "$(field "$line" workers)" == "$workers" ]] || { echo "worker provenance mismatch cfg=$cfg" >&2; exit 4; }
  [[ "$(field "$line" domain_size)" == "$domain" ]] || { echo "domain provenance mismatch cfg=$cfg" >&2; exit 4; }

  domains="$(field "$line" domains)"
  python3 - "$domains" \
    "$(field "$line" refined_max_worker_cells)" \
    "$(field "$line" local_max_worker_cells)" \
    "$(field "$line" global_max_worker_cells)" \
    "$(field "$line" local_global_pages_2m)" \
    "$(field "$line" local_global_pages_4k)" \
    "$(field "$line" global_pages_2m)" \
    "$(field "$line" global_pages_4k)" <<'PY'
import sys
domains,refined,local,glob,l2,l4,g2,g4=map(int,sys.argv[1:])
if local > refined or glob > local:
    raise SystemExit('max-worker regression')
if (g2,g4) > (l2,l4):
    raise SystemExit('global unique page regression')
if domains == 2 and (g2,g4) != (l2,l4):
    raise SystemExit('two-domain local/global equivalence violated')
PY

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$workers" "$domain" "$domains" \
    "$(field "$line" refined_max_worker_cells)" \
    "$(field "$line" local_max_worker_cells)" \
    "$(field "$line" global_max_worker_cells)" \
    "$(field "$line" refined_global_pages_2m)" \
    "$(field "$line" refined_global_pages_4k)" \
    "$(field "$line" local_global_pages_2m)" \
    "$(field "$line" local_global_pages_4k)" \
    "$(field "$line" global_pages_2m)" \
    "$(field "$line" global_pages_4k)" \
    "$(field "$line" local_boundary_moves)" \
    "$(field "$line" local_candidate_evaluations)" \
    "$(field "$line" local_max_guard_rejections)" \
    "$(field "$line" local_page_improve_sum_increase_moves)" \
    "$(field "$line" global_boundary_moves)" \
    "$(field "$line" global_candidate_evaluations)" \
    "$(field "$line" global_max_guard_rejections)" \
    "$(field "$line" global_page_improving_moves)" \
    "$(field "$line" global_page_tie_load_moves)" \
    "$(field "$line" global_page_improve_sum_increase_moves)" \
    "$(field "$line" global_build_s)" \
    "$(field "$line" global_total_build_s)" \
    "$(field "$line" score_mask_index_mib)" \
    "$(field "$line" score_mask_index_build_s)" \
    "$(field "$line" local_mask_index_build_s)" \
    "$(field "$line" global_mask_index_build_s)" \
    "$line" >>"$out"
done

python3 - "$out" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as f:
    for r in csv.DictReader(f, delimiter='\t'):
        domains=int(r['domains'])
        local=(int(r['local_pages_2m']),int(r['local_pages_4k']))
        glob=(int(r['global_pages_2m']),int(r['global_pages_4k']))
        refined=(int(r['refined_pages_2m']),int(r['refined_pages_4k']))
        if domains <= 2:
            cls='single_boundary_equivalent'
        elif glob < local:
            cls='global_beats_local'
        elif glob == local:
            cls='global_ties_local'
        else:
            cls='invalid_regression'
        evals=int(r['global_candidate_evaluations'])
        rejects=int(r['global_max_guard_rejections'])
        frac=rejects/evals if evals else 0.0
        print(
            f"comparison workers={r['workers']} domain_size={r['domain_size']} domains={domains} "
            f"classification={cls} "
            f"refined_pages_2m={refined[0]} refined_pages_4k={refined[1]} "
            f"local_pages_2m={local[0]} local_pages_4k={local[1]} "
            f"global_pages_2m={glob[0]} global_pages_4k={glob[1]} "
            f"global_vs_local_2m_delta={glob[0]-local[0]} "
            f"global_vs_local_4k_delta={glob[1]-local[1]} "
            f"global_candidate_evaluations={evals} "
            f"global_max_guard_reject_fraction={frac:.9f} "
            f"global_boundary_moves={r['global_boundary_moves']} "
            f"global_build_s={float(r['global_build_s']):.9f} "
            f"score_mask_index_build_s={float(r['score_mask_index_build_s']):.9f}"
        )
PY

echo "results=$out"
echo "metadata=$meta"
