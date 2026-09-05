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
RUNS="${RUNS:-1 2 4 8}"
BUILD="${BUILD:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_worker_run_plan}"

if (( N < 2 || N > 27 )); then echo "N must be in 2..27" >&2; exit 2; fi
read -r -a configs <<<"$CONFIGS"
read -r -a runs <<<"$RUNS"
if ((${#configs[@]} == 0 || ${#runs[@]} == 0)); then echo "CONFIGS and RUNS must be non-empty" >&2; exit 2; fi
for cfg in "${configs[@]}"; do
  [[ "$cfg" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]] || { echo "invalid CONFIGS entry: $cfg" >&2; exit 2; }
  (( BASH_REMATCH[2] <= BASH_REMATCH[1] )) || { echo "domain size exceeds workers: $cfg" >&2; exit 2; }
done
for r in "${runs[@]}"; do
  [[ "$r" =~ ^[1-9][0-9]*$ ]] || { echo "invalid RUNS entry: $r" >&2; exit 2; }
  (( r <= 64 )) || { echo "RUNS entry exceeds 64: $r" >&2; exit 2; }
done

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-worker-run-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_worker_run_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/run-plan-n${N}-${ts}.tsv"
meta="$OUT_DIR/run-plan-n${N}-${ts}.meta"
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
runs=$RUNS
objective=bounded-run-global-unique-v5.32-plan
EOF

printf 'workers\tdomain_size\tmax_run\tselected_source\tbefore_pages_2m\tbefore_pages_4k\tbefore_transitions\tafter_pages_2m\tafter_pages_4k\tafter_transitions\tpages_2m_delta\tpages_4k_delta\ttransition_delta\taccepted_runs\tmoved_jobs\tmax_run_used\tcandidate_evaluations\tcap_rejections\tmove_limit_hit\trun_build_s\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"; domain="${cfg#*:}"
  for r in "${runs[@]}"; do
    echo "run-plan n=$N workers=$workers domain_size=$domain max_run=$r" >&2
    line="$($bin "$N" "$workers" "$domain" "$r" 2> >(tee /dev/stderr) | tail -n1)"
    [[ "$(field "$line" objective)" == bounded-run-global-unique-v5.32-plan ]] || exit 4
    [[ "$(field "$line" max_run)" == "$r" ]] || exit 4
    [[ "$(field "$line" move_limit_hit)" == 0 ]] || {
      echo "move limit hit cfg=$cfg max_run=$r" >&2; exit 5;
    }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$workers" "$domain" "$r" "$(field "$line" selected_source)" \
      "$(field "$line" before_pages_2m)" "$(field "$line" before_pages_4k)" "$(field "$line" before_transitions)" \
      "$(field "$line" after_pages_2m)" "$(field "$line" after_pages_4k)" "$(field "$line" after_transitions)" \
      "$(field "$line" pages_2m_delta)" "$(field "$line" pages_4k_delta)" "$(field "$line" transition_delta)" \
      "$(field "$line" accepted_runs)" "$(field "$line" moved_jobs)" "$(field "$line" max_run_used)" \
      "$(field "$line" candidate_evaluations)" "$(field "$line" cap_rejections)" "$(field "$line" move_limit_hit)" \
      "$(field "$line" run_build_s)" "$line" >>"$out"
  done
done

python3 - "$out" <<'PY'
import csv, collections, sys
rows=list(csv.DictReader(open(sys.argv[1], newline=''), delimiter='\t'))
groups=collections.defaultdict(list)
for r in rows:
    groups[(r['workers'],r['domain_size'])].append(r)
for key, rs in groups.items():
    rs.sort(key=lambda r:int(r['max_run']))
    base=next((r for r in rs if int(r['max_run'])==1), rs[0])
    b=(int(base['after_pages_2m']),int(base['after_pages_4k']),int(base['after_transitions']))
    for r in rs:
        z=(int(r['after_pages_2m']),int(r['after_pages_4k']),int(r['after_transitions']))
        if int(r['max_run']) > 1 and z < b:
            cls='atomic_run_improvement'
        elif z < (int(r['before_pages_2m']),int(r['before_pages_4k']),int(r['before_transitions'])):
            cls='run_improvement'
        else:
            cls='no_change'
        print(
            f"comparison workers={key[0]} domain_size={key[1]} max_run={r['max_run']} "
            f"classification={cls} after_2m={z[0]} after_4k={z[1]} transitions={z[2]} "
            f"vs_run1_2m_delta={z[0]-b[0]} vs_run1_4k_delta={z[1]-b[1]} "
            f"vs_run1_transition_delta={z[2]-b[2]} accepted_runs={r['accepted_runs']} "
            f"max_run_used={r['max_run_used']} build_s={float(r['run_build_s']):.9f}"
        )
PY

echo "results=$out"
echo "metadata=$meta"
