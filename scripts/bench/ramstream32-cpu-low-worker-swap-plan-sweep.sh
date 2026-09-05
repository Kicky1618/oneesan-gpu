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
SWAPS="${SWAPS:-1 2 4 8}"
RUN_MAX="${RUN_MAX:-4}"
BUILD="${BUILD:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_worker_swap_plan}"

if (( N < 2 || N > 27 )); then echo "N must be in 2..27" >&2; exit 2; fi
[[ "$RUN_MAX" =~ ^[1-9][0-9]*$ ]] && (( RUN_MAX <= 64 )) || { echo "RUN_MAX must be 1..64" >&2; exit 2; }
read -r -a configs <<<"$CONFIGS"
read -r -a swaps <<<"$SWAPS"
if ((${#configs[@]} == 0 || ${#swaps[@]} == 0)); then echo "CONFIGS and SWAPS must be non-empty" >&2; exit 2; fi
for cfg in "${configs[@]}"; do
  [[ "$cfg" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]] || { echo "invalid CONFIGS entry: $cfg" >&2; exit 2; }
  (( BASH_REMATCH[2] <= BASH_REMATCH[1] )) || { echo "domain size exceeds workers: $cfg" >&2; exit 2; }
done
for s in "${swaps[@]}"; do
  [[ "$s" =~ ^[1-9][0-9]*$ ]] && (( s <= 64 )) || { echo "invalid SWAPS entry: $s" >&2; exit 2; }
done

if [[ "$BUILD" != 0 ]]; then
  N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-worker-swap-plan.sh
fi
bin="$ROOT/build/ramstream32_cpu_low_worker_swap_plan_n${N}"
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 3; }

mkdir -p "$OUT_DIR"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$OUT_DIR/swap-plan-n${N}-${ts}.tsv"
meta="$OUT_DIR/swap-plan-n${N}-${ts}.meta"
field() { local line="$1" key="$2" token; for token in $line; do if [[ "$token" == "$key="* ]]; then printf '%s\n' "${token#*=}"; return 0; fi; done; return 1; }

cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
host=$(hostname 2>/dev/null || echo unknown)
n=$N
arch=$ARCH
configs=$CONFIGS
swaps=$SWAPS
run_max=$RUN_MAX
objective=bounded-swap-global-unique-v5.33-plan
EOF

printf 'workers\tdomain_size\tmax_swap\tmax_run\tselected_source\tbefore_pages_2m\tbefore_pages_4k\tbefore_transitions\tafter_pages_2m\tafter_pages_4k\tafter_transitions\tpages_2m_delta\tpages_4k_delta\ttransition_delta\taccepted_swaps\tmoved_jobs\tmax_left_used\tmax_right_used\tswap_candidate_evaluations\tswap_cap_rejections\tmove_limit_hit\tswap_build_s\traw\n' >"$out"

for cfg in "${configs[@]}"; do
  workers="${cfg%%:*}"; domain="${cfg#*:}"
  for s in "${swaps[@]}"; do
    echo "swap-plan n=$N workers=$workers domain_size=$domain max_swap=$s run_max=$RUN_MAX" >&2
    line="$($bin "$N" "$workers" "$domain" "$s" "$RUN_MAX" 2> >(tee /dev/stderr) | tail -n1)"
    [[ "$(field "$line" objective)" == bounded-swap-global-unique-v5.33-plan ]] || exit 4
    [[ "$(field "$line" max_swap)" == "$s" ]] || exit 4
    [[ "$(field "$line" max_run)" == "$RUN_MAX" ]] || exit 4
    [[ "$(field "$line" move_limit_hit)" == 0 ]] || { echo "swap move limit hit cfg=$cfg max_swap=$s" >&2; exit 5; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$workers" "$domain" "$s" "$RUN_MAX" "$(field "$line" selected_source)" \
      "$(field "$line" before_pages_2m)" "$(field "$line" before_pages_4k)" "$(field "$line" before_transitions)" \
      "$(field "$line" after_pages_2m)" "$(field "$line" after_pages_4k)" "$(field "$line" after_transitions)" \
      "$(field "$line" pages_2m_delta)" "$(field "$line" pages_4k_delta)" "$(field "$line" transition_delta)" \
      "$(field "$line" accepted_swaps)" "$(field "$line" moved_jobs)" \
      "$(field "$line" max_left_used)" "$(field "$line" max_right_used)" \
      "$(field "$line" swap_candidate_evaluations)" "$(field "$line" swap_cap_rejections)" \
      "$(field "$line" move_limit_hit)" "$(field "$line" swap_build_s)" "$line" >>"$out"
  done
done

python3 - "$out" <<'PY'
import csv,collections,sys
rows=list(csv.DictReader(open(sys.argv[1],newline=''),delimiter='\t'))
g=collections.defaultdict(list)
for r in rows:g[(r['workers'],r['domain_size'])].append(r)
for key,rs in g.items():
    rs.sort(key=lambda r:int(r['max_swap']))
    base=next((r for r in rs if int(r['max_swap'])==1),rs[0])
    b=(int(base['after_pages_2m']),int(base['after_pages_4k']),int(base['after_transitions']))
    for r in rs:
        before=(int(r['before_pages_2m']),int(r['before_pages_4k']),int(r['before_transitions']))
        z=(int(r['after_pages_2m']),int(r['after_pages_4k']),int(r['after_transitions']))
        ms=int(r['max_swap'])
        if ms>1 and z<b: cls='atomic_swap_improvement'
        elif z<before: cls='swap_improvement'
        else: cls='no_change'
        print(
          f"comparison workers={key[0]} domain_size={key[1]} max_swap={ms} "
          f"classification={cls} after_2m={z[0]} after_4k={z[1]} transitions={z[2]} "
          f"vs_swap1_2m_delta={z[0]-b[0]} vs_swap1_4k_delta={z[1]-b[1]} "
          f"vs_swap1_transition_delta={z[2]-b[2]} accepted_swaps={r['accepted_swaps']} "
          f"max_left_used={r['max_left_used']} max_right_used={r['max_right_used']} "
          f"build_s={float(r['swap_build_s']):.9f}"
        )
PY

echo "results=$out"
echo "metadata=$meta"
