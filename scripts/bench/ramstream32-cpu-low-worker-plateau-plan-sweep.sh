#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"; [[ -n "$ROOT" ]] || ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
N="${N:-27}"; ARCH="${ARCH:-native}"; CONFIGS="${CONFIGS:-32:16 64:32 96:48 128:64}"; MAX_RUN="${MAX_RUN:-4}"; MAX_SWAP="${MAX_SWAP:-4}"; BUILD="${BUILD:-1}"; OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_worker_plateau}"
if ((N<2||N>27));then echo "N must be in 2..27" >&2;exit 2;fi
for x in MAX_RUN MAX_SWAP;do v="${!x}";[[ "$v" =~ ^[1-9][0-9]*$ ]]&&((v<=64))||{ echo "$x must be 1..64" >&2;exit 2;};done
read -r -a configs <<<"$CONFIGS";for cfg in "${configs[@]}";do [[ "$cfg" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]]||exit 2;((BASH_REMATCH[2]<=BASH_REMATCH[1]))||exit 2;done
if [[ "$BUILD" != 0 ]];then N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-worker-plateau-plan.sh;fi
bin="$ROOT/build/ramstream32_cpu_low_worker_plateau_plan_n${N}";[[ -x "$bin" ]]||exit 3
mkdir -p "$OUT_DIR";ts="$(date -u +%Y%m%dT%H%M%SZ)";out="$OUT_DIR/plateau-n${N}-${ts}.tsv";meta="$OUT_DIR/plateau-n${N}-${ts}.meta"
field(){ local line="$1" key="$2" token;for token in $line;do [[ "$token" == "$key="* ]]&&{ printf '%s\n' "${token#*=}";return 0;};done;return 1;}
cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
n=$N
configs=$CONFIGS
max_run=$MAX_RUN
max_swap=$MAX_SWAP
objective=neutral-load-bridge-v5.35-plan
EOF
printf 'workers\tdomain_size\tbaseline_order\tbaseline_pages_2m\tbaseline_pages_4k\tbaseline_transitions\tbaseline_max_worker_cells\tneutral_accepted_moves\tneutral_candidate_evaluations\tneutral_exact_rejections\tneutral_profile_rejections\tneutral_max_worker_cells_after\tneutral_build_s\trefixed_order\tfinal_pages_2m\tfinal_pages_4k\tfinal_transitions\tfinal_max_worker_cells\tfinal_2m_delta\tfinal_4k_delta\tfinal_transition_delta\trefixed_rs_rounds\trefixed_sr_rounds\traw\n' >"$out"
for cfg in "${configs[@]}";do workers="${cfg%%:*}";domain="${cfg#*:}";echo "plateau n=$N workers=$workers domain=$domain" >&2;line="$($bin "$N" "$workers" "$domain" "$MAX_RUN" "$MAX_SWAP" 2> >(tee /dev/stderr)|tail -n1)";[[ "$(field "$line" objective)" == neutral-load-bridge-v5.35-plan ]]||exit 4;[[ "$(field "$line" limits_clear)" == 1 ]]||exit 4
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$workers" "$domain" "$(field "$line" baseline_order)" "$(field "$line" baseline_pages_2m)" "$(field "$line" baseline_pages_4k)" "$(field "$line" baseline_transitions)" "$(field "$line" baseline_max_worker_cells)" "$(field "$line" neutral_accepted_moves)" "$(field "$line" neutral_candidate_evaluations)" "$(field "$line" neutral_exact_rejections)" "$(field "$line" neutral_profile_rejections)" "$(field "$line" neutral_max_worker_cells_after)" "$(field "$line" neutral_build_s)" "$(field "$line" refixed_order)" "$(field "$line" final_pages_2m)" "$(field "$line" final_pages_4k)" "$(field "$line" final_transitions)" "$(field "$line" final_max_worker_cells)" "$(field "$line" final_2m_delta)" "$(field "$line" final_4k_delta)" "$(field "$line" final_transition_delta)" "$(field "$line" refixed_rs_rounds)" "$(field "$line" refixed_sr_rounds)" "$line" >>"$out";done
python3 - "$out" <<'PY'
import csv,sys
for r in csv.DictReader(open(sys.argv[1],newline=''),delimiter='\t'):
 b=(int(r['baseline_pages_2m']),int(r['baseline_pages_4k']),int(r['baseline_transitions']));f=(int(r['final_pages_2m']),int(r['final_pages_4k']),int(r['final_transitions']));moves=int(r['neutral_accepted_moves']);bm=int(r['baseline_max_worker_cells']);fm=int(r['final_max_worker_cells'])
 if f>b or fm>bm: raise SystemExit('plateau regression')
 if moves and f<b: cls='plateau_escape_improvement'
 elif moves and f==b and fm<bm: cls='load_profile_only'
 elif moves: cls='neutral_rearrangement'
 else: cls='no_neutral_move'
 print(f"comparison workers={r['workers']} domain_size={r['domain_size']} classification={cls} neutral_moves={moves} final_2m_delta={f[0]-b[0]} final_4k_delta={f[1]-b[1]} final_transition_delta={f[2]-b[2]} max_worker_delta={fm-bm} neutral_build_s={float(r['neutral_build_s']):.9f}")
PY
echo "results=$out";echo "metadata=$meta"
