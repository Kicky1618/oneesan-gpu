#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)";[[ -n "$ROOT" ]]||ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."&&pwd)";cd "$ROOT"
N="${N:-27}";ARCH="${ARCH:-native}";CONFIGS="${CONFIGS:-32:16 64:32 96:48 128:64}";MAX_RUN="${MAX_RUN:-4}";MAX_SWAP="${MAX_SWAP:-4}";BUILD="${BUILD:-1}";OUT_DIR="${OUT_DIR:-$ROOT/work/bench_ramstream32_cpu_low_worker_augmented}"
if((N<2||N>27));then exit 2;fi;for x in MAX_RUN MAX_SWAP;do v="${!x}";[[ "$v" =~ ^[1-9][0-9]*$ ]]&&((v<=64))||exit 2;done
read -r -a configs<<<"$CONFIGS";for cfg in "${configs[@]}";do [[ "$cfg" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]]||exit 2;((BASH_REMATCH[2]<=BASH_REMATCH[1]))||exit 2;done
if[[ "$BUILD" != 0 ]];then N="$N" ARCH="$ARCH" bash scripts/build/gridfp-ramstream32-cpu-low-worker-augmented-plan.sh;fi
bin="$ROOT/build/ramstream32_cpu_low_worker_augmented_plan_n${N}";[[ -x "$bin" ]]||exit 3;mkdir -p "$OUT_DIR";ts="$(date -u +%Y%m%dT%H%M%SZ)";out="$OUT_DIR/augmented-n${N}-${ts}.tsv";meta="$OUT_DIR/augmented-n${N}-${ts}.meta"
field(){ local line="$1" key="$2" token;for token in $line;do [[ "$token" == "$key="* ]]&&{ printf '%s\n' "${token#*=}";return 0;};done;return 1;}
cat >"$meta" <<EOF
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
n=$N
configs=$CONFIGS
max_run=$MAX_RUN
max_swap=$MAX_SWAP
objective=exact-neutral-augmented-v5.36-plan
EOF
printf 'workers\tdomain_size\tbaseline_pages_2m\tbaseline_pages_4k\tbaseline_transitions\tbaseline_max_worker_cells\taugmented_pages_2m\taugmented_pages_4k\taugmented_transitions\taugmented_max_worker_cells\tpages_2m_delta\tpages_4k_delta\ttransition_delta\taugmented_rounds\texact_schedule_changes\texact_primary_improvements\texact_profile_improvements\tneutral_moves\tneutral_candidates\taugmented_build_s\tworkspace_build_s\traw\n' >"$out"
for cfg in "${configs[@]}";do workers="${cfg%%:*}";domain="${cfg#*:}";echo "augmented n=$N workers=$workers domain=$domain" >&2;line="$($bin "$N" "$workers" "$domain" "$MAX_RUN" "$MAX_SWAP" 2> >(tee /dev/stderr)|tail -n1)";[[ "$(field "$line" objective)" == exact-neutral-augmented-v5.36-plan ]]||exit 4;[[ "$(field "$line" limits_clear)" == 1 ]]||exit 4
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$workers" "$domain" "$(field "$line" baseline_pages_2m)" "$(field "$line" baseline_pages_4k)" "$(field "$line" baseline_transitions)" "$(field "$line" baseline_max_worker_cells)" "$(field "$line" augmented_pages_2m)" "$(field "$line" augmented_pages_4k)" "$(field "$line" augmented_transitions)" "$(field "$line" augmented_max_worker_cells)" "$(field "$line" pages_2m_delta)" "$(field "$line" pages_4k_delta)" "$(field "$line" transition_delta)" "$(field "$line" augmented_rounds)" "$(field "$line" exact_schedule_changes)" "$(field "$line" exact_primary_improvements)" "$(field "$line" exact_profile_improvements)" "$(field "$line" neutral_moves)" "$(field "$line" neutral_candidates)" "$(field "$line" augmented_build_s)" "$(field "$line" workspace_build_s)" "$line" >>"$out";done
python3 - "$out" <<'PY'
import csv,sys
for r in csv.DictReader(open(sys.argv[1],newline=''),delimiter='\t'):
 b=(int(r['baseline_pages_2m']),int(r['baseline_pages_4k']),int(r['baseline_transitions']));a=(int(r['augmented_pages_2m']),int(r['augmented_pages_4k']),int(r['augmented_transitions']));bm=int(r['baseline_max_worker_cells']);am=int(r['augmented_max_worker_cells']);moves=int(r['neutral_moves']);rounds=int(r['augmented_rounds'])
 if a>b or am>bm:raise SystemExit('augmented regression')
 if a<b and moves:cls='augmented_plateau_escape'
 elif a<b:cls='augmented_exact_improvement'
 elif am<bm:cls='augmented_load_improvement'
 elif moves:cls='neutral_fixedpoint_change'
 else:cls='no_change'
 print(f"comparison workers={r['workers']} domain_size={r['domain_size']} classification={cls} rounds={rounds} neutral_moves={moves} exact_primary_improvements={r['exact_primary_improvements']} pages_2m_delta={a[0]-b[0]} pages_4k_delta={a[1]-b[1]} transition_delta={a[2]-b[2]} max_worker_delta={am-bm} build_s={float(r['augmented_build_s']):.9f}")
PY
echo "results=$out";echo "metadata=$meta"
