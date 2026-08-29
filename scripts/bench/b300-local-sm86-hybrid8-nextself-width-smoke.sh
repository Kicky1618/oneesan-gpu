#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'local hybrid8 next-self width smoke targets n=27' >&2; exit 2; }
ARCH="${ARCH:-sm_86}"
NGPU="${NGPU:-1}"
ROWS="${ROWS:-1}"
THREADS="${THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-1024}"
MAX_WINDOW="${MAX_WINDOW:-14}"
MOD="${MOD:-4294967291}"
HYBRID_THRESHOLD="${HYBRID_THRESHOLD:-0}"
WIDTH_LIST="${WIDTH_LIST:-1 2 4 8}"
WORK="${WORK:-$ONEESAN_ROOT/work/b300_local_sm86_hybrid8_nextself_width_smoke}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$WORK" "$ONEESAN_BUILD_DIR"

[[ "$NGPU" =~ ^[1-8]$ ]] || { echo 'NGPU must be 1..8' >&2; exit 2; }
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32 && THREADS<=1024 && THREADS%32==0)) || { echo 'THREADS must be warp multiple 32..1024' >&2; exit 2; }
[[ "$TARGET_MIB" =~ ^[1-9][0-9]*$ ]] || { echo 'TARGET_MIB must be positive' >&2; exit 2; }
[[ "$MAX_WINDOW" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_WINDOW must be positive' >&2; exit 2; }
[[ "$MOD" =~ ^[1-9][0-9]*$ ]] || { echo 'MOD must be positive' >&2; exit 2; }
[[ "$HYBRID_THRESHOLD" =~ ^[0-9]+$ ]] || { echo 'HYBRID_THRESHOLD must be non-negative' >&2; exit 2; }
widths=()
for w in $WIDTH_LIST; do
  case "$w" in 1|2|4|8) ;; *) echo "bad width=$w" >&2; exit 2;; esac
  seen=0; for old in "${widths[@]}"; do [[ "$old" == "$w" ]] && seen=1; done
  ((seen)) || widths+=("$w")
done
((${#widths[@]})) || { echo 'WIDTH_LIST empty' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU visible GPU(s), found $visible" >&2; exit 2; }

printf 'config\twidth\tbinary\tbuild_err\n' >"$WORK/binaries.tsv"
build_one(){
  local config="$1" width="$2" nextself="$3"
  local bin="$ONEESAN_BUILD_DIR/b300_local_sm86_hybrid8_${config}_t${HYBRID_THRESHOLD}_n27"
  local err="$WORK/${config}.build.err" out="$WORK/${config}.build.out" driver="$WORK/${config}.build.driver.err"
  echo "=== build config=$config width=$width arch=$ARCH ===" >&2
  if [[ "$nextself" == 0 ]]; then
    N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK=0 RECURRENCE_ILP=2 \
      RECURRENCE_HYBRID_ILP8=1 RECURRENCE_HYBRID_ILP8_MIN_STATES="$HYBRID_THRESHOLD" \
      RECURRENCE_HYBRID_ILP8_NEXTSELF=0 RECURRENCE_HYBRID_ILP8_NEXTSELF_WIDTH=8 \
      RANDOM_CG=0 RANDOM_CG_L2_FETCH_BYTES=0 PREFETCH_L2=0 DUALMASK=0 CLOSURE_BATCH=0 \
      MAXRREGCOUNT=0 PTXAS_VERBOSE=1 BUILD_ERR="$err" \
      bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$out" 2>"$driver"
  else
    N=27 ARCH="$ARCH" OUT="$bin" HIGH_DROP_CHUNK=0 RECURRENCE_ILP=2 \
      RECURRENCE_HYBRID_ILP8=1 RECURRENCE_HYBRID_ILP8_MIN_STATES="$HYBRID_THRESHOLD" \
      RECURRENCE_HYBRID_ILP8_NEXTSELF=1 RECURRENCE_HYBRID_ILP8_NEXTSELF_WIDTH="$width" \
      RANDOM_CG=0 RANDOM_CG_L2_FETCH_BYTES=0 PREFETCH_L2=0 DUALMASK=0 CLOSURE_BATCH=0 \
      MAXRREGCOUNT=0 PTXAS_VERBOSE=1 BUILD_ERR="$err" \
      bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$out" 2>"$driver"
  fi
  [[ -x "$bin" ]] || { echo "binary missing config=$config" >&2; exit 3; }
  grep -Fq 'recurrence_hybrid_ilp8=1' "$out" || { echo "hybrid marker missing config=$config" >&2; exit 3; }
  if [[ "$nextself" == 1 ]]; then
    grep -Fq "recurrence_hybrid_ilp8_nextself=1 recurrence_hybrid_ilp8_nextself_width=$width" "$out" || {
      echo "next-self width summary missing config=$config" >&2; exit 3;
    }
  fi
  printf '%s\t%s\t%s\t%s\n' "$config" "$width" "$bin" "$err" >>"$WORK/binaries.tsv"
}

build_one plain 0 0
for w in "${widths[@]}"; do build_one "w${w}" "$w" 1; done

printf 'config\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$WORK/ptxas.tsv"
while IFS=$'\t' read -r config width bin err; do
  [[ "$config" == config ]] && continue
  python3 "$PARSER" "$err" --label "$config" --contains main_pull_kernel_ilp2 >>"$WORK/ptxas.tsv" || true
  python3 "$PARSER" "$err" --label "$config" --contains main_pull_kernel_ilp8_hybrid >>"$WORK/ptxas.tsv" || true
done <"$WORK/binaries.tsv"

field(){
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}
run_one(){
  local config="$1" width="$2" bin="$3" so="$WORK/${config}.run.out" se="$WORK/${config}.run.err"
  echo "=== run config=$config width=$width rows=$ROWS threads=$THREADS ngpu=$NGPU ===" >&2
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" \
    "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line residue wall high fallback
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$config missing backend result" >&2; tail -n 80 "$se" >&2 || true; exit 4; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  high="$(field high_rec_groups "$line")"; fallback="$(field high_rec_fallback_groups "$line")"
  [[ "$residue" =~ ^[0-9]+$ && -n "$wall" ]] || { echo "$config missing residue/wall" >&2; exit 4; }
  [[ "$high" =~ ^[1-9][0-9]*$ ]] || { echo "$config hybrid path inactive high_rec_groups=${high:-NA}" >&2; exit 4; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$config" "$width" "$residue" "$wall" "$high" "${fallback:-NA}"
}

printf 'config\twidth\tresidue\twall_s\thigh_rec_groups\thigh_rec_fallback_groups\n' >"$WORK/results.tsv"
while IFS=$'\t' read -r config width bin err; do
  [[ "$config" == config ]] && continue
  run_one "$config" "$width" "$bin" | tee -a "$WORK/results.tsv"
done <"$WORK/binaries.tsv"

python3 - "$WORK/results.tsv" "$WORK/ptxas.tsv" <<'PY'
import csv,sys
results=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
resources=list(csv.DictReader(open(sys.argv[2]),delimiter='\t'))
if len(results)<2: raise SystemExit('need plain plus at least one width result')
res={r['residue'] for r in results}
if len(res)!=1: raise SystemExit('FATAL local hybrid8 next-self width residue mismatch '+repr({r['config']:r['residue'] for r in results}))
by={}
for r in resources:
    try: by.setdefault(r['config'],[]).append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (KeyError,ValueError): pass
missing=[r['config'] for r in results if len(by.get(r['config'],[]))<2]
if missing: raise SystemExit('missing ILP2/ILP8 ptxas records '+repr(missing))
for r in results:
    rv=by[r['config']]
    regs=max(x[0] for x in rv); ss=max(x[1] for x in rv); sl=max(x[2] for x in rv)
    print(f"LOCAL_SM86_WIDTH config={r['config']} width={r['width']} wall_s={r['wall_s']} regs_max={regs} spill_store={ss} spill_load={sl}",file=sys.stderr)
best=min(results,key=lambda r:float(r['wall_s']))
print('b300_local_sm86_hybrid8_nextself_width_exact_match=1')
print('b300_local_sm86_hybrid8_nextself_width_ptxas_complete=1')
print(f"b300_local_sm86_hybrid8_nextself_width_residue={next(iter(res))}")
print(f"b300_local_sm86_hybrid8_nextself_width_local_best={best['config']}")
PY

cat "$WORK/results.tsv"
cat "$WORK/ptxas.tsv"
echo "b300-local-sm86-hybrid8-nextself-width-smoke OK widths=${widths[*]} rows=$ROWS threads=$THREADS ngpu=$NGPU threshold=$HYBRID_THRESHOLD arch=$ARCH results=$WORK/results.tsv" >&2
