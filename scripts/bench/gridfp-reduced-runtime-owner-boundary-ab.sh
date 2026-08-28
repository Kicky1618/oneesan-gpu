#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; NGPU="${NGPU:-2}"; BLOCKS="${BLOCKS:-256}"; MOD="${MOD:-4294967291}"
REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-1}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( W < 8 || W > 28 || W % 2 != 0 )); then echo "W must be even and in [8,28]" >&2; exit 2; fi
if (( NGPU < 2 || NGPU > 16 || BLOCKS < 1 || REPEATS < 1 || WARMUP < 0 )); then echo "invalid NGPU/BLOCKS/REPEATS/WARMUP" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_reduced_runtime_owner_boundary_ab_w${W}_g${NGPU}_b${BLOCKS}}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-boundary-proof.sh" >"$LOGDIR/owner-boundary-proof.out" 2>"$LOGDIR/owner-boundary-proof.err"
build_one(){ local boundary="$1" bin="$2"; MODE=two-row-runtime-multigpu RUNTIME_CACHE_EDGES=1 RUNTIME_FAST_P32M5_MOD=1 RUNTIME_POLL_GLOBAL_ERROR=0 RUNTIME_PACK_SHARED_KEYS=1 RUNTIME_FAST_DIV64=1 RUNTIME_PRIMITIVE_RANK_SETBITS=1 RUNTIME_BROADWORD_SUPPORT=1 RUNTIME_OWNER_RECIPROCAL=0 RUNTIME_OWNER_FROM_BOUNDARIES="$boundary" ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$bin" bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" >"$LOGDIR/boundary${boundary}.build.out" 2>"$LOGDIR/boundary${boundary}.build.err"; }
BIN0="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_owner_boundary0_w${W}_g${NGPU}_b${BLOCKS}"; BIN1="$ONEESAN_BUILD_DIR/gridfp_reduced_runtime_owner_boundary1_w${W}_g${NGPU}_b${BLOCKS}"; build_one 0 "$BIN0"; build_one 1 "$BIN1"
run_one(){ local boundary="$1" bin="$2" rep="$3" phase="$4"; local out="$LOGDIR/boundary${boundary}_${phase}${rep}.out" err="$LOGDIR/boundary${boundary}_${phase}${rep}.err"; "$bin" "$W" "$NGPU" "$BLOCKS" "$MOD" >"$out" 2>"$err"; local line; line="$(grep '^gridfp-reduced-two-row-runtime-multigpu ' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }; grep -Fq ' exact=OK' <<<"$line" || { echo "$line" >&2; exit 4; }; if [[ "$phase" == run ]]; then local wall; wall="$(sed -nE 's/.* wall_ms=([^[:space:]]+).*/\1/p' <<<"$line")"; [[ -n "$wall" ]] || exit 5; printf '%s\t%s\t%s\n' "$boundary" "$rep" "$wall" >>"$RESULT"; fi; }
for ((r=1;r<=WARMUP;++r)); do run_one 0 "$BIN0" "$r" warmup; run_one 1 "$BIN1" "$r" warmup; done
printf 'owner_from_boundaries\trepeat\twall_ms\n' >"$RESULT"
for ((r=1;r<=REPEATS;++r)); do if ((r&1)); then order=(0 1); else order=(1 0); fi; for boundary in "${order[@]}"; do [[ "$boundary" == 0 ]] && bin="$BIN0" || bin="$BIN1"; echo "=== runtime owner-from-boundaries=$boundary run $r/$REPEATS ===" >&2; run_one "$boundary" "$bin" "$r" run; done; done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in ('0','1'):
 xs=[float(r['wall_ms']) for r in rows if r['owner_from_boundaries']==mode]
 if not xs: raise SystemExit(f'missing owner_from_boundaries={mode}')
 out.append({'owner_from_boundaries':mode,'repeats':len(xs),'wall_ms_median':f'{statistics.median(xs):.9f}','wall_ms_min':f'{min(xs):.9f}','wall_ms_max':f'{max(xs):.9f}'})
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=('owner_from_boundaries','repeats','wall_ms_median','wall_ms_min','wall_ms_max'),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['owner_from_boundaries']:r for r in out}; old=float(q['0']['wall_ms_median']); new=float(q['1']['wall_ms_median'])
print(f'runtime_owner_boundary_wall_speedup={old/new:.6f}x'); print(f'runtime_owner_boundary_wall_delta_pct={(new/old-1)*100:.4f}%'); print('runtime_owner_boundary_old=weighted_midpoint_recompute'); print('runtime_owner_boundary_new=owner_begin_boundary_scan'); print('runtime_owner_boundary_max_global_boundary_loads=15'); print('runtime_owner_boundary_reciprocal=0'); print(f'summary={dst}')
PY
if [[ "$PTXAS_VERBOSE" == 1 ]]; then echo '--- ptxas weighted owner ---' >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/boundary0.build.err" >&2 || true; echo '--- ptxas boundary owner ---' >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/boundary1.build.err" >&2 || true; fi
echo "gridfp-reduced-runtime-owner-boundary-ab OK W=$W ngpu=$NGPU blocks=$BLOCKS repeats=$REPEATS result=$RESULT" >&2
