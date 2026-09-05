#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BLOCKS="${BLOCKS:-4096}"; THREADS="${THREADS:-256}"; ITERS="${ITERS:-128}"; REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-2}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( BLOCKS < 1 || THREADS < 1 || THREADS > 1024 || ITERS < 1 || REPEATS < 1 || WARMUP < 0 )); then echo "invalid microprobe dimensions" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_runtime_owner_local_sector_parity_microprobe_b${BLOCKS}_t${THREADS}_i${ITERS}}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-owner-local-sector-parity-proof.sh" >"$LOGDIR/parity-proof.out" 2>"$LOGDIR/parity-proof.err"

SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_runtime_owner_local_sector_parity_microprobe.cu"
BIN0="$ONEESAN_BUILD_DIR/gridfp_runtime_owner_local_sector_parity_microprobe0"
BIN1="$ONEESAN_BUILD_DIR/gridfp_runtime_owner_local_sector_parity_microprobe1"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")
build_one(){ local parity="$1" bin="$2"; TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" -DRP_RUNTIME_OWNER_LOCAL_SECTOR_TABLE=1 -DRP_RUNTIME_OWNER_LOCAL_SECTOR_PARITY="$parity" -DRP_RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE=0 "$SRC" -o "$bin" >"$LOGDIR/parity${parity}.build.out" 2>"$LOGDIR/parity${parity}.build.err"; }
build_one 0 "$BIN0"; build_one 1 "$BIN1"

printf 'parity\trepeat\tkernel_ms\tns_per_call\tchecksum\n' >"$RESULT"
run_one(){ local parity="$1" bin="$2" rep="$3"; local out="$LOGDIR/parity${parity}_run${rep}.out" err="$LOGDIR/parity${parity}_run${rep}.err"; "$bin" "$BLOCKS" "$THREADS" "$ITERS" "$WARMUP" >"$out" 2>"$err"; local line; line="$(grep '^gridfp-runtime-owner-local-sector-parity-microprobe ' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }; grep -Fq " parity=$parity " <<<"$line" || { echo "unexpected parity result: $line" >&2; exit 4; }; grep -Fq ' w28_tree=0 ' <<<"$line" || { echo "parity A/B must keep w28_tree=0: $line" >&2; exit 4; }; local ms ns checksum; ms="$(sed -nE 's/.* kernel_ms=([^[:space:]]+).*/\1/p' <<<"$line")"; ns="$(sed -nE 's/.* ns_per_call=([^[:space:]]+).*/\1/p' <<<"$line")"; checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"; [[ -n "$ms" && -n "$ns" && -n "$checksum" ]] || exit 5; printf '%s\t%s\t%s\t%s\t%s\n' "$parity" "$rep" "$ms" "$ns" "$checksum" >>"$RESULT"; }
for ((r=1;r<=REPEATS;++r)); do if ((r&1)); then order=(0 1); else order=(1 0); fi; for parity in "${order[@]}"; do [[ "$parity" == 0 ]] && bin="$BIN0" || bin="$BIN1"; echo "=== W28 owner-local-sector parity=$parity run $r/$REPEATS ===" >&2; run_one "$parity" "$bin" "$r"; done; done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src, dst = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter='\t'))
out = []
checksums = {}
for mode in ('0', '1'):
    rs = [r for r in rows if r['parity'] == mode]
    if not rs:
        raise SystemExit(f'missing parity={mode}')
    cs = {r['checksum'] for r in rs}
    if len(cs) != 1:
        raise SystemExit(f'nondeterministic checksum parity={mode}: {sorted(cs)}')
    checksums[mode] = next(iter(cs))
    ms = [float(r['kernel_ms']) for r in rs]
    ns = [float(r['ns_per_call']) for r in rs]
    out.append({
        'parity': mode,
        'repeats': len(rs),
        'kernel_ms_median': f'{statistics.median(ms):.9f}',
        'ns_per_call_median': f'{statistics.median(ns):.9f}',
        'kernel_ms_min': f'{min(ms):.9f}',
        'kernel_ms_max': f'{max(ms):.9f}',
        'checksum': checksums[mode],
    })
if checksums['0'] != checksums['1']:
    raise SystemExit(f'checksum mismatch full={checksums["0"]} positive={checksums["1"]}')
with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=out[0].keys(), delimiter='\t')
    w.writeheader(); w.writerows(out)
q = {r['parity']: r for r in out}
old = float(q['0']['ns_per_call_median']); new = float(q['1']['ns_per_call_median'])
print(f'owner_local_sector_parity_microprobe_speedup={old/new:.6f}x')
print(f'owner_local_sector_parity_microprobe_delta_pct={(new/old-1)*100:.4f}%')
print('owner_local_sector_W28_candidates_old=15')
print('owner_local_sector_W28_candidates_new=7')
print('owner_local_sector_W28_max_comparisons_old=4')
print('owner_local_sector_W28_max_comparisons_new=3')
print('owner_local_sector_W28_tree=0')
print(f'checksum={checksums["0"]}')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas full endpoint binary ---' >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/parity0.build.err" >&2 || true
  echo '--- ptxas positive-sector binary ---' >&2; grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/parity1.build.err" >&2 || true
fi
echo "gridfp-runtime-owner-local-sector-parity-microprobe OK repeats=$REPEATS result=$RESULT" >&2
