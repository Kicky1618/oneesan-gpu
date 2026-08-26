#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-21}"
PRIME="${PRIME:-4294967291}"
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
PM_ACCUM="${PM_ACCUM:-0}"
REBUILD="${REBUILD:-0}"
BUCKET_TRANSPOSE_CHUNK_MIB="${BUCKET_TRANSPOSE_CHUNK_MIB:-1024}"
BUCKET_RESERVE_MIB="${BUCKET_RESERVE_MIB:-8192}"

if (( NGPU != 8 )); then echo "benchmark currently requires NGPU=8" >&2; exit 2; fi
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2 ;; esac
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; fi

suffix="_${TRANSPOSE_MODE}"
if [[ "$PM_ACCUM" == 1 ]]; then suffix="${suffix}_pm"; fi
inline_bin="$ONEESAN_BUILD_DIR/bench_hybrid18_inline8_graph_batch${suffix}_n${N}"
zero_bin="$ONEESAN_BUILD_DIR/bench_zero_graph_batch${suffix}_n${N}"

build_one(){
  local kind="$1" out="$2" script="$3"
  if [[ "$REBUILD" == 1 || ! -x "$out" ]]; then
    echo "building $kind: $out" >&2
    N="$N" OUT="$out" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/$script"
  fi
}
build_one inline8 "$inline_bin" scripts/build/b300-bucket-snake-hybrid18-inline8-graph-batch.sh
build_one zero "$zero_bin" scripts/build/b300-bucket-snake-zero-graph-batch.sh

export BUCKET_TRANSPOSE_CHUNK_MIB BUCKET_RESERVE_MIB
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

run_one(){
  local name="$1" bin="$2" out="$TMP/$1.out" err="$TMP/$1.err"
  echo "=== $name n=$N p=$PRIME transpose=$TRANSPOSE_MODE pm=$PM_ACCUM ===" >&2
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$PRIME" > >(tee "$out") 2> >(tee "$err" >&2)
  local line residue wall meta fatt ratt
  line="$(grep -E '^residue=[0-9]+ modulus=[0-9]+ wall_s=' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$name produced no residue line" >&2; exit 3; }
  residue="$(sed -nE 's/^residue=([0-9]+).*/\1/p' <<<"$line")"
  wall="$(sed -nE 's/.*wall_s=([^ ]+).*/\1/p' <<<"$line")"
  meta="$(sed -nE 's/.*metadata_mib_per_gpu=([^ ]+).*/\1/p' "$err" | tail -n1)"
  fatt="$(sed -nE 's/.*forward_attach_mib=([^ ]+).*/\1/p' "$err" | tail -n1)"
  ratt="$(sed -nE 's/.*reverse_attach_mib=([^ ]+).*/\1/p' "$err" | tail -n1)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$residue" "$wall" "${meta:-NA}" "${fatt:-NA}" "${ratt:-NA}" > "$TMP/$1.result"
}

run_one inline8 "$inline_bin"
run_one zero "$zero_bin"
IFS=$'\t' read -r ir iw im ifa ira < "$TMP/inline8.result"
IFS=$'\t' read -r zr zw zm zfa zra < "$TMP/zero.result"
if [[ "$ir" != "$zr" ]]; then
  echo "residue mismatch: inline8=$ir zero=$zr mod $PRIME" >&2
  exit 4
fi
ratio="$(awk -v a="$iw" -v b="$zw" 'BEGIN{if(b==0)print "inf";else printf "%.6f",a/b}')"
mem_delta="$(awk -v a="$im" -v b="$zm" 'BEGIN{if(a=="NA"||b=="NA")print "NA";else printf "%.3f",a-b}')"
printf 'backend\twall_s\tmetadata_mib_per_gpu\tforward_attach_mib\treverse_attach_mib\tresidue\n'
printf 'inline8\t%s\t%s\t%s\t%s\t%s\n' "$iw" "$im" "$ifa" "$ira" "$ir"
printf 'zero\t%s\t%s\t%s\t%s\t%s\n' "$zw" "$zm" "$zfa" "$zra" "$zr"
printf 'zero_speedup_vs_inline8=%sx metadata_saved_mib_per_gpu=%s\n' "$ratio" "$mem_delta"
