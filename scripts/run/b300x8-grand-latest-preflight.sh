#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
[[ "$N" == 27 ]] || { echo 'latest grand preflight currently targets n=27' >&2; exit 2; }
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-65536}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
MIN_HOST_AVAILABLE_MIB="${MIN_HOST_AVAILABLE_MIB:-32768}"
STRICT_B300_MODEL="${STRICT_B300_MODEL:-0}"
REQUIRE_FULL_P2P="${REQUIRE_FULL_P2P:-1}"
ALLOW_DIRTY_WORKTREE="${ALLOW_DIRTY_WORKTREE:-0}"
for x in STRICT_B300_MODEL REQUIRE_FULL_P2P ALLOW_DIRTY_WORKTREE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$NGPU" =~ ^[1-9][0-9]*$ ]] && ((NGPU==8)) || { echo 'latest grand chain requires NGPU=8' >&2; exit 2; }
for x in TARGET_MIB GRIDFP_VRAM_RESERVE_MIB MIN_HOST_AVAILABLE_MIB; do
  v="${!x}"; [[ "$v" =~ ^[0-9]+$ ]] || { echo "$x must be a nonnegative integer" >&2; exit 2; }
done
((TARGET_MIB>0)) || { echo 'TARGET_MIB must be positive' >&2; exit 2; }

required_cmds=(git python3 nvcc nvidia-smi sha256sum awk sed grep mktemp)
for c in "${required_cmds[@]}"; do command -v "$c" >/dev/null || { echo "missing required command=$c" >&2; exit 2; }; done
required_files=(
  scripts/run/b300x8-grand-firstpass-latest.sh
  scripts/run/b300x8-grand-promote-exact-latest.sh
  scripts/run/b300x8-grand-firstpass-stages.sh
  scripts/run/b300x8-grand-promote-exact-stages.sh
  scripts/run/b300x8-joint-nextself-hybrid8-select-stages.sh
  scripts/run/b300x8-nextgen-hybrid8-stages-ilp2-cg-l2-staged-fullprime-race.sh
  scripts/bench/b300-stages-preflight.sh
  scripts/bench/b300-grand-stages-contract-preflight.sh
  scripts/bench/b300-grand-stages-firstpass-preflight.sh
  scripts/bench/b300-grand-stages-exact-promotion-preflight.sh
  scripts/bench/b300-grand-stages-exact-controlpath-preflight.sh
)
for rel in "${required_files[@]}"; do [[ -s "$ONEESAN_ROOT/$rel" ]] || { echo "missing latest-chain artifact=$rel" >&2; exit 3; }; done

HEAD_SHA="$(git -C "$ONEESAN_ROOT" rev-parse HEAD)"
DIRTY=0
[[ -z "$(git -C "$ONEESAN_ROOT" status --porcelain=v1 --untracked-files=normal)" ]] || DIRTY=1
if ((DIRTY)) && [[ "$ALLOW_DIRTY_WORKTREE" != 1 ]]; then
  echo 'worktree is dirty; exact promotion will later reject it by default' >&2
  echo 'commit/stash changes or set ALLOW_DIRTY_WORKTREE=1 only after auditing them' >&2
  exit 3
fi

require_nvcc_version_at_least nvcc 12 8 'Blackwell/B300 build path'
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | tr -d '[:space:]')"
VISIBLE="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
((VISIBLE>=NGPU)) || { echo "need $NGPU visible GPUs, found $VISIBLE" >&2; exit 4; }

GPU_CSV="$(mktemp "${TMPDIR:-/tmp}/oneesan-b300-preflight-gpu.XXXXXX")"
P2P_SRC="$(mktemp "${TMPDIR:-/tmp}/oneesan-b300-preflight-p2p.XXXXXX.cu")"
P2P_BIN="$(mktemp "${TMPDIR:-/tmp}/oneesan-b300-preflight-p2p.XXXXXX.bin")"
trap 'rm -f "$GPU_CSV" "$P2P_SRC" "$P2P_BIN"' EXIT
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_mode --format=csv,noheader,nounits >"$GPU_CSV"
MIN_FREE_REQ=$((TARGET_MIB + GRIDFP_VRAM_RESERVE_MIB))
BAD_MODEL=0; BAD_MEM=0; BAD_MODE=0; COUNT=0
while IFS=',' read -r idx name total free mode; do
  idx="${idx//[[:space:]]/}"; name="$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' <<<"$name")"
  total="${total//[[:space:]]/}"; free="${free//[[:space:]]/}"; mode="$(sed 's/^[[:space:]]*//;s/[[:space:]]*$//' <<<"$mode")"
  ((++COUNT))
  ((COUNT<=NGPU)) || continue
  [[ "$name" == *B300* ]] || BAD_MODEL=1
  [[ "$free" =~ ^[0-9]+$ ]] && ((free>=MIN_FREE_REQ)) || { echo "GPU $idx free VRAM ${free}MiB < required ${MIN_FREE_REQ}MiB" >&2; BAD_MEM=1; }
  [[ "$mode" == Default ]] || { echo "GPU $idx compute_mode=$mode (expected Default)" >&2; BAD_MODE=1; }
  printf 'gpu=%s name=%q total_mib=%s free_mib=%s compute_mode=%q\n' "$idx" "$name" "$total" "$free" "$mode"
done <"$GPU_CSV"
((COUNT>=NGPU)) || { echo 'GPU query returned fewer devices than expected' >&2; exit 4; }
((BAD_MEM==0)) || exit 4
((BAD_MODE==0)) || exit 4
if ((BAD_MODEL)); then
  if [[ "$STRICT_B300_MODEL" == 1 ]]; then echo 'one or more GPUs are not named B300' >&2; exit 4; fi
  echo 'warning: one or more GPU names do not contain B300; continuing because STRICT_B300_MODEL=0' >&2
fi

HOST_AVAIL_KIB="$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)"
[[ "$HOST_AVAIL_KIB" =~ ^[0-9]+$ ]] || { echo 'cannot read MemAvailable from /proc/meminfo' >&2; exit 4; }
HOST_AVAIL_MIB=$((HOST_AVAIL_KIB/1024))
((HOST_AVAIL_MIB>=MIN_HOST_AVAILABLE_MIB)) || { echo "host MemAvailable ${HOST_AVAIL_MIB}MiB < ${MIN_HOST_AVAILABLE_MIB}MiB" >&2; exit 4; }

cat >"$P2P_SRC" <<'CU'
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){
  int need=8; if(argc>1) need=std::atoi(argv[1]);
  int n=0; cudaError_t e=cudaGetDeviceCount(&n); if(e!=cudaSuccess){std::fprintf(stderr,"cudaGetDeviceCount: %s\n",cudaGetErrorString(e));return 2;}
  if(n<need){std::fprintf(stderr,"cuda device count %d < %d\n",n,need);return 3;}
  int bad=0;
  for(int i=0;i<need;++i) for(int j=0;j<need;++j) if(i!=j){int can=0; e=cudaDeviceCanAccessPeer(&can,i,j); if(e!=cudaSuccess){std::fprintf(stderr,"peer %d->%d error: %s\n",i,j,cudaGetErrorString(e));return 4;} std::printf("p2p=%d->%d can=%d\n",i,j,can); if(!can) bad=1;}
  return bad?5:0;
}
CU
nvcc -O2 -std=c++17 "$P2P_SRC" -o "$P2P_BIN"
set +e
"$P2P_BIN" "$NGPU"; P2P_RC=$?
set -e
if ((P2P_RC!=0)); then
  if [[ "$REQUIRE_FULL_P2P" == 1 ]]; then echo "full GPU peer access check failed rc=$P2P_RC" >&2; exit 5; fi
  echo "warning: full GPU peer access unavailable rc=$P2P_RC; REQUIRE_FULL_P2P=0" >&2
fi

TOPO="$(nvidia-smi topo -m 2>&1 || true)"
printf '%s\n' "$TOPO"
printf 'b300_latest_preflight=OK stage=S head=%s dirty=%s ngpu=%s target_mib=%s reserve_mib=%s min_gpu_free_mib=%s host_available_mib=%s driver=%s full_p2p=%s strict_b300_model=%s\n' \
  "$HEAD_SHA" "$DIRTY" "$NGPU" "$TARGET_MIB" "$GRIDFP_VRAM_RESERVE_MIB" "$MIN_FREE_REQ" "$HOST_AVAIL_MIB" "$DRIVER" "$((P2P_RC==0))" "$STRICT_B300_MODEL"
