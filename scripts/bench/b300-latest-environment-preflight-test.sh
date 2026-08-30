#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
PRE="$ONEESAN_ROOT/scripts/run/b300x8-grand-latest-preflight.sh"
[[ -s "$PRE" ]] || exit 2
bash -n "$PRE"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-latest-env-preflight.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"

cat >"$tmp/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
args="$*"
if [[ "$args" == *'--query-gpu=driver_version'* ]]; then
  for i in {0..7}; do echo '590.00'; done
elif [[ "$args" == *'--query-gpu=index,name,memory.total,memory.free,compute_mode'* ]]; then
  free="${FAKE_GPU_FREE:-200000}"
  for i in {0..7}; do printf '%s, NVIDIA B300, 288000, %s, Default\n' "$i" "$free"; done
elif [[ "$args" == *'--query-gpu=index'* ]]; then
  for i in {0..7}; do echo "$i"; done
elif [[ "$1" == topo && "${2:-}" == -m ]]; then
  printf 'GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7\n'
  for i in {0..7}; do printf 'GPU%s X NV18 NV18 NV18 NV18 NV18 NV18 NV18\n' "$i"; done
else
  echo "unexpected fake nvidia-smi args: $args" >&2; exit 9
fi
SH
chmod +x "$tmp/bin/nvidia-smi"

cat >"$tmp/bin/nvcc" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --version ]]; then
  echo 'Cuda compilation tools, release 13.0, V13.0.0'; exit 0
fi
out=''
while (($#)); do
  if [[ "$1" == -o ]]; then shift; out="${1:?}"; fi
  shift || true
done
[[ -n "$out" ]] || { echo 'fake nvcc missing -o' >&2; exit 9; }
cat >"$out" <<'BIN'
#!/usr/bin/env bash
set -euo pipefail
need="${1:-8}"
for ((i=0;i<need;++i)); do for ((j=0;j<need;++j)); do ((i==j)) && continue; printf 'p2p=%s->%s can=%s\n' "$i" "$j" "$([[ "${FAKE_P2P_FAIL:-0}" == 1 ]] && echo 0 || echo 1)"; done; done
[[ "${FAKE_P2P_FAIL:-0}" == 1 ]] && exit 5
exit 0
BIN
chmod +x "$out"
SH
chmod +x "$tmp/bin/nvcc"

run_env=(PATH="$tmp/bin:$PATH" TARGET_MIB=1024 GRIDFP_VRAM_RESERVE_MIB=128 MIN_HOST_AVAILABLE_MIB=1 STRICT_B300_MODEL=1 ALLOW_DIRTY_WORKTREE=0)

env "${run_env[@]}" bash "$PRE" 27 >"$tmp/good.out" 2>"$tmp/good.err"
grep -Fq 'b300_latest_preflight=OK stage=T ' "$tmp/good.out" || { cat "$tmp/good.out" >&2; exit 3; }
grep -Fq 'full_p2p=1' "$tmp/good.out" || exit 3
grep -Fq 'strict_b300_model=1' "$tmp/good.out" || exit 3

set +e
env "${run_env[@]}" FAKE_GPU_FREE=100 bash "$PRE" 27 >"$tmp/vram.out" 2>"$tmp/vram.err"; rc=$?
set -e
((rc==4)) || { echo "expected VRAM rc=4 got $rc" >&2; cat "$tmp/vram.err" >&2; exit 3; }
grep -Fq 'free VRAM 100MiB < required 1152MiB' "$tmp/vram.err" || exit 3

set +e
env "${run_env[@]}" FAKE_P2P_FAIL=1 REQUIRE_FULL_P2P=1 bash "$PRE" 27 >"$tmp/p2p.out" 2>"$tmp/p2p.err"; rc=$?
set -e
((rc==5)) || { echo "expected P2P rc=5 got $rc" >&2; cat "$tmp/p2p.err" >&2; exit 3; }
grep -Fq 'full GPU peer access check failed rc=5' "$tmp/p2p.err" || exit 3

env "${run_env[@]}" FAKE_P2P_FAIL=1 REQUIRE_FULL_P2P=0 bash "$PRE" 27 >"$tmp/p2p-warn.out" 2>"$tmp/p2p-warn.err"
grep -Fq 'full_p2p=0' "$tmp/p2p-warn.out" || exit 3
grep -Fq 'warning: full GPU peer access unavailable rc=5' "$tmp/p2p-warn.err" || exit 3

echo 'b300-latest-environment-preflight-test OK latest_stage=T good=1 vram_fail_closed=1 p2p_fail_closed=1 p2p_override=1 gpu_work=0'
