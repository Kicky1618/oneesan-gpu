#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
W=$((N + 1))
ARCH="${ARCH:-native}"
SRC="${SRC:-}"
if [[ -z "$SRC" ]]; then
  if (( N >= 27 )); then
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_forced2window_opt_batch.cu"
  else
    SRC="src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN_batch.cu"
  fi
fi
SRC="$(repo_path "$SRC")"
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_batch_n${N}}")"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"

if [[ -z "$LOW_LUT_K" ]]; then
  if (( N >= 27 )); then LOW_LUT_K=14; else LOW_LUT_K=0; fi
fi
if [[ -z "$HIGH_LUT_K" ]]; then
  if (( N >= 27 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi
fi

TMPDIR="$ONEESAN_TMP_DIR" nvcc \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$SRC" -o "$OUT"

python3 - "$ONEESAN_ROOT" "$OUT" "$SRC" "$N" "$W" "$ARCH" "$LOW_LUT_K" "$HIGH_LUT_K" "$0" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
src = Path(sys.argv[3]).resolve()
n, width = map(int, sys.argv[4:6])
arch = sys.argv[6]
low, high = map(int, sys.argv[7:9])
build_script = Path(sys.argv[9]).resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

cp = subprocess.run(
    ["git", "-C", str(root), "rev-parse", "HEAD"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    check=False,
)
meta = {
    "schema": 1,
    "git_head": cp.stdout.strip() if cp.returncode == 0 else "",
    "binary": str(out),
    "binary_sha256": sha256(out),
    "source": rel(src),
    "source_sha256": sha256(src),
    "build_script": rel(build_script),
    "build_script_sha256": sha256(build_script),
    "n": n,
    "width": width,
    "arch": arch,
    "low_lut_k": low,
    "high_lut_k": high,
    "nvcc_flags": [
        "-O3", "-std=c++17", "-lineinfo", f"-arch={arch}",
        f"-DTARGET_W={width}", f"-DLOW_LUT_K={low}", f"-DHIGH_LUT_K={high}",
    ],
}
Path(str(out) + ".build.json").write_text(json.dumps(meta, indent=2) + "\n")
PY

echo "built $OUT"
echo "  provenance=${OUT}.build.json"
echo "  source=$SRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K"
