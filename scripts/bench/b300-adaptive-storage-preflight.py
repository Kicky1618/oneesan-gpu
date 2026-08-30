#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
BUILD = ROOT / "scripts/build"

STAGES = [
    "gen-b300-adaptive-storage.py",
    "gen-b300-main-mate-cache.py",
    "gen-b300-main-pull.py",
    "gen-b300-block-pull.py",
    "gen-b300-block-mate-cache.py",
    "normalize-b300-rank-delta-input.py",
    "gen-b300-rank-delta-cache.py",
    "gen-b300-rank-delta-free-step.py",
    "gen-b300-rank-delta-report.py",
    "gen-b300-rank-state-packed.py",
    "gen-b300-rank-state-ilp2.py",
    "gen-b300-block-rank-state-ilp2.py",
    "gen-b300-concurrent-group-io.py",
    "lower-b300-row-limit.py",
    "gen-b300-runtime-threads.py",
    "gen-b300-plan-target.py",
]


def run_stage(script: str, src: Path, dst: Path, env: dict[str, str]) -> None:
    subprocess.run(
        [sys.executable, str(BUILD / script), str(src), str(dst)],
        check=True,
        env=env,
    )


def main() -> int:
    env = os.environ.copy()
    env.update(
        {
            "N": "22",
            "B300_ROW_LIMIT": "23",
            "GRIDFP_THREADS": "256",
            "GRIDFP_PLAN_TARGET_MIB": "8192",
            "GRIDFP_PLAN_TARGET_DIVISOR": "1",
        }
    )

    with tempfile.TemporaryDirectory(prefix="oneesan-adaptive-preflight-") as td:
        tmp = Path(td)
        current = SOURCE
        for i, stage in enumerate(STAGES):
            dst = tmp / f"{i:02d}_{stage.removesuffix('.py')}.cu"
            run_stage(stage, current, dst, env)
            current = dst

        text = current.read_text()
        required = (
            "MAXGPU=16",
            "cudaMemAddressReserve",
            "cudaMemCreate",
            "cudaMemMap",
            "cudaMemSetAccess",
            "adaptive storage=device-vmm",
            "adaptive storage=managed-host",
            "ONEESAN_STORAGE",
        )
        for marker in required:
            if marker not in text:
                raise AssertionError(f"missing final adaptive marker: {marker}")

        forbidden = (
            "MAXGPU=8;",
            'HBM mode requires full P2P:',
            'cudaMalloc(&mp[d]',
            'cudaMalloc(&bp[d]',
            'need 1..8 GPUs',
        )
        for marker in forbidden:
            if marker in text:
                raise AssertionError(f"stale non-adaptive marker remains: {marker}")

    print("b300-adaptive-storage-preflight OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
