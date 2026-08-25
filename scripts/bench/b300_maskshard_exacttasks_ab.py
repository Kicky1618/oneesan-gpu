#!/usr/bin/env python3
"""A/B entrypoint isolating v0.19 exact compact BLOCKED orbit tasks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE / "b300_maskshard_ab.py"

spec = importlib.util.spec_from_file_location("b300_maskshard_ab", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load base A/B driver: {BASE}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.VARIANTS.update(
    {
        "v0.18": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_lazyblockinit_rowdepthexact_tightlaunch_rowdepthorbit_rowcaplaunch_batch_guarded.cu",
        "v0.19": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_lazyblockinit_rowdepthexact_tightlaunch_rowdepthorbit_rowcaplaunch_exacttasks_batch_guarded.cu",
    }
)

if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.18", "v0.19"]

raise SystemExit(base.main())
