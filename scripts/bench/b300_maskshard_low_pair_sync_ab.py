#!/usr/bin/env python3
"""A/B entrypoint isolating v0.34 LOW orbit/closure pair synchronization."""

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
        "v0.33": "src/cuda/b300/oneesan_b300_maskshard_v033_loworbit_warpdecode_fullcap.cu",
        "v0.34": "src/cuda/b300/oneesan_b300_maskshard_v034_low_pair_sync.cu",
    }
)

if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.33", "v0.34"]

raise SystemExit(base.main())
