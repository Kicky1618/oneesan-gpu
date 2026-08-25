#!/usr/bin/env python3
"""A/B entrypoint isolating the v0.25 HIGH closure row-pack threshold change."""

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
        "v0.24": "src/cuda/b300/oneesan_b300_maskshard_v024_lowclosure_rowdepth.cu",
        "v0.25": "src/cuda/b300/oneesan_b300_maskshard_v025_highrowpack29.cu",
    }
)

if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.24", "v0.25"]

raise SystemExit(base.main())
