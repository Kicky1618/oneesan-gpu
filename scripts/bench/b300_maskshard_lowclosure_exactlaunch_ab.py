#!/usr/bin/env python3
"""A/B entrypoint isolating v0.27 exact LOW closure host launch sizing."""

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
        "v0.26": "src/cuda/b300/oneesan_b300_maskshard_v026_lowclosure_exacttasks.cu",
        "v0.27": "src/cuda/b300/oneesan_b300_maskshard_v027_lowclosure_exactlaunch.cu",
    }
)

if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.26", "v0.27"]

raise SystemExit(base.main())
