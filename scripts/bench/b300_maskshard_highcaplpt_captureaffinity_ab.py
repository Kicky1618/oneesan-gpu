#!/usr/bin/env python3
"""A/B v0.76 exact HIGH cap LPT vs v0.77 capture-affinity tie-break."""
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

base.VARIANTS.update({
    "v0.76": "src/cuda/b300/oneesan_b300_maskshard_v076_highcaplpt_exactclosure.cu",
    "v0.77": "src/cuda/b300/oneesan_b300_maskshard_v077_highcaplpt_captureaffinity.cu",
})
base.PHASE_KEYS = (
    "wall_s",
    "high_graph_captures",
    "high_graph_launches",
    "max_scratch_gib",
)
if "--variants" not in sys.argv[1:]:
    sys.argv[1:1] = ["--variants", "v0.76", "v0.77"]
raise SystemExit(base.main())
