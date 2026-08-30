#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "scripts" / "run" / "auto-plan.py"
spec = importlib.util.spec_from_file_location("oneesan_auto_plan", PLAN_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

for key in (
    "TARGET_MIB",
    "GRIDFP_PLAN_TARGET_MIB",
    "GRIDFP_VRAM_RESERVE_MIB",
    "MAX_WINDOW",
):
    os.environ.pop(key, None)


def gpu(i: int, name: str, total: int, free: int):
    return mod.GPU(i, name, total, free)


def full_p2p(n: int):
    return tuple(tuple(True for _ in range(n)) for _ in range(n))


def diagonal_p2p(n: int):
    return tuple(tuple(i == j for j in range(n)) for i in range(n))


def hardware(gpus, *, ram=1_000_000, p2p=None, managed=True):
    gpus = tuple(gpus)
    return mod.Hardware(
        gpus=gpus,
        system_ram_available_mib=ram,
        p2p=p2p if p2p is not None else full_p2p(len(gpus)),
        managed_memory=managed,
    )


# The production recurrence is part of the planner contract.
assert mod.state_counts(22) == (2_062_967_382, 728_997_192)

p = mod.plan(
    27,
    hardware([gpu(i, "NVIDIA B300", 288_000, 280_000) for i in range(8)]),
)
assert (p.runner, p.storage, p.ngpu) == ("optimized", "device-sharded", 8), p

# Regression: homogeneous 8-GPU hardware is not enough to select b300x8.sh.
p = mod.plan(
    20,
    hardware([gpu(i, "NVIDIA H100 80GB HBM3", 81_920, 80_000) for i in range(8)]),
)
assert (p.runner, p.storage) == ("adaptive", "device-vmm"), p

# Mixed GPUs use capacity-weighted VMM when every GPU can peer-map the state.
p = mod.plan(
    22,
    hardware(
        [
            gpu(0, "NVIDIA B300", 288_000, 270_000),
            gpu(1, "NVIDIA H100 80GB HBM3", 81_920, 72_000),
        ]
    ),
)
assert (p.runner, p.storage) == ("adaptive", "device-vmm"), p
assert p.vmm_weights_mib[0] > p.vmm_weights_mib[1] > 0, p

# Consumer GPUs commonly lack cross-card P2P. Keep GPU memory for scratch and
# put the authoritative state in host-preferred Managed Memory.
p = mod.plan(
    22,
    hardware(
        [gpu(i, "NVIDIA GeForce RTX 4090", 24_576, 23_000) for i in range(4)],
        ram=256_000,
        p2p=diagonal_p2p(4),
    ),
)
assert (p.runner, p.storage, p.full_p2p) == ("adaptive", "managed-host", False), p

# More than eight B300s must not enter the 8-way hand-specialized path.
p = mod.plan(
    24,
    hardware([gpu(i, "NVIDIA B300", 288_000, 270_000) for i in range(12)]),
)
assert (p.runner, p.storage, p.ngpu) == ("adaptive", "device-vmm", 12), p

# GB300 is deliberately not treated as the exact B300 model token.
p = mod.plan(
    20,
    hardware([gpu(i, "NVIDIA GB300", 288_000, 270_000) for i in range(8)]),
)
assert p.runner == "adaptive", p

try:
    mod.plan(
        27,
        hardware(
            [gpu(i, "NVIDIA GeForce RTX 4090", 24_576, 23_000) for i in range(4)],
            ram=64_000,
            p2p=diagonal_p2p(4),
        ),
    )
except ValueError as exc:
    assert "System RAM" in str(exc), exc
else:
    raise AssertionError("n=27 must reject an undersized host-memory fallback")

print("auto-plan-selftest OK")
