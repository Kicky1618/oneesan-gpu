#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MIB = 1 << 20
MAX_N = 27
MAX_GPU = 16


@dataclass(frozen=True)
class GPU:
    index: int
    name: str
    memory_total_mib: int
    memory_free_mib: int


@dataclass(frozen=True)
class Hardware:
    gpus: tuple[GPU, ...]
    system_ram_available_mib: int
    p2p: tuple[tuple[bool, ...], ...]
    managed_memory: bool = True


@dataclass(frozen=True)
class Plan:
    n: int
    runner: str
    storage: str
    ngpu: int
    state_mib: int
    target_mib: int
    planner_target_mib: int
    reserve_mib: int
    max_window: int
    full_p2p: bool
    system_ram_available_mib: int
    gpu_names: tuple[str, ...]
    vmm_weights_mib: tuple[int, ...]
    reason: str


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL)


def _parse_gpu_csv(text: str) -> tuple[GPU, ...]:
    out: list[GPU] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = [x.strip() for x in raw.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi row: {raw!r}")
        out.append(GPU(int(fields[0]), fields[1], int(fields[2]), int(fields[3])))
    return tuple(out)


def _parse_p2p(text: str, ngpu: int) -> tuple[tuple[bool, ...], ...]:
    rows: list[list[bool]] = []
    for raw in text.splitlines():
        fields = raw.split()
        if not fields or not re.fullmatch(r"GPU\d+", fields[0]):
            continue
        cells = fields[1 : 1 + ngpu]
        if len(cells) != ngpu:
            continue
        rows.append([i == len(rows) or cell.upper() == "OK" for i, cell in enumerate(cells)])
    if len(rows) != ngpu:
        # Unknown topology is treated conservatively. A single GPU needs no peer path.
        return tuple(
            tuple(i == j for j in range(ngpu))
            for i in range(ngpu)
        )
    return tuple(tuple(row) for row in rows)


def _mem_available_mib() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return 0


def discover_hardware() -> Hardware:
    try:
        gpu_csv = _run(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi is required for automatic hardware discovery") from exc

    gpus = _parse_gpu_csv(gpu_csv)
    if not gpus:
        raise RuntimeError("no CUDA GPUs found")
    if len(gpus) > MAX_GPU:
        raise RuntimeError(f"adaptive backend supports at most {MAX_GPU} GPUs; found {len(gpus)}")

    try:
        p2p_text = _run("nvidia-smi", "topo", "-p2p", "r")
    except (FileNotFoundError, subprocess.CalledProcessError):
        p2p_text = ""

    managed = os.environ.get("ONEESAN_DISABLE_MANAGED", "0") != "1"
    return Hardware(
        gpus=gpus,
        system_ram_available_mib=_mem_available_mib(),
        p2p=_parse_p2p(p2p_text, len(gpus)),
        managed_memory=managed,
    )


def hardware_from_json(data: dict[str, Any]) -> Hardware:
    gpus = tuple(
        GPU(
            int(g["index"]),
            str(g["name"]),
            int(g["memory_total_mib"]),
            int(g["memory_free_mib"]),
        )
        for g in data["gpus"]
    )
    ngpu = len(gpus)
    raw_p2p = data.get("p2p")
    if raw_p2p is None:
        p2p = tuple(tuple(i == j for j in range(ngpu)) for i in range(ngpu))
    else:
        if len(raw_p2p) != ngpu or any(len(row) != ngpu for row in raw_p2p):
            raise ValueError("p2p must be an ngpu x ngpu matrix")
        p2p = tuple(tuple(bool(x) for x in row) for row in raw_p2p)
    return Hardware(
        gpus=gpus,
        system_ram_available_mib=int(data.get("system_ram_available_mib", 0)),
        p2p=p2p,
        managed_memory=bool(data.get("managed_memory", True)),
    )


def _load_hardware_arg(value: str) -> Hardware:
    text = value if value.lstrip().startswith("{") else Path(value).read_text()
    return hardware_from_json(json.loads(text))


def state_counts(n: int) -> tuple[int, int]:
    if not 2 <= n <= MAX_N:
        raise ValueError(f"n must be in 2..{MAX_N}")
    maxw = MAX_N + 1
    dp = [[0] * (maxw + 2) for _ in range(maxw + 1)]
    dp[0][0] = 1
    for w in range(1, maxw + 1):
        for h in range(maxw + 1):
            value = dp[w - 1][h]
            if h:
                value += dp[w - 1][h - 1]
            value += dp[w - 1][h + 1]
            dp[w][h] = value
    width = n + 1
    return dp[width][1], dp[width - 1][1]


def _state_mib(n: int) -> int:
    main, block = state_counts(n)
    return math.ceil((main + block) * 4 / MIB)


def _is_b300(name: str) -> bool:
    return re.search(r"(^|[^A-Z0-9])B300([^A-Z0-9]|$)", name.upper()) is not None


def _full_p2p(hw: Hardware) -> bool:
    n = len(hw.gpus)
    return all(i == j or hw.p2p[i][j] for i in range(n) for j in range(n))


def _reserve_mib(hw: Hardware) -> int:
    # Keep enough headroom for contexts, LUTs and transient allocations without
    # throwing away a large fraction of smaller cards.
    per_gpu = [min(8192, max(512, g.memory_total_mib // 32)) for g in hw.gpus]
    return min(per_gpu)


def _window_for_target(target_mib: int) -> int:
    if target_mib >= 16384:
        return 14
    if target_mib >= 8192:
        return 12
    if target_mib >= 4096:
        return 10
    if target_mib >= 2048:
        return 8
    return 6


def _env_int(name: str, default: int, low: int = 0) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < low:
        raise ValueError(f"{name} must be >= {low}")
    return parsed


def plan(n: int, hw: Hardware) -> Plan:
    if not hw.gpus:
        raise ValueError("at least one GPU is required")
    if len(hw.gpus) > MAX_GPU:
        raise ValueError(f"at most {MAX_GPU} GPUs are supported")

    ngpu = len(hw.gpus)
    state_mib = _state_mib(n)
    full_p2p = _full_p2p(hw)
    reserve = _env_int("GRIDFP_VRAM_RESERVE_MIB", _reserve_mib(hw), 0)
    usable = tuple(max(0, g.memory_free_mib - reserve) for g in hw.gpus)
    total_usable = sum(usable)

    # The hand-tuned path contains B300x8-specific addressing and scheduling.
    # Never select it merely because another 8-GPU machine is homogeneous.
    b300x8 = ngpu == 8 and all(_is_b300(g.name) for g in hw.gpus)

    if b300x8 and full_p2p:
        storage = "device-sharded"
        runner = "optimized"
        reason = "exact B300x8 with full peer access"
        auth_each = math.ceil(state_mib / ngpu)
        scratch_room = min(max(0, u - auth_each) for u in usable)
        weights = usable
    elif full_p2p and total_usable >= state_mib:
        storage = "device-vmm"
        runner = "adaptive"
        reason = "full peer access; authoritative state fits aggregate free VRAM"
        # The VMM allocator uses the same free-VRAM weights. Estimate the
        # post-state scratch room with the proportional allocation it will use.
        fraction_left = (total_usable - state_mib) / max(1, total_usable)
        scratch_room = min(int(u * fraction_left) for u in usable)
        weights = usable
    else:
        if not hw.managed_memory:
            why = "P2P is incomplete" if not full_p2p else "aggregate VRAM is insufficient"
            raise ValueError(f"{why}, and Managed Memory fallback is disabled")
        host_margin = max(2048, state_mib // 20)
        if hw.system_ram_available_mib < state_mib + host_margin:
            raise ValueError(
                "authoritative state does not fit safely in available System RAM: "
                f"need about {state_mib + host_margin} MiB, have {hw.system_ram_available_mib} MiB"
            )
        storage = "managed-host"
        runner = "adaptive"
        reason = (
            "P2P incomplete; use host-preferred Managed Memory"
            if not full_p2p
            else "authoritative state exceeds aggregate VRAM; spill to System RAM"
        )
        scratch_room = min(usable)
        weights = tuple(0 for _ in hw.gpus)

    default_target = max(256, min(65536, scratch_room))
    target = _env_int("TARGET_MIB", default_target, 1)
    planner_target = _env_int("GRIDFP_PLAN_TARGET_MIB", min(16384, target), 1)
    planner_target = min(planner_target, target)
    max_window = _env_int("MAX_WINDOW", _window_for_target(target), 1)

    return Plan(
        n=n,
        runner=runner,
        storage=storage,
        ngpu=ngpu,
        state_mib=state_mib,
        target_mib=target,
        planner_target_mib=planner_target,
        reserve_mib=reserve,
        max_window=max_window,
        full_p2p=full_p2p,
        system_ram_available_mib=hw.system_ram_available_mib,
        gpu_names=tuple(g.name for g in hw.gpus),
        vmm_weights_mib=weights,
        reason=reason,
    )


def _shell(plan: Plan) -> str:
    values = {
        "ONEESAN_RUNNER": plan.runner,
        "ONEESAN_STORAGE": plan.storage,
        "NGPU": str(plan.ngpu),
        "TARGET_MIB": str(plan.target_mib),
        "GRIDFP_PLAN_TARGET_MIB": str(plan.planner_target_mib),
        "GRIDFP_VRAM_RESERVE_MIB": str(plan.reserve_mib),
        "MAX_WINDOW": str(plan.max_window),
        "ONEESAN_VMM_WEIGHTS_MIB": ",".join(map(str, plan.vmm_weights_mib)),
    }
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in values.items())


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan oneesan-gpu storage and runner from live hardware")
    parser.add_argument("n", type=int)
    parser.add_argument("--hardware-json", help="JSON literal or path; skips live discovery")
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()

    try:
        hw = _load_hardware_arg(args.hardware_json) if args.hardware_json else discover_hardware()
        result = plan(args.n, hw)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"auto-plan: {exc}\n")

    if args.format == "shell":
        print(_shell(result))
    else:
        print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
