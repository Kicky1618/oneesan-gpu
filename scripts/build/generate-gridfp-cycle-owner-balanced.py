#!/usr/bin/env python3
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: generate-gridfp-cycle-owner-balanced.py BASE OUT", file=sys.stderr)
        return 2
    base = Path(sys.argv[1])
    out = Path(sys.argv[2])
    s = base.read_text()

    marker = "namespace {\n\nstatic constexpr int CYCLE_OWNER_MAX_SEGMENTS"
    if s.count(marker) != 1:
        raise SystemExit("balanced cycle-owner generator: namespace marker mismatch")

    helpers = r'''namespace {

// The cycle-owner list contains exactly one canonical leader per non-local
// physical support cycle.  Therefore batching no longer needs a rotation-
// invariant self-correlation hash: hash the leader itself.  Bits 12.. are the
// W=28 exact-balanced choice; masking supports both B=16 and B=32 probes.
int cycle_owner_balanced_host_batch_id(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int batches
) {
    std::uint32_t key = support;
    if (blocked) {
        std::uint32_t compact = 0;
        int cp = 0;
        for (int bit = 0; bit < W; ++bit) {
            if (bit == q - 1 || bit == q) continue;
            if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
            ++cp;
        }
        const std::uint32_t half_mask = (std::uint32_t(1) << Kwin) - 1u;
        const std::uint32_t a = compact & half_mask;
        const std::uint32_t b = (compact >> Kwin) & half_mask;
        const std::uint32_t hi = a < b ? b : a;
        const std::uint32_t lo = a < b ? a : b;
        key = hi | (lo << Kwin);
    }
    return int((hostlist_mix32(key) >> 12) &
               static_cast<std::uint32_t>(batches - 1));
}

__device__ __forceinline__ int cycle_owner_balanced_device_batch_id(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int batches
) {
    std::uint32_t key = support;
    if (blocked) {
        const std::uint32_t compact = persistent_blocked_compact(support, W, q);
        const std::uint32_t half_mask = (std::uint32_t(1) << Kwin) - 1u;
        const std::uint32_t a = compact & half_mask;
        const std::uint32_t b = (compact >> Kwin) & half_mask;
        const std::uint32_t hi = a < b ? b : a;
        const std::uint32_t lo = a < b ? a : b;
        key = hi | (lo << Kwin);
    }
    return int((persistent_mix32(key) >> 12) &
               static_cast<std::uint32_t>(batches - 1));
}

static constexpr int CYCLE_OWNER_MAX_SEGMENTS'''
    s = s.replace(marker, helpers)

    host_calls = s.count("hostlist_batch_id(")
    device_calls = s.count("persistent_batch_id(")
    if host_calls != 1 or device_calls != 2:
        raise SystemExit(
            f"balanced cycle-owner generator: batch call count host={host_calls} device={device_calls}")
    s = s.replace("hostlist_batch_id(", "cycle_owner_balanced_host_batch_id(")
    s = s.replace("persistent_batch_id(", "cycle_owner_balanced_device_batch_id(")

    old_tag = 'std::cout << "gridfp-p2p-cycle-owner-descriptorless"'
    new_tag = 'std::cout << "gridfp-p2p-cycle-owner-balanced-descriptorless"'
    if s.count(old_tag) != 1:
        raise SystemExit("balanced cycle-owner generator: output tag mismatch")
    s = s.replace(old_tag, new_tag)

    flag = '              << " descriptor_bytes=0"\n'
    if s.count(flag) != 1:
        raise SystemExit("balanced cycle-owner generator: flag marker mismatch")
    s = s.replace(
        flag,
        '              << " leader_batch_hash=1"\n'
        '              << " batch_hash_shift=12"\n' + flag)

    old_ok = 'std::cout << "ALL_OK gridfp_p2p_cycle_owner_descriptorless=1\\n";'
    new_ok = 'std::cout << "ALL_OK gridfp_p2p_cycle_owner_balanced_descriptorless=1\\n";'
    if s.count(old_ok) != 1:
        raise SystemExit("balanced cycle-owner generator: ALL_OK marker mismatch")
    s = s.replace(old_ok, new_ok)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
