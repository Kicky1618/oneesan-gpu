#!/usr/bin/env python3

WARP = 32
GROUP = 64
BLOCK = 32
cases = 0
max_pair_loads = 0
one_pair_full = two_pair_full = 0

for start_mod in range(GROUP):
    first_b32 = start_mod // BLOCK
    first_off = start_mod & (BLOCK - 1)
    split = BLOCK - first_off
    second_b32 = first_b32 + 1
    pair0 = first_b32 // 2
    pair1 = second_b32 // 2
    full_pairs = 1 + int(split < WARP and pair1 != pair0)
    if full_pairs == 1: one_pair_full += 1
    else: two_pair_full += 1

    for active_width in range(1, WARP + 1):
        active = set(range(active_width))
        loaded = {0}
        if split < WARP and pair1 != pair0 and split in active:
            loaded.add(split)
        for lane in active:
            after = split < WARP and lane >= split
            needs_second_pair = after and pair1 != pair0
            source = split if needs_second_pair else 0
            assert source in active, (start_mod, active_width, lane, split, source)
            lane_compact = start_mod + lane
            lane_b32 = lane_compact // BLOCK
            lane_pair = lane_b32 // 2
            source_pair = pair1 if needs_second_pair else pair0
            assert lane_pair == source_pair, (
                start_mod, active_width, lane, lane_pair, source_pair, split)
        max_pair_loads = max(max_pair_loads, len(loaded))
        assert len(loaded) <= 2
        cases += 1

assert (one_pair_full, two_pair_full) == (33, 31)
assert max_pair_loads == 2

print(
    "rankchunk32-basepair64-warp-proof OK"
    f" cases={cases}"
    f" fullwarp_one_pair_offsets={one_pair_full}/64"
    f" fullwarp_two_pair_offsets={two_pair_full}/64"
    f" pair_loads_per_warp_max={max_pair_loads}"
    " partial_source_active=1 pair_selection_exact=1"
)
