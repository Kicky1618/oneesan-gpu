#!/usr/bin/env python3

BLOCK = 64
WARP = 32
HEIGHTS = 30  # MAXW+2 for W=28

max_padding_per_height = BLOCK - 1
max_padding_per_owner = HEIGHTS * max_padding_per_height
cases = 0

# Once a height starts on a 64-entry boundary, every q*32 warp stripe starts
# at offset 0 or 32 inside its block. Partial final stripes are prefixes of the
# same interval, so all active lanes use exactly one block base.
for q in range(128):
    start = q * WARP
    for active in range(1, WARP + 1):
        first_block = start // BLOCK
        last_block = (start + active - 1) // BLOCK
        assert first_block == last_block, (q, active, start, first_block, last_block)
        cases += 1

# Padding to the next selected block boundary is always in [0, 63].
for cursor_mod in range(BLOCK):
    pad = (-cursor_mod) & (BLOCK - 1)
    assert 0 <= pad <= max_padding_per_height
    assert (cursor_mod + pad) % BLOCK == 0
    cases += 1

print(
    "rankchunk32-block64-align-proof OK"
    f" cases={cases} block={BLOCK} warp={WARP}"
    " height_align=64 block_base_loads_per_warp_max=1"
    f" padding_per_height_max={max_padding_per_height}"
    f" padding_per_owner_entries_max={max_padding_per_owner}"
    f" padding_per_owner_bytes_max={max_padding_per_owner * 4}"
)
