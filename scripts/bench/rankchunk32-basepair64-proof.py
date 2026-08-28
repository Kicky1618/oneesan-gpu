#!/usr/bin/env python3
from functools import lru_cache

K = 14
BASE_BITS = 22
DELTA_BITS = 8
HALF = 32
MAX_L_PER_CODE = K // 2

# Reproduce the LOW factor recurrence used by gridfp_low_rank16_plan.cpp.
# Sum all legal LOW codes over starting heights and all L digits in them.
total_codes = 0
total_l = 0
for h0 in range(K + 2):
    @lru_cache(None)
    def dp(pos: int, h: int):
        if pos < 0:
            return (1, 0) if h == 0 else (0, 0)
        if h < 0 or h > pos + 1:
            return (0, 0)
        count = lsum = 0
        c, s = dp(pos - 1, h)  # N
        count += c; lsum += s
        if h > 0:              # R
            c, s = dp(pos - 1, h - 1)
            count += c; lsum += s
        c, s = dp(pos - 1, h + 1)  # L
        count += c; lsum += s + c
        return count, lsum

    c, s = dp(K - 1, h0)
    total_codes += c
    total_l += s

assert total_codes == 1_201_917, total_codes
assert total_l == 3_720_805, total_l
assert total_l < (1 << BASE_BITS)
max_half_delta = HALF * MAX_L_PER_CODE
assert max_half_delta == 224
assert max_half_delta < (1 << DELTA_BITS)
assert BASE_BITS + DELTA_BITS <= 32

# For an unaligned full warp, a 64-code packed pair needs one word unless the
# 32-lane interval crosses the 64-code group boundary.
one_word = two_word = 0
for off in range(64):
    groups = 1 + int(off + 31 >= 64)
    if groups == 1: one_word += 1
    else: two_word += 1
assert (one_word, two_word) == (33, 31)

print(
    "rankchunk32-basepair64-proof OK"
    f" low_codes={total_codes} total_l_digits={total_l}"
    f" base_bits={BASE_BITS} base_limit={1 << BASE_BITS}"
    f" half_delta_max={max_half_delta} delta_bits={DELTA_BITS}"
    " packed_bits=30 pair_bytes=4 codes_per_pair=64"
    " block_base_bytes_per_code=0.0625"
    f" unaligned_one_word_offsets={one_word}/64"
    f" unaligned_two_word_offsets={two_word}/64"
    " aligned32_words_per_warp=1"
)
