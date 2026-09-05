#include <algorithm>
#include <cstdint>
#include <iostream>

namespace {
using Rank64 = std::uint64_t;

Rank64 choose(int n, int k) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    Rank64 z = 1;
    for (int i = 1; i <= k; ++i)
        z = z * Rank64(n - k + i) / Rank64(i);
    return z;
}

std::uint32_t suffix_mask(int pos, int len) {
    if (pos >= len) return 0;
    const std::uint32_t hi = len == 32 ? ~0u : ((std::uint32_t(1) << len) - 1u);
    const std::uint32_t lo = pos ? ((std::uint32_t(1) << pos) - 1u) : 0u;
    return hi & ~lo;
}

struct Stats {
    std::uint64_t iterations = 0;
    std::uint64_t choose_calls = 0;
};

std::uint32_t old_unrank(
    int len, int ones, int mark0, int mark1, Rank64 rank, Stats& st
) {
    std::uint32_t support = 0;
    int left = ones;
    bool seen_mark = false;
    for (int pos = 0; pos < len; ++pos) {
        ++st.iterations;
        if (!left) break;
        const int remaining = len - pos;
        if (left == remaining) {
            support |= suffix_mask(pos, len);
            break;
        }
        const int rem = remaining - 1;
        const int future_marks = (mark0 > pos ? 1 : 0) + (mark1 > pos ? 1 : 0);
        Rank64 zero_count = choose(rem, left);
        ++st.choose_calls;
        if (!seen_mark) {
            zero_count -= choose(rem - future_marks, left);
            ++st.choose_calls;
        }
        if (rank < zero_count) continue;
        rank -= zero_count;
        support |= std::uint32_t(1) << pos;
        --left;
        if (pos == mark0 || pos == mark1) seen_mark = true;
    }
    return support;
}

std::uint32_t new_unrank_adjacent(
    int len, int ones, int mark0, int mark1, Rank64 rank, Stats& st
) {
    const int a = std::min(mark0, mark1);
    const int b = std::max(mark0, mark1);
    if (b != a + 1) return 0xffffffffu;

    std::uint32_t support = 0;
    int left = ones;

    // Before the adjacent marked pair, both marks remain in the future.
    for (int pos = 0; pos < a; ++pos) {
        ++st.iterations;
        if (!left) return 0xffffffffu; // no valid completion can reach this
        const int remaining = len - pos;
        if (left == remaining) {
            support |= suffix_mask(pos, len);
            return support;
        }
        const int rem = remaining - 1;
        Rank64 zero_count = choose(rem, left) - choose(rem - 2, left);
        st.choose_calls += 2;
        if (rank < zero_count) continue;
        rank -= zero_count;
        support |= std::uint32_t(1) << pos;
        --left;
    }

    if (!left) return 0xffffffffu;
    const int remaining_at_a = len - a;
    if (left == remaining_at_a) {
        support |= suffix_mask(a, len);
        return support;
    }

    // At the first mark, the zero branch forces the second mark to one.
    // Its suffix count is C(len-a-2,left-1). If we take the one branch,
    // the mark constraint is already satisfied and the rest is ordinary.
    const int suffix = len - a - 2;
    const Rank64 zero_count_a = choose(suffix, left - 1);
    ++st.choose_calls;
    ++st.iterations; // the first marked position
    int start = b;
    if (rank < zero_count_a) {
        support |= std::uint32_t(1) << b;
        --left;
        start = b + 1;
    } else {
        rank -= zero_count_a;
        support |= std::uint32_t(1) << a;
        --left;
    }

    for (int pos = start; pos < len; ++pos) {
        ++st.iterations;
        if (!left) break;
        const int remaining = len - pos;
        if (left == remaining) {
            support |= suffix_mask(pos, len);
            break;
        }
        const int rem = remaining - 1;
        const Rank64 zero_count = choose(rem, left);
        ++st.choose_calls;
        if (rank < zero_count) continue;
        rank -= zero_count;
        support |= std::uint32_t(1) << pos;
        --left;
    }
    return support;
}
} // namespace

int main() {
    Stats old_total{}, new_total{};
    std::uint64_t cases = 0;
    std::uint64_t supports = 0;

    // Production owner windows use len=L-1, which is 4..14 for W=8..28.
    for (int len = 4; len <= 14; ++len) {
        for (int a = 0; a + 1 < len; ++a) {
            for (int ones = 1; ones <= len; ++ones) {
                const Rank64 count = choose(len, ones) - choose(len - 2, ones);
                for (Rank64 rank = 0; rank < count; ++rank) {
                    Stats so{}, sn{};
                    const std::uint32_t oldv = old_unrank(
                        len, ones, a + 1, a, rank, so); // production order is often reversed
                    const std::uint32_t newv = new_unrank_adjacent(
                        len, ones, a + 1, a, rank, sn);
                    if (oldv != newv) {
                        std::cerr << "mismatch len=" << len << " a=" << a
                                  << " ones=" << ones << " rank=" << rank
                                  << " old=" << oldv << " new=" << newv << '\n';
                        return 2;
                    }
                    if (((oldv >> a) & 3u) == 0u) return 3;
                    if (__builtin_popcount(oldv) != ones) return 4;
                    old_total.iterations += so.iterations;
                    old_total.choose_calls += so.choose_calls;
                    new_total.iterations += sn.iterations;
                    new_total.choose_calls += sn.choose_calls;
                    ++cases;
                    ++supports;
                }
            }
        }
    }

    if (!(new_total.iterations < old_total.iterations)) return 5;
    if (!(new_total.choose_calls < old_total.choose_calls)) return 6;
    std::cout << "gridfp-component-support-adjacent-marks-proof OK"
              << " exhaustive_cases=" << cases
              << " supports=" << supports
              << " len_min=4 len_max=14"
              << " old_iterations=" << old_total.iterations
              << " new_iterations=" << new_total.iterations
              << " iteration_ratio=" << double(new_total.iterations) / double(old_total.iterations)
              << " old_choose_calls=" << old_total.choose_calls
              << " new_choose_calls=" << new_total.choose_calls
              << " choose_call_ratio=" << double(new_total.choose_calls) / double(old_total.choose_calls)
              << " adjacent_exact=1 lexicographic_exact=1\n";
    return 0;
}
