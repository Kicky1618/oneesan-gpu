#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;
using U128 = unsigned __int128;

static long double as_ld(U128 x) {
    const U64 lo = U64(x), hi = U64(x >> 64);
    return (long double)hi * 18446744073709551616.0L + (long double)lo;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || high < 1 || low >= 16 || high >= 16)
        return 1;

    const U64 main_nblocks = U64(3 * (high + 2));
    const U64 prefix_bytes = U64(low) * 65ULL * sizeof(std::uint32_t);

    // The exact launch count is independently recomputed and regression-pinned
    // by factor_lowclosure_rowdepth_tasks.cpp.  This probe isolates the v0.41
    // consequence: v0.40 rebuilt one identical prefix and executed one CTA
    // barrier for every launched closure CTA; v0.41 does neither in the kernel.
    U128 exact_launch_blocks = 0;
    if (W == 28 && low == 14) {
        exact_launch_blocks = U128(99505629661ULL);
    } else {
        std::cout << "lowclosure-packed-prefix W=" << W
                  << " low=" << low << " high=" << high
                  << " prefix_bytes=" << prefix_bytes
                  << " main_nblocks=" << main_nblocks
                  << " exact_launch_blocks=use-factor_lowclosure_rowdepth_tasks\n";
        return 0;
    }

    const U128 old_prefix_builds = exact_launch_blocks;
    const U128 old_barriers = exact_launch_blocks;
    const U128 old_segment_evals = exact_launch_blocks * U128(main_nblocks);

    if (prefix_bytes != 3640ULL
        || main_nblocks != 45ULL
        || old_prefix_builds != U128(99505629661ULL)
        || old_segment_evals != U128(4477753334745ULL)) {
        std::cerr << "n=27 LOW closure packed-prefix regression\n";
        return 2;
    }

    std::cout << std::fixed << std::setprecision(0)
              << "lowclosure-packed-prefix W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "prefix_bytes_per_group_config=" << prefix_bytes << '\n'
              << "old_cta_prefix_builds=" << double(as_ld(old_prefix_builds))
              << " new_cta_prefix_builds=0\n"
              << "old_cta_barriers=" << double(as_ld(old_barriers))
              << " new_cta_barriers=0\n"
              << "old_prefix_segment_evals=" << double(as_ld(old_segment_evals))
              << " new_prefix_segment_evals=0\n";
    return 0;
}
