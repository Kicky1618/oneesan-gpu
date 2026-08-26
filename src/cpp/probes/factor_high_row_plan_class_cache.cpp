#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

static U64 align_up(U64 x, U64 a) { return (x + a - 1) / a * a; }

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || low >= 31 || high < 1) return 1;

    const U64 masks = U64(1) << low;
    const U64 classes = U64(low + 1);
    const U64 full_cap = U64(W / 2);
    const U64 prefix_bytes = U64(high + 3) * 8;
    const U64 low_count_bytes = U64(high + 2) * 2;
    const U64 plan_bytes = align_up(prefix_bytes + low_count_bytes, 8) + 8;

    const U64 old_entries = masks * full_cap;
    const U64 class_entries = classes * full_cap;
    const U64 old_bytes = old_entries * plan_bytes;
    const U64 class_bytes = class_entries * plan_bytes;
    const U64 old_plan_builds = masks * (full_cap - 1);
    const U64 class_plan_builds = classes * (full_cap - 1);

    if (W == 28 && low == 14) {
        if (plan_bytes != 168ULL
            || old_entries != 229376ULL
            || class_entries != 210ULL
            || old_bytes != 38535168ULL
            || class_bytes != 35280ULL
            || old_plan_builds != 212992ULL
            || class_plan_builds != 195ULL) {
            std::cerr << "n=27 HIGH row-plan class-cache regression mismatch\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "high-row-plan-class-cache W=" << W
              << " low=" << low << " high=" << high << '\n'
              << "masks=" << masks << " classes=" << classes
              << " full_cap=" << full_cap << '\n'
              << "plan_bytes=" << plan_bytes << '\n'
              << "old_plan_entries=" << old_entries
              << " class_plan_entries=" << class_entries << '\n'
              << "old_pinned_bytes=" << old_bytes
              << " class_pinned_bytes=" << class_bytes
              << " pinned_reduction_pct="
              << 100.0 * (1.0 - double(class_bytes) / double(old_bytes)) << '\n'
              << "old_plan_builds=" << old_plan_builds
              << " class_plan_builds=" << class_plan_builds
              << " build_reduction_pct="
              << 100.0 * (1.0 - double(class_plan_builds) / double(old_plan_builds))
              << '\n';
    return 0;
}
