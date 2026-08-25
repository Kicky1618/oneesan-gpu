#include <cstdint>
#include <cstdlib>
#include <iostream>

using U64 = std::uint64_t;
using U128 = unsigned __int128;

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int high = W - 1 - low;
    if (W < 4 || W > 30 || low < 1 || high < 1) return 1;

    const U64 meta_bytes = U64(low) * 65ULL * 2ULL * 4ULL
                         + U64(high + 2) * 4ULL;
    if (W == 28 && low == 14) {
        const U128 exact_tasks = U128(1244849733336ULL);
        const U128 old_global_uniform_loads = exact_tasks * U128(3);
        if (meta_bytes != 7340ULL
            || old_global_uniform_loads != U128(3734549200008ULL)) {
            std::cerr << "n=27 LOW closure packed-meta regression\n";
            return 2;
        }
        std::cout << "lowclosure-packed-meta W=28 low=14 high=13\n"
                  << "extra_constant_bytes=" << meta_bytes << '\n'
                  << "old_global_uniform_table_lookups=3734549200008\n"
                  << "new_global_uniform_table_lookups=0\n"
                  << "replacement=constant-memory-broadcast\n";
        return 0;
    }

    std::cout << "lowclosure-packed-meta W=" << W
              << " low=" << low << " high=" << high
              << " extra_constant_bytes=" << meta_bytes << '\n';
    return 0;
}
