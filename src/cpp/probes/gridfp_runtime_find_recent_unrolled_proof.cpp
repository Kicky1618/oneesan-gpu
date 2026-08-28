#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
int loop_find(const std::uint64_t* a, int n, std::uint64_t needle) {
    for (int i = n - 1; i >= 0; --i)
        if (a[i] == needle) return i;
    return -1;
}

int unrolled_find(const std::uint64_t* a, int n, std::uint64_t needle) {
#define RP_CHECK_CASE(I) case I: if (a[(I)-1] == needle) return (I)-1; [[fallthrough]]
    switch (n) {
    RP_CHECK_CASE(20); RP_CHECK_CASE(19); RP_CHECK_CASE(18); RP_CHECK_CASE(17);
    RP_CHECK_CASE(16); RP_CHECK_CASE(15); RP_CHECK_CASE(14); RP_CHECK_CASE(13);
    RP_CHECK_CASE(12); RP_CHECK_CASE(11); RP_CHECK_CASE(10); RP_CHECK_CASE(9);
    RP_CHECK_CASE(8); RP_CHECK_CASE(7); RP_CHECK_CASE(6); RP_CHECK_CASE(5);
    RP_CHECK_CASE(4); RP_CHECK_CASE(3); RP_CHECK_CASE(2); RP_CHECK_CASE(1);
    case 0: break;
    default: return -2;
    }
#undef RP_CHECK_CASE
    return -1;
}
}

int main() {
    std::mt19937_64 rng(0x756e726f6c6c6564ULL);
    std::array<std::uint64_t, 20> a{};
    std::uint64_t cases = 0;
    for (int n = 0; n <= 20; ++n) {
        for (int t = 0; t < 200000; ++t) {
            for (int i = 0; i < n; ++i) a[std::size_t(i)] = rng();
            std::uint64_t needle = rng();
            if (n && (t & 1)) needle = a[std::size_t(rng() % std::uint64_t(n))];
            const int want = loop_find(a.data(), n, needle);
            const int got = unrolled_find(a.data(), n, needle);
            ++cases;
            if (want != got) {
                std::cerr << "mismatch n=" << n << " want=" << want << " got=" << got << '\n';
                return 2;
            }
        }
    }
    std::cout << "gridfp-runtime-find-recent-unrolled-proof OK"
              << " max_pairs=20 cases=" << cases
              << " exact=1 order=recent_first loop_control=fallthrough_switch\n";
    return 0;
}
