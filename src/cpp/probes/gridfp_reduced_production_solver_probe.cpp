#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_channel_probe_main_unused
#include "gridfp_reduced_production_channel_probe.cpp"
#pragma pop_macro("main")

#include <functional>

namespace {

static constexpr std::uint64_t MOD = 4294967291ULL;
using ModVec = std::map<Key, std::uint64_t>;

void add_mod(ModVec& v, Key k, std::uint64_t x) {
    if (!x) return;
    auto& z = v[k];
    z += x;
    if (z >= MOD) z %= MOD;
    if (!z) v.erase(k);
}

ModVec apply_columns(
    const ModVec& in,
    const std::function<Vec(Key)>& column
) {
    ModVec out;
    for (const auto& [k, value] : in) {
        for (const auto& [d, c] : column(k)) {
            if (c != 1 && c != -1) fail("solver coefficient outside +/-1");
            const std::uint64_t x = c > 0 ? value : (value ? MOD - value : 0);
            add_mod(out, d, x);
        }
    }
    return out;
}

ModVec full_step_mod(const ModVec& in, int W, int p, bool reverse) {
    return apply_columns(in, [&](Key k) { return step_basis(k, W, p, reverse); });
}

ModVec project_mod(const ModVec& in, int W, int p, bool reverse) {
    return apply_columns(in, [&](Key k) { return project_term(k, 1, W, p, reverse); });
}

ModVec full_row_mod(ModVec v, int W, bool reverse) {
    if (!reverse) {
        for (int p = W - 1; p >= 1; --p) v = full_step_mod(v, W, p, false);
    } else {
        for (int p = 1; p < W; ++p) v = full_step_mod(v, W, p, true);
    }
    return v;
}

ModVec reduced_row_mod(ModVec v, int W, bool reverse) {
    if (!reverse) {
        for (int p = W - 1; p >= 3; --p) {
            v = full_step_mod(v, W, p, false);
            v = project_mod(v, W, p - 1, false);
        }
        v = full_step_mod(v, W, 2, false);
        v = full_step_mod(v, W, 1, false);
    } else {
        for (int p = 1; p <= W - 3; ++p) {
            v = full_step_mod(v, W, p, true);
            v = project_mod(v, W, p + 1, true);
        }
        v = full_step_mod(v, W, W - 2, true);
        v = full_step_mod(v, W, W - 1, true);
    }
    return v;
}

std::uint64_t solve_compare(int n) {
    const int W = n + 1;
    const Key initial{false, MateID(R) << (2 * (W - 1))};
    ModVec full{{initial, 1}}, reduced = full;

    for (int row = 0; row < W; ++row) {
        const bool reverse = (row & 1) != 0;
        full = full_row_mod(std::move(full), W, reverse);
        reduced = reduced_row_mod(std::move(reduced), W, reverse);
        if (full != reduced)
            fail("complete solver vector mismatch n=" + std::to_string(n) +
                 " row=" + std::to_string(row + 1));
        for (const auto& [k, c] : reduced) {
            (void)c;
            if (k.blocked) fail("row boundary retained blocked state");
        }
    }

    const auto it = full.find(Key{false, MateID(R)});
    return it == full.end() ? 0 : it->second;
}

} // namespace

int main(int argc, char** argv) {
    const int max_n = argc > 1 ? std::atoi(argv[1]) : 7;
    if (max_n < 3 || max_n > 9) return 2;
    const std::vector<std::uint64_t> known = {
        0ULL, 2ULL, 12ULL, 184ULL, 8512ULL, 1262816ULL,
        575780564ULL, 3381038999ULL
    };

    for (int n = 3; n <= max_n; ++n) {
        const std::uint64_t answer = solve_compare(n);
        if (n < static_cast<int>(known.size()) && answer != known[static_cast<std::size_t>(n)])
            fail("known residue n=" + std::to_string(n) +
                 " got=" + std::to_string(answer) +
                 " want=" + std::to_string(known[static_cast<std::size_t>(n)]));
        std::cout << "n=" << n
                  << " W=" << n + 1
                  << " residue=" << answer
                  << " complete_vectors=OK"
                  << " row_boundary_main_only=OK\n";
    }
    std::cout << "ALL_OK complete_reduced_production_solver=1 modulus=" << MOD << '\n';
    return 0;
}
