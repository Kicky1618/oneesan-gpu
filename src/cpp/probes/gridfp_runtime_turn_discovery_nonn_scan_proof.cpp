#include "../../common/gridfp_transition.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <random>

namespace {
using namespace oneesan::gridfp;

struct ScanResult {
    std::array<int, 28> candidate_q{};
    int candidate_count = 0;
    int final_balance = 0;
    int iterations = 0;
};

ScanResult full_scan(MateID mate, int width) {
    ScanResult out{};
    int bal = 0;
    for (int q = 2; q < width; ++q) {
        ++out.iterations;
        const MateValue v = mget(mate, q);
        if (bal == 0 && v == R)
            out.candidate_q[out.candidate_count++] = q;
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
    }
    out.final_balance = bal;
    return out;
}

ScanResult nonn_scan(MateID mate, int width) {
    ScanResult out{};
    std::uint32_t mask = mate_non_n_mask(mate, width) & ~std::uint32_t(3u);
    int bal = 0;
    while (mask) {
        ++out.iterations;
        const int q = mate_lsb_index32(mask);
        const MateValue v = mget(mate, q);
        if (bal == 0 && v == R)
            out.candidate_q[out.candidate_count++] = q;
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
        mask &= mask - 1u;
    }
    out.final_balance = bal;
    return out;
}

bool same(const ScanResult& a, const ScanResult& b) {
    if (a.candidate_count != b.candidate_count ||
        a.final_balance != b.final_balance) return false;
    for (int i = 0; i < a.candidate_count; ++i)
        if (a.candidate_q[i] != b.candidate_q[i]) return false;
    return true;
}

MateID ternary_mate(std::uint64_t code, int width) {
    MateID mate = 0;
    for (int q = 0; q < width; ++q) {
        const unsigned digit = unsigned(code % 3);
        code /= 3;
        const MateValue v = digit == 0 ? N : (digit == 1 ? R : L);
        mate = mset(mate, q, v);
    }
    return mate;
}

std::uint64_t pow3(int n) {
    std::uint64_t z = 1;
    while (n--) z *= 3;
    return z;
}
} // namespace

int main() {
    std::uint64_t exhaustive_cases = 0;
    std::uint64_t random_cases = 0;
    std::uint64_t full_iterations = 0;
    std::uint64_t nonn_iterations = 0;
    std::uint64_t strict_reductions = 0;
    int max_candidates = 0;

    for (int width = 2; width <= 10; ++width) {
        const std::uint64_t cases = pow3(width);
        for (std::uint64_t code = 0; code < cases; ++code) {
            const MateID mate = ternary_mate(code, width);
            const ScanResult full = full_scan(mate, width);
            const ScanResult nonn = nonn_scan(mate, width);
            ++exhaustive_cases;
            if (!same(full, nonn)) return 2;
            if (nonn.iterations > full.iterations) return 3;
            full_iterations += full.iterations;
            nonn_iterations += nonn.iterations;
            strict_reductions += nonn.iterations < full.iterations;
            if (full.candidate_count > max_candidates)
                max_candidates = full.candidate_count;
        }
    }

    std::mt19937_64 rng(0x7475726e6e6f6e6eULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int width = 8 + int(rng() % 21);
        MateID mate = 0;
        for (int q = 0; q < width; ++q) {
            const std::uint64_t x = rng();
            const unsigned digit = unsigned(x % 3);
            const MateValue v = digit == 0 ? N : (digit == 1 ? R : L);
            mate = mset(mate, q, v);
        }
        // The optimized production branch is entered only for pair==NN.
        mate = msetpair(mate, 1, NN);
        const ScanResult full = full_scan(mate, width);
        const ScanResult nonn = nonn_scan(mate, width);
        ++random_cases;
        if (!same(full, nonn)) return 4;
        if (nonn.iterations > full.iterations) return 5;
        full_iterations += full.iterations;
        nonn_iterations += nonn.iterations;
        strict_reductions += nonn.iterations < full.iterations;
        if (full.candidate_count > max_candidates)
            max_candidates = full.candidate_count;
    }

    if (!strict_reductions || nonn_iterations >= full_iterations) return 6;
    const double iteration_ratio =
        double(nonn_iterations) / double(full_iterations);
    std::cout << "gridfp-runtime-turn-discovery-nonn-scan-proof OK"
              << " exhaustive_width_max=10 exhaustive_cases=" << exhaustive_cases
              << " random_width_max=28 random_cases=" << random_cases
              << " candidate_sequence_exact=1 balance_exact=1 stop_exact=1"
              << " nonn_iterations_le_full=1 strict_reductions=" << strict_reductions
              << " full_iterations=" << full_iterations
              << " nonn_iterations=" << nonn_iterations
              << " iteration_ratio=" << iteration_ratio
              << " max_candidates=" << max_candidates << '\n';
    return 0;
}
