#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t MOD = 65521;

std::uint32_t mod_pow(std::uint32_t a, std::uint32_t e) {
    std::uint64_t r = 1;
    while (e) {
        if (e & 1u) r = r * a % MOD;
        a = std::uint64_t{a} * a % MOD;
        e >>= 1;
    }
    return static_cast<std::uint32_t>(r);
}

const std::uint32_t ZETA = mod_pow(17, (MOD - 1) / 16);
const std::uint32_t ZETA_INV = mod_pow(ZETA, 15);

int charge(std::uint32_t code) {
    return code == 1 ? 1 : code == 2 ? -1 : 0;
}

int code_from_charge(int q) {
    if (q == 0) return 0;
    if (q == 1) return 1;
    if (q == -1) return 2;
    return -1;
}

struct EdgeFlow {
    bool used = false;
    bool incoming = false;
    int dir = 0;
};

EdgeFlow left_edge(std::uint32_t c) {
    if (c == 1) return {true, true, 0};
    if (c == 2) return {true, false, 2};
    return {};
}
EdgeFlow up_edge(std::uint32_t c) {
    if (c == 1) return {true, true, 3};
    if (c == 2) return {true, false, 1};
    return {};
}
EdgeFlow down_edge(std::uint32_t c) {
    if (c == 1) return {true, false, 3};
    if (c == 2) return {true, true, 1};
    return {};
}
EdgeFlow right_edge(std::uint32_t c) {
    if (c == 1) return {true, false, 0};
    if (c == 2) return {true, true, 2};
    return {};
}

std::uint32_t turn_weight(int in_dir, int out_dir) {
    const int d = (out_dir - in_dir + 4) & 3;
    if (d == 0) return 1;
    if (d == 1) return ZETA;
    if (d == 3) return ZETA_INV;
    return 0;
}

// [left][up][down][right]
std::array<std::uint32_t, 81> vertex{};

void init_vertex() {
    for (unsigned l = 0; l < 3; ++l) {
        for (unsigned u = 0; u < 3; ++u) {
            for (unsigned d = 0; d < 3; ++d) {
                for (unsigned r = 0; r < 3; ++r) {
                    const std::array<EdgeFlow, 4> e = {
                        left_edge(l), up_edge(u), down_edge(d), right_edge(r)};
                    int used = 0, nin = 0, nout = 0, idir = 0, odir = 0;
                    for (const auto& x : e) {
                        if (!x.used) continue;
                        ++used;
                        if (x.incoming) { ++nin; idir = x.dir; }
                        else { ++nout; odir = x.dir; }
                    }
                    std::uint32_t w = 0;
                    if (used == 0) w = 1;
                    else if (used == 2 && nin == 1 && nout == 1) w = turn_weight(idir, odir);
                    vertex[((l * 3 + u) * 3 + d) * 3 + r] = w;
                }
            }
        }
    }
}

std::uint32_t vw(unsigned l, unsigned u, unsigned d, unsigned r) {
    return vertex[((l * 3 + u) * 3 + d) * 3 + r];
}

std::uint64_t pow3(unsigned e) {
    std::uint64_t x = 1;
    while (e--) x *= 3;
    return x;
}

std::uint32_t add_mod(std::uint32_t a, std::uint32_t b) {
    auto x = a + b;
    if (x >= MOD) x -= MOD;
    return x;
}

std::uint32_t mul_mod(std::uint32_t a, std::uint32_t b) {
    return static_cast<std::uint32_t>(std::uint64_t{a} * b % MOD);
}

struct SolveResult {
    std::uint32_t residue = 0;
    double seconds = 0;
    std::uint64_t states = 0;
};

SolveResult solve(unsigned n) {
    if (n < 2 || n > 15) throw std::runtime_error("n must be in [2,15]");
    const auto total = pow3(n);
    std::vector<std::uint32_t> a(total, 0);

    // State just after the source (0,0), with the horizontal carry at x=1
    // omitted.  Global oriented charge is +1, so carry = 1 - charge(vertical).
    a[0] = 1;          // source -> right, carry=+1
    a[1] = ZETA_INV;   // source -> down, vertical digit 0=+1, carry=0

    std::vector<std::int8_t> total_charge(total);
    for (std::uint64_t i = 1; i < total; ++i) {
        total_charge[i] = static_cast<std::int8_t>(total_charge[i / 3] + charge(i % 3));
    }

    const auto t0 = std::chrono::steady_clock::now();

    auto gate = [&](unsigned x, bool allow_down, bool allow_right) {
        const std::uint64_t stride = pow3(x);
        const std::uint64_t block = stride * 3;
        for (std::uint64_t hi = 0; hi < total; hi += block) {
            for (std::uint64_t lo = 0; lo < stride; ++lo) {
                const std::uint64_t base = hi + lo; // digit x == 0
                const int B = total_charge[base];   // charge of all other vertical digits
                const std::uint32_t old[3] = {
                    a[base], a[base + stride], a[base + 2 * stride]};
                std::uint32_t out[3]{};

                for (unsigned u = 0; u < 3; ++u) {
                    if (!old[u]) continue;
                    const int S = B + charge(u);
                    const int l = code_from_charge(1 - S);
                    if (l < 0) continue;
                    for (unsigned d = 0; d < 3; ++d) {
                        if (!allow_down && d != 0) continue;
                        const int Sp = B + charge(d);
                        const int r = code_from_charge(1 - Sp);
                        if (r < 0) continue;
                        if (!allow_right && r != 0) continue;
                        const auto w = vw(static_cast<unsigned>(l), u, d,
                                          static_cast<unsigned>(r));
                        if (!w) continue;
                        out[d] = add_mod(out[d], mul_mod(old[u], w));
                    }
                }
                a[base] = out[0];
                a[base + stride] = out[1];
                a[base + 2 * stride] = out[2];
            }
        }
    };

    // Complete top row after the already-processed source.
    for (unsigned x = 1; x < n; ++x) gate(x, true, x + 1 < n);

    // Interior rows.
    for (unsigned y = 1; y + 1 < n; ++y) {
        for (unsigned x = 0; x < n; ++x) gate(x, true, x + 1 < n);
    }

    // Bottom row, stopping before the target. No downward edges are allowed.
    for (unsigned x = 0; x + 1 < n; ++x) gate(x, false, true);

    // Before target: either left carry is +1 and up is empty, or carry=0 and
    // the target's up edge is +1.  All earlier bottom-row verticals are zero.
    const auto up_idx = pow3(n - 1);
    std::uint32_t residue = add_mod(a[0], mul_mod(a[up_idx], ZETA));

    const auto t1 = std::chrono::steady_clock::now();
    return {residue, std::chrono::duration<double>(t1 - t0).count(), total};
}

} // namespace

int main(int argc, char** argv) {
    init_vertex();
    unsigned first = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 2;
    unsigned last = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : first;
    std::cerr << "mod=" << MOD << " zeta=" << ZETA << " zeta^8=" << mod_pow(ZETA, 8) << '\n';
    for (unsigned n = first; n <= last; ++n) {
        const auto r = solve(n);
        std::cout << "n=" << n << " residue=" << r.residue
                  << " vertical_states=" << r.states
                  << " sec=" << std::fixed << std::setprecision(6) << r.seconds << '\n';
    }
}
