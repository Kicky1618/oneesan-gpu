#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

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

// 17 is a primitive root modulo 65521. zeta has order 16.
const std::uint32_t ZETA = mod_pow(17, (MOD - 1) / 16);
const std::uint32_t ZETA_INV = mod_pow(ZETA, 15);

// qutrit code per frontier slot: 0=empty, 1=positive global direction
// (E for horizontal / S for vertical), 2=negative direction (W/N).
std::uint32_t get(std::uint64_t state, unsigned slot) {
    return (state >> (2 * slot)) & 3u;
}
std::uint64_t set(std::uint64_t state, unsigned slot, std::uint32_t v) {
    const auto sh = 2 * slot;
    return (state & ~(std::uint64_t{3} << sh)) | (std::uint64_t{v} << sh);
}

struct EdgeFlow {
    bool used = false;
    bool incoming = false;
    int dir = 0; // E=0, N=1, W=2, S=3 (counter-clockwise quarter-turn coordinates)
};

EdgeFlow left_edge(std::uint32_t c) {
    if (c == 1) return {true, true, 0};   // E into vertex
    if (c == 2) return {true, false, 2};  // W out of vertex
    return {};
}
EdgeFlow up_edge(std::uint32_t c) {
    if (c == 1) return {true, true, 3};   // S into vertex
    if (c == 2) return {true, false, 1};  // N out of vertex
    return {};
}
EdgeFlow down_edge(std::uint32_t c) {
    if (c == 1) return {true, false, 3};  // S out of vertex
    if (c == 2) return {true, true, 1};   // N into vertex
    return {};
}
EdgeFlow right_edge(std::uint32_t c) {
    if (c == 1) return {true, false, 0};  // E out of vertex
    if (c == 2) return {true, true, 2};   // W into vertex
    return {};
}

std::uint32_t turn_weight(int in_dir, int out_dir) {
    const int d = (out_dir - in_dir + 4) & 3;
    if (d == 0) return 1;
    if (d == 1) return ZETA;
    if (d == 3) return ZETA_INV;
    return 0; // U-turn cannot be part of a simple degree-2 local path.
}

struct Transition {
    std::uint8_t down = 0;
    std::uint8_t right = 0;
    std::uint32_t weight = 0;
};

std::array<std::array<Transition, 6>, 9> normal_trans{};
std::array<std::uint8_t, 9> normal_count{};

void init_transitions() {
    for (unsigned l = 0; l < 3; ++l) {
        for (unsigned u = 0; u < 3; ++u) {
            const unsigned key = l * 3 + u;
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
                    if (!w) continue;
                    auto& t = normal_trans[key][normal_count[key]++];
                    t = {static_cast<std::uint8_t>(d), static_cast<std::uint8_t>(r), w};
                }
            }
        }
    }
}

using Table = std::unordered_map<std::uint64_t, std::uint32_t>;
void add(Table& t, std::uint64_t state, std::uint32_t value) {
    auto [it, inserted] = t.try_emplace(state, value);
    if (!inserted) {
        std::uint32_t x = it->second + value;
        if (x >= MOD) x -= MOD;
        it->second = x;
    }
}

struct Result {
    std::uint32_t residue = 0;
    std::size_t peak_states = 0;
    double seconds = 0;
};

Result solve(unsigned n, bool verbose) {
    const auto start = std::chrono::steady_clock::now();
    Table cur, next;
    cur.reserve(1024);
    cur[0] = 1;
    std::size_t peak = 1;

    for (unsigned y = 0; y < n; ++y) {
        for (unsigned x = 0; x < n; ++x) {
            next.clear();
            next.reserve(cur.size() * 2 + 64);
            const unsigned p = x, q = x + 1;
            const bool source = x == 0 && y == 0;
            const bool target = x + 1 == n && y + 1 == n;
            const bool allow_down = y + 1 < n;
            const bool allow_right = x + 1 < n;

            for (const auto& [state, count] : cur) {
                const auto l = get(state, p);
                const auto u = get(state, q);
                auto base = set(set(state, p, 0), q, 0);

                if (source) {
                    if (state != 0 || l || u) continue;
                    // Endpoint correction from the fixed external return path:
                    // E->E contributes 1; E->S contributes zeta^-1.
                    if (allow_down) add(next, set(base, p, 1), ZETA_INV);
                    if (allow_right) add(next, set(base, q, 1), 1);
                    continue;
                }

                if (target) {
                    // Only the desired s->t line can terminate here.  Boundary
                    // conditions imply the incoming arrow must be positive.
                    if (state != (std::uint64_t{l} << (2 * p)) +
                                 (std::uint64_t{u} << (2 * q))) continue;
                    if (l == 1 && u == 0) add(next, base, count);              // incoming E
                    if (l == 0 && u == 1) add(next, base, std::uint64_t{count} * ZETA % MOD); // incoming S
                    continue;
                }

                const unsigned key = l * 3 + u;
                for (unsigned ti = 0; ti < normal_count[key]; ++ti) {
                    const auto& tr = normal_trans[key][ti];
                    if (!allow_down && tr.down != 0) continue;
                    if (!allow_right && tr.right != 0) continue;
                    auto out = set(set(base, p, tr.down), q, tr.right);
                    add(next, out, std::uint64_t{count} * tr.weight % MOD);
                }
            }

            cur.swap(next);
            peak = std::max(peak, cur.size());
            if (verbose) std::cerr << "o0 n=" << n << " y=" << y << " x=" << x << " states=" << cur.size() << '\n';
        }

        // Kink reset: old slots [0,n-1] become new slots [1,n].
        next.clear();
        next.reserve(cur.size() + 16);
        for (const auto& [state, count] : cur) {
            if (get(state, n) != 0) continue;
            add(next, state << 2, count);
        }
        cur.swap(next);
    }

    const auto it = cur.find(0);
    const auto end = std::chrono::steady_clock::now();
    return {it == cur.end() ? 0u : it->second, peak,
            std::chrono::duration<double>(end - start).count()};
}

} // namespace

int main(int argc, char** argv) {
    init_transitions();
    std::cerr << "mod=" << MOD << " zeta=" << ZETA
              << " zeta^8=" << mod_pow(ZETA, 8) << '\n';
    unsigned first = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 2;
    unsigned last = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 10;
    bool verbose = argc > 3;
    for (unsigned n = first; n <= last; ++n) {
        const auto r = solve(n, verbose);
        std::cout << n << ' ' << r.residue << " peak=" << r.peak_states
                  << " sec=" << r.seconds << '\n';
    }
}
