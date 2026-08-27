#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define main odd_tl_integer_probe_unused_main
#include "odd_tl_integer_normalization_probe.cpp"
#undef main

struct CompiledEdge {
    std::uint32_t src = 0;
    std::int8_t coeff = 0;
};

struct CompiledRow {
    std::uint32_t dst = 0;
    std::uint32_t edge_begin = 0;
    std::uint16_t edge_count = 0;
    std::int8_t self_coeff = 1;
};

struct CompiledPhase {
    std::vector<CompiledRow> rows;
    std::vector<CompiledEdge> edges;
};

struct CompiledStage {
    CompiledPhase phase1; // A and B. Reads old C,D; may overwrite A,B.
    CompiledPhase phase2; // C. Runs after a barrier; reads D only.
};

struct CompiledTransform {
    int n = 0;
    int d = 0;
    std::vector<CompiledStage> stages;
};

static std::uint32_t mul_small_signed(std::uint32_t x, int c) {
    if (c >= 0) return mulm(x, std::uint32_t(c));
    return negm(mulm(x, std::uint32_t(-c)));
}

static void emit_rows(
    CompiledPhase& phase,
    std::uint32_t base,
    std::vector<std::vector<std::pair<std::uint32_t, int>>> const& incoming,
    int self_coeff,
    bool emit_empty
) {
    for (std::uint32_t j = 0; j < incoming.size(); ++j) {
        auto const& in = incoming[j];
        if (!emit_empty && in.empty()) continue;
        if (in.size() > 0xffffu) {
            std::cerr << "too many incoming edges for one row\n";
            std::exit(80);
        }
        CompiledRow row;
        row.dst = base + j;
        row.edge_begin = std::uint32_t(phase.edges.size());
        row.edge_count = std::uint16_t(in.size());
        row.self_coeff = std::int8_t(self_coeff);
        for (auto [src, coeff] : in) {
            if (coeff < -127 || coeff > 127) {
                std::cerr << "coefficient does not fit int8: " << coeff << "\n";
                std::exit(81);
            }
            phase.edges.push_back({src, std::int8_t(coeff)});
        }
        phase.rows.push_back(row);
    }
}

static void ensure_depth(CompiledTransform& out, int depth) {
    if (int(out.stages.size()) <= depth) out.stages.resize(depth + 1);
}

static void compile_node(
    CompiledTransform& out,
    int n,
    int d,
    std::uint32_t base,
    int depth
) {
    if (n <= 1 || d < 0 || d > n) return;
    int da = int(basis(n - 2, d - 2).words.size());
    int d0 = int(basis(n - 2, d).words.size());
    int dp = int(basis(n - 2, d + 2).words.size());
    int total = da + 2 * d0 + dp;
    if (total != int(basis(n, d).words.size())) {
        std::cerr << "branching dimension mismatch n=" << n << " d=" << d << "\n";
        std::exit(82);
    }
    ensure_depth(out, depth);
    auto& stage = out.stages[depth];

    std::uint32_t offA = base;
    std::uint32_t offB = offA + da;
    std::uint32_t offC = offB + d0;
    std::uint32_t offD = offC + d0;
    int s = (d + 1) / 2;

    if (da) {
        std::vector<std::vector<std::pair<std::uint32_t, int>>> inA(da);

        // s * partial_d(C).
        auto const& BC = basis(n - 2, d);
        int sign_base = (d - 3) / 2;
        for (int i = 0; i < int(BC.words.size()); ++i) {
            for (int j = 0; j <= (d - 3) / 2; ++j) {
                int to = map_cup_word(BC.words[i], n - 2, d, 2 * j, 2 * j + 1);
                assert(to >= 0);
                int coeff = ((sign_base - j) & 1) ? -s : s;
                inA[to].push_back({offC + std::uint32_t(i), coeff});
            }
        }

        // Qtilde_d(D), with the common denominator removed.
        int din = d + 2;
        int r = (d - 1) / 2;
        if (r) {
            auto const& BD = basis(n - 2, din);
            for (int i = 0; i < int(BD.words.size()); ++i) {
                auto w = BD.words[i];
                for (int q = 0; q < 2 * r; ++q) {
                    int to = map_two_cups(w, n - 2, din,
                                          {q, q + 3}, {q + 1, q + 2});
                    assert(to >= 0);
                    int coeff = q / 2 + 1;
                    if (q & 1) coeff = -coeff;
                    inA[to].push_back({offD + std::uint32_t(i), coeff});
                }
                for (int a = 0; a <= r; ++a) {
                    for (int b = a + 1; b <= r; ++b) {
                        int to = map_two_cups(w, n - 2, din,
                                              {2 * a, 2 * a + 1},
                                              {2 * b + 1, 2 * b + 2});
                        assert(to >= 0);
                        int coeff = 1;
                        int sign_exp = b - a;
                        if (b == r) {
                            coeff = r;
                            sign_exp = r - 1 - a;
                        }
                        if (sign_exp & 1) coeff = -coeff;
                        inA[to].push_back({offD + std::uint32_t(i), coeff});
                    }
                }
            }
        }

        // Every A row must be emitted because its old value is multiplied by s.
        emit_rows(stage.phase1, offA, inA, s, true);
    }

    if (d0 && dp) {
        std::vector<std::vector<std::pair<std::uint32_t, int>>> inB(d0);
        std::vector<std::vector<std::pair<std::uint32_t, int>>> inC(d0);
        auto const& BD = basis(n - 2, d + 2);
        int sign_base = (d - 1) / 2;
        for (int i = 0; i < int(BD.words.size()); ++i) {
            auto w = BD.words[i];

            // partial_{d+2}(D) -> B.
            for (int j = 0; j <= (d - 1) / 2; ++j) {
                int to = map_cup_word(w, n - 2, d + 2, 2 * j, 2 * j + 1);
                assert(to >= 0);
                int coeff = ((sign_base - j) & 1) ? -1 : 1;
                inB[to].push_back({offD + std::uint32_t(i), coeff});
            }

            // J_{d+2}(D) -> C.
            int to = map_cup_word(w, n - 2, d + 2, d, d + 1);
            assert(to >= 0);
            inC[to].push_back({offD + std::uint32_t(i), 1});
        }
        emit_rows(stage.phase1, offB, inB, 1, false);
        emit_rows(stage.phase2, offC, inC, 1, false);
    }

    if (da) compile_node(out, n - 2, d - 2, offA, depth + 1);
    if (d0) {
        compile_node(out, n - 2, d, offB, depth + 1);
        compile_node(out, n - 2, d, offC, depth + 1);
    }
    if (dp) compile_node(out, n - 2, d + 2, offD, depth + 1);
}

static CompiledTransform compile_transform(int n, int d) {
    CompiledTransform out;
    out.n = n;
    out.d = d;
    compile_node(out, n, d, 0, 0);
    return out;
}

static void execute_phase(CompiledPhase const& p, std::vector<std::uint32_t>& v) {
    for (auto const& row : p.rows) {
        std::uint32_t acc = mul_small_signed(v[row.dst], row.self_coeff);
        for (std::uint32_t e = 0; e < row.edge_count; ++e) {
            auto const& edge = p.edges[row.edge_begin + e];
            acc = addm(acc, mul_small_signed(v[edge.src], edge.coeff));
        }
        v[row.dst] = acc;
    }
}

static void execute_compiled(
    CompiledTransform const& t,
    std::vector<std::uint32_t>& v
) {
    for (auto const& stage : t.stages) {
        execute_phase(stage.phase1, v);
        // GPU executor inserts __syncthreads() / cluster.sync() here.
        execute_phase(stage.phase2, v);
        // And another barrier here before child nodes at the next depth.
    }
}

static void print_stats(CompiledTransform const& t) {
    std::uint64_t p1rows = 0, p2rows = 0, edges = 0;
    int max_coeff = 0;
    for (auto const& s : t.stages) {
        p1rows += s.phase1.rows.size();
        p2rows += s.phase2.rows.size();
        edges += s.phase1.edges.size() + s.phase2.edges.size();
        for (auto const& e : s.phase1.edges) max_coeff = std::max(max_coeff, std::abs(int(e.coeff)));
        for (auto const& e : s.phase2.edges) max_coeff = std::max(max_coeff, std::abs(int(e.coeff)));
        for (auto const& r : s.phase1.rows) max_coeff = std::max(max_coeff, std::abs(int(r.self_coeff)));
        for (auto const& r : s.phase2.rows) max_coeff = std::max(max_coeff, std::abs(int(r.self_coeff)));
    }
    std::cout << "compile n=" << t.n << " d=" << t.d
              << " dim=" << basis(t.n, t.d).words.size()
              << " stages=" << t.stages.size()
              << " phase1_rows=" << p1rows
              << " phase2_rows=" << p2rows
              << " edges=" << edges
              << " max_coeff=" << max_coeff << "\n";
}

int main(int argc, char** argv) {
    int maxn = argc > 1 ? std::atoi(argv[1]) : 15;
    if ((maxn & 1) == 0) --maxn;
    std::mt19937_64 rng(0x7f4a7c15ULL);

    for (int n = 1; n <= maxn; n += 2) {
        for (int d = 1; d <= n; d += 2) {
            int dimv = int(basis(n, d).words.size());
            if (!dimv) continue;
            auto compiled = compile_transform(n, d);
            for (int trial = 0; trial < 4; ++trial) {
                std::vector<std::uint32_t> a(dimv), b;
                for (auto& x : a) x = std::uint32_t(rng() % MOD);
                b = a;
                transform_integer(n, d, a);
                execute_compiled(compiled, b);
                if (a != b) {
                    std::cerr << "compiled transform mismatch n=" << n
                              << " d=" << d << " trial=" << trial << "\n";
                    for (int i = 0; i < dimv; ++i) if (a[i] != b[i]) {
                        std::cerr << " first i=" << i << " recursive=" << a[i]
                                  << " compiled=" << b[i] << "\n";
                        break;
                    }
                    return 1;
                }
            }
            print_stats(compiled);
        }
    }
    std::cout << "PASS static gather-stage compiler\n";
    return 0;
}
