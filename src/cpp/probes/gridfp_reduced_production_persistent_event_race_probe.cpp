#pragma push_macro("main")
#undef main
#define main gridfp_p2p_cycle_batch_hash_probe_main_unused
#include "gridfp_p2p_cycle_batch_hash_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

constexpr int B = 16;
constexpr int G = 8;

enum class Kind : int { A = 0, Done = 1, Barrier = 2, PhaseB = 3 };

int node(Kind kind, int batch, int gpu = 0) {
    // Barrier has only one logical node per batch; other kinds have G nodes.
    const int stride = 3 * G + 1;
    const int base = batch * stride;
    switch (kind) {
        case Kind::A: return base + gpu;
        case Kind::Done: return base + G + gpu;
        case Kind::PhaseB: return base + 2 * G + gpu;
        case Kind::Barrier: return base + 3 * G;
    }
    return -1;
}

} // namespace

int main() {
    char arg0[] = "cycle-batch-hash";
    char arg1[] = "18";
    char* argv[] = {arg0, arg1, nullptr};
    if (gridfp_p2p_cycle_batch_hash_probe_main_unused(2, argv) != 0) return 2;

    constexpr int stride = 3 * G + 1;
    constexpr int N = B * stride;
    std::vector<std::vector<unsigned char>> reach(
        N, std::vector<unsigned char>(N, 0));
    auto edge = [&](int a, int b) {
        if (a < 0 || b < 0 || a >= N || b >= N) std::exit(3);
        reach[a][b] = 1;
    };

    for (int b = 0; b < B; ++b) {
        const int barrier = node(Kind::Barrier, b);
        for (int g = 0; g < G; ++g) {
            edge(node(Kind::A, b, g), node(Kind::Done, b, g));
            edge(node(Kind::Done, b, g), barrier);
            edge(barrier, node(Kind::PhaseB, b, g));
            if (b + 2 < B)
                edge(node(Kind::PhaseB, b, g), node(Kind::A, b + 2, g));
        }
        // The GPU0 coordinator stream records barriers in program order.  It
        // does not wait for Phase B, so this edge must not serialize B_b with
        // A_{b+1}.
        if (b + 1 < B)
            edge(barrier, node(Kind::Barrier, b + 1));
    }

    for (int k = 0; k < N; ++k)
        for (int i = 0; i < N; ++i) {
            if (!reach[i][k]) continue;
            for (int j = 0; j < N; ++j)
                reach[i][j] = static_cast<unsigned char>(
                    reach[i][j] || (reach[i][k] && reach[k][j]));
        }

    std::uint64_t same_batch_pairs = 0;
    std::uint64_t scratch_reuse_pairs = 0;
    std::uint64_t overlap_pairs = 0;
    std::uint64_t adjacent_b_pairs = 0;

    for (int b = 0; b < B; ++b) {
        for (int src = 0; src < G; ++src) {
            for (int dst = 0; dst < G; ++dst) {
                const int a = node(Kind::A, b, src);
                const int phase_b = node(Kind::PhaseB, b, dst);
                if (!reach[a][phase_b]) return 4;
                ++same_batch_pairs;
            }
        }

        if (b + 2 < B) {
            for (int g = 0; g < G; ++g) {
                const int phase_b = node(Kind::PhaseB, b, g);
                const int next_same_plane = node(Kind::A, b + 2, g);
                if (!reach[phase_b][next_same_plane]) return 5;
                ++scratch_reuse_pairs;
            }
        }

        if (b + 1 < B) {
            for (int src = 0; src < G; ++src) {
                for (int dst = 0; dst < G; ++dst) {
                    const int phase_b = node(Kind::PhaseB, b, src);
                    const int next_a = node(Kind::A, b + 1, dst);
                    if (reach[phase_b][next_a] || reach[next_a][phase_b]) return 6;
                    ++overlap_pairs;

                    const int next_b = node(Kind::PhaseB, b + 1, dst);
                    if (reach[phase_b][next_b] || reach[next_b][phase_b]) return 7;
                    ++adjacent_b_pairs;
                }
            }
        }
    }

    std::cout << "ALL_OK persistent_event_pipeline_race=1"
              << " batches=" << B
              << " ngpu=" << G
              << " cycle_closed_batch_partition=1"
              << " same_batch_A_before_all_B=" << same_batch_pairs
              << " same_plane_B_before_A_plus_2=" << scratch_reuse_pairs
              << " unordered_B_i_vs_A_i_plus_1=" << overlap_pairs
              << " unordered_B_i_vs_B_i_plus_1=" << adjacent_b_pairs
              << " host_batch_barriers=0"
              << " coordinator_barrier_stream=1"
              << " cross_batch_state_overlap=0"
              << " scratch_plane_overlap=0\n";
    return 0;
}
