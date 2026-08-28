#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_shift_cycle_microprobe_main_unused
#include "gridfp_reduced_production_shift_cycle_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_grouped_support_device.cuh"
#include "gridfp_reduced_production_owner_support_device.cuh"

#include <vector>

namespace {

__device__ int owner_support_seed_slot_device(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    bool reverse
) {
    std::uint32_t compact = 0;
    int cp = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
        ++cp;
    }
    const bool odd = (__popc(compact) & 1) != 0;
    const bool a = ((support >> (q - 1)) & 1u) != 0;
    const bool b = ((support >> q) & 1u) != 0;
    int ri = -1;
    if (odd) {
        if (blocked) return -1;
        if (!a && !b) ri = 0;
        else if (a && b) ri = 1;
    } else {
        const bool fixed = reverse ? b : a;
        const bool missing = reverse ? a : b;
        if (fixed && !missing) ri = blocked ? 1 : 0;
        else if (!fixed && missing && !blocked) ri = 2;
    }
    if (ri < 0) return -1;
    return static_cast<int>(compact * 3u + static_cast<std::uint32_t>(ri));
}

__global__ void owner_support_counts_kernel(
    int W,
    int K,
    int ngpu,
    unsigned long long* counts
) {
    const int owner = threadIdx.x;
    if (owner >= ngpu) return;
    counts[owner] = owner_support_slab_count_device(W, K, owner, ngpu);
}

__global__ void owner_support_mark_kernel(
    Rank64 count,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int K,
    int owner,
    int ngpu,
    unsigned int* seen,
    unsigned long long* marked,
    int* error
) {
    unsigned long long local_marked = 0;
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    for (Rank64 rank = first; rank < count; rank += stride) {
        const OwnerSupportSlabDevice slab = owner_support_slab_unrank_device(
            W, q, reverse, tile_start, K, owner, ngpu, rank);
        if (!slab.valid) {
            set_error(error, 301);
            continue;
        }
        const int actual_owner = grouped_support_owner_device(
            slab.support, W, reverse, tile_start, K, ngpu);
        if (actual_owner != owner) {
            set_error(error, 302);
            continue;
        }
        const int slot = owner_support_seed_slot_device(
            slab.support, slab.blocked != 0, W, q, reverse);
        if (slot < 0) {
            set_error(error, 303);
            continue;
        }
        if (atomicCAS(seen + slot, 0u, static_cast<unsigned int>(owner + 1)) != 0u) {
            set_error(error, 304);
            continue;
        }
        ++local_marked;
    }
    if (local_marked) atomicAdd(marked, local_marked);
}

__global__ void owner_support_coverage_kernel(
    Rank64 base_supports,
    int W,
    int q,
    bool reverse,
    const unsigned int* seen,
    unsigned long long* checked,
    int* error
) {
    unsigned long long local_checked = 0;
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    for (Rank64 base = first; base < base_supports; base += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const Rank64 slot = base * 3 + static_cast<Rank64>(ri);
            if (seen[slot] == 0u) set_error(error, 305);
            ++local_checked;
        }
        if (nr == 2 && seen[base * 3 + 2] != 0u) set_error(error, 306);
    }
    if (local_checked) atomicAdd(checked, local_checked);
}

void run_owner_support_proof(
    int W,
    int K,
    int S,
    bool reverse,
    int ngpu,
    unsigned requested_blocks
) {
    ProductionFactorTables tables(W);
    ck(cudaSetDevice(0), "owner support set device");
    install_tables(tables);

    const int tile_start = reverse ? 1 : W - 1;
    const int q = tile_start + (reverse ? S : -S);
    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 seen_slots = base_supports * 3;
    const unsigned long long expected =
        5ULL * (1ULL << (W - 3));

    unsigned long long* d_counts = nullptr;
    unsigned int* d_seen = nullptr;
    unsigned long long* d_marked = nullptr;
    unsigned long long* d_checked = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_counts, ngpu * sizeof(unsigned long long)),
       "owner support alloc counts");
    ck(cudaMalloc(&d_seen, seen_slots * sizeof(unsigned int)),
       "owner support alloc seen");
    ck(cudaMalloc(&d_marked, sizeof(unsigned long long)),
       "owner support alloc marked");
    ck(cudaMalloc(&d_checked, sizeof(unsigned long long)),
       "owner support alloc checked");
    ck(cudaMalloc(&d_error, sizeof(int)), "owner support alloc error");
    ck(cudaMemset(d_seen, 0, seen_slots * sizeof(unsigned int)),
       "owner support zero seen");
    ck(cudaMemset(d_marked, 0, sizeof(unsigned long long)),
       "owner support zero marked");
    ck(cudaMemset(d_checked, 0, sizeof(unsigned long long)),
       "owner support zero checked");
    ck(cudaMemset(d_error, 0, sizeof(int)), "owner support zero error");

    owner_support_counts_kernel<<<1, ngpu>>>(W, K, ngpu, d_counts);
    ck(cudaGetLastError(), "owner support counts launch");
    ck(cudaDeviceSynchronize(), "owner support counts sync");
    std::vector<unsigned long long> counts(static_cast<std::size_t>(ngpu));
    ck(cudaMemcpy(counts.data(), d_counts, ngpu * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "owner support copy counts");
    unsigned long long sum_counts = 0;
    for (auto x : counts) sum_counts += x;
    if (sum_counts != expected) fail("owner support count total");

    for (int owner = 0; owner < ngpu; ++owner) {
        const Rank64 count = counts[static_cast<std::size_t>(owner)];
        if (!count) continue;
        const Rank64 needed = (count + Rank64(THREADS) - 1) / Rank64(THREADS);
        const unsigned blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed)));
        owner_support_mark_kernel<<<blocks, THREADS>>>(
            count, W, q, reverse, tile_start, K, owner, ngpu,
            d_seen, d_marked, d_error);
        ck(cudaGetLastError(), "owner support mark launch");
    }
    ck(cudaDeviceSynchronize(), "owner support mark sync");

    const Rank64 needed =
        (base_supports + Rank64(THREADS) - 1) / Rank64(THREADS);
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed)));
    owner_support_coverage_kernel<<<blocks, THREADS>>>(
        base_supports, W, q, reverse, d_seen, d_checked, d_error);
    ck(cudaGetLastError(), "owner support coverage launch");
    ck(cudaDeviceSynchronize(), "owner support coverage sync");

    int error = 0;
    unsigned long long marked = 0, checked = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "owner support copy error");
    ck(cudaMemcpy(&marked, d_marked, sizeof(marked), cudaMemcpyDeviceToHost),
       "owner support copy marked");
    ck(cudaMemcpy(&checked, d_checked, sizeof(checked), cudaMemcpyDeviceToHost),
       "owner support copy checked");
    if (error) fail("owner support proof device error=" + std::to_string(error));
    if (marked != expected || checked != expected)
        fail("owner support proof cardinality");

    unsigned long long min_owner = counts[0], max_owner = counts[0];
    for (auto x : counts) {
        min_owner = std::min(min_owner, x);
        max_owner = std::max(max_owner, x);
    }
    std::cout << "gridfp-owner-support-proof"
              << " W=" << W
              << " K=" << K
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " support_slabs=" << expected
              << " min_owner_slabs=" << min_owner
              << " max_owner_slabs=" << max_owner
              << " owner_imbalance="
              << (min_owner ? double(max_owner) / double(min_owner) : 0.0)
              << " exact_coverage=1 duplicates=0 owner_exact=1"
              << " primitive_rank_calls=0 exact=OK\n";

    cudaFree(d_error);
    cudaFree(d_checked);
    cudaFree(d_marked);
    cudaFree(d_seen);
    cudaFree(d_counts);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 14;
    const int K = argc > 2 ? std::atoi(argv[2]) : (W - 2) / 2;
    const unsigned blocks = argc > 3
        ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    if (W < 6 || W > 16 || K < 1 || K + 3 > W || !blocks ||
        ngpu < 2 || ngpu > 8) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "owner support device count");
    if (visible < 1) return 3;

    int cases = 0;
    for (int S = 1; S <= K && K + S + 2 <= W; ++S) {
        run_owner_support_proof(W, K, S, false, ngpu, blocks);
        run_owner_support_proof(W, K, S, true, ngpu, blocks);
        cases += 2;
    }
    if (!cases) return 4;
    std::cout << "ALL_OK gridfp_owner_support_codec=1 cases=" << cases << '\n';
    return 0;
}
