#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_sectorcache_microprobe_main_unused
#include "two_cell_fusion2_sectorcache_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_primitive_reflection.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

struct ReversePrimitiveLut {
    std::vector<std::uint32_t> label_left;       // occupied <=25
    std::vector<std::uint32_t> reflection_meta;  // occupied <=27
    Rank offset[oneesan::twocell::kMaxWidth + 1]{};
};

ReversePrimitiveLut build_reverse_primitive_lut(
    int W,
    const RankTables& rt
) {
    ReversePrimitiveLut lut;
    for (int occupied = 1; occupied <= W - 1; occupied += 2) {
        lut.offset[occupied] = static_cast<Rank>(lut.reflection_meta.size());
        const Rank pc = rt.primitive[occupied][1];
        const std::uint32_t support = oneesan::twocell::low_mask(occupied);
        for (Rank r = 0; r < pc; ++r) {
            const std::uint32_t left = oneesan::twocell::primitive_left_unrank(
                support, occupied, occupied, r, rt);
            if (occupied <= W - 2) lut.label_left.push_back(left);
            lut.reflection_meta.push_back(
                oneesan::twocell::make_primitive_reflection_meta(
                    left, occupied, rt));
        }
    }
    return lut;
}

__device__ __forceinline__ std::uint32_t reverse_deposit_left_warp(
    std::uint32_t support,
    std::uint32_t compact_left,
    int len
) {
    const int lane = threadIdx.x & 31;
    const bool occupied = lane < len && ((support >> lane) & 1u);
    const int ordinal = __popc(support & oneesan::twocell::low_mask(lane));
    const bool is_left = occupied && ((compact_left >> ordinal) & 1u);
    return __ballot_sync(0xffffffffu, is_left) & oneesan::twocell::low_mask(len);
}

__global__ void two_cell_reverse_fusion2_register_sector_kernel(
    std::uint32_t* __restrict__ values,
    const std::uint32_t* __restrict__ label_left_lut,
    const std::uint32_t* __restrict__ reflection_lut,
    const Rank* __restrict__ primitive_offset,
    int W,
    int start,
    int outer_ones,
    Rank support_count,
    Rank state_count,
    std::uint32_t mod,
    int* error
) {
    extern __shared__ std::uint32_t block_values[];
    __shared__ PackedKey sh_forward[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ PackedWord sh_forward_label[FUSION_WARPS];
    __shared__ Rank sh_forward_label_primitive[FUSION_WARPS];
    __shared__ oneesan::twocell::FusionSector sh_sector[FUSION2_SECTORS];
    __shared__ int sh_ns[FUSION_WARPS];
    __shared__ int sh_deep[FUSION_WARPS];
    __shared__ int sh_partner_rounds[FUSION_WARPS];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (warp >= FUSION_WARPS) return;

    const int outer_bits = W - FUSION_STEPS - 3;
    const Rank n = oneesan::twocell::fusion_block_size(
        FUSION_STEPS, outer_ones, TC_RANK_TABLES);

    for (Rank support_rank = blockIdx.x; support_rank < support_count;
         support_rank += gridDim.x) {
        const std::uint32_t outer = oneesan::twocell::support_unrank(
            outer_bits, outer_ones, support_rank, TC_RANK_TABLES);

        if (threadIdx.x < FUSION2_SECTORS) {
            sh_sector[threadIdx.x] = oneesan::twocell::fusion_sector(
                threadIdx.x, outer, W, start, FUSION_STEPS,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
        }
        __syncthreads();

        for (int q = 0; q < FUSION2_SECTORS; ++q) {
            const auto sec = sh_sector[q];
            if (!sec.valid) continue;
            for (Rank p = threadIdx.x; p < sec.count; p += blockDim.x)
                block_values[sec.local_base + p] = values[sec.global_base + p];
        }
        __syncthreads();

        for (int phase = 0; phase < FUSION_STEPS; ++phase) {
            const int reverse_active = start + FUSION_STEPS - phase;
            const int forward_i = W - start - FUSION_STEPS - 3 + phase;

            for (std::uint32_t code = 0; code < 8; ++code) {
                const int occupied = outer_ones + oneesan::twocell::popcount32(code);
                const Rank pc = oneesan::twocell::primitive_count_for_occupied(
                    occupied, TC_RANK_TABLES);
                if (!pc) continue;
                const std::uint32_t label_support =
                    oneesan::twocell::insert_support_window(
                        outer, start, FUSION_STEPS + 1, code);

                for (Rank primitive = warp; primitive < pc;
                     primitive += FUSION_WARPS) {
                    std::uint32_t compact_left = 0;
                    std::uint32_t label_ref_meta = 0;
                    if (lane == 0) {
                        compact_left = label_left_lut[
                            primitive_offset[occupied] + primitive];
                        label_ref_meta = reflection_lut[
                            primitive_offset[occupied] + primitive];
                    }
                    compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
                    label_ref_meta = __shfl_sync(
                        0xffffffffu, label_ref_meta, 0);
                    const std::uint32_t left = reverse_deposit_left_warp(
                        label_support, compact_left, W - 2);

                    if (lane == 0) {
                        const PackedWord reverse_label{
                            label_support, left, static_cast<std::uint8_t>(W - 2)};
                        sh_forward_label[warp] =
                            oneesan::twocell::reflect_word_with_reflection_meta(
                                reverse_label, label_ref_meta);
                        sh_forward_label_primitive[warp] =
                            oneesan::twocell::primitive_reflection_mirror_rank(
                                label_ref_meta);
                        sh_ns[warp] = 0;
                        sh_partner_rounds[warp] = 0;
                        PackedWord collapsed{};
                        sh_deep[warp] = oneesan::twocell::deep_collapse(
                            sh_forward_label[warp], forward_i, collapsed) ? 1 : 0;
                    }
                    __syncwarp();

                    if (sh_deep[warp]) {
                        oneesan::twocell::cuda_face::deep_component_sources_compact(
                            sh_forward_label[warp], W, forward_i,
                            sh_forward[warp], &sh_ns[warp],
                            &sh_partner_rounds[warp], error);
                        __syncwarp();
                    } else if (lane == 0) {
                        const auto src = oneesan::twocell::direct_component_sources(
                            sh_forward_label[warp], W, forward_i);
                        if (src.overflow || src.size <= 0 ||
                            src.size > FUSION_MAX_COMPONENT) {
                            set_error(error, 571);
                        } else {
                            sh_ns[warp] = src.size;
                            for (int s = 0; s < src.size; ++s)
                                sh_forward[warp][s] = src.value[s];
                        }
                    }
                    __syncwarp();

                    const int ns = sh_ns[warp];
                    if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                        if (lane == 0) set_error(error, 572);
                        __syncwarp();
                        continue;
                    }

                    PackedKey forward_source{};
                    PackedKey reverse_source{};
                    Rank reverse_primitive = 0;
                    int sector = -1;
                    std::uint32_t x = 0;
                    if (lane < ns) {
                        forward_source = sh_forward[warp][lane];
                        const PackedWord flabel = sh_forward_label[warp];
                        const bool retained = oneesan::twocell::symbol(
                            flabel, forward_i) != oneesan::twocell::TC_N;
                        const bool xN_deep = retained &&
                            oneesan::twocell::symbol(flabel, forward_i + 1) ==
                            oneesan::twocell::TC_N;
                        const bool reuse_forward = retained &&
                            (lane < 3 || (lane == 3 && xN_deep));

                        Rank forward_primitive = sh_forward_label_primitive[warp];
                        if (!reuse_forward) {
                            const int len = forward_source.type ? W - 2 : W - 1;
                            forward_primitive = oneesan::twocell::primitive_rank(
                                forward_source.support, forward_source.left,
                                len, TC_RANK_TABLES);
                        }
                        const int source_len = forward_source.type ? W - 2 : W - 1;
                        const int source_occupied = oneesan::twocell::popcount32(
                            forward_source.support &
                            oneesan::twocell::low_mask(source_len));
                        const std::uint32_t source_meta = reflection_lut[
                            primitive_offset[source_occupied] + forward_primitive];
                        reverse_primitive =
                            oneesan::twocell::primitive_reflection_mirror_rank(
                                source_meta);
                        reverse_source =
                            oneesan::twocell::reflect_key_with_reflection_meta(
                                forward_source, W, source_meta);

                        sector = oneesan::twocell::fusion_sector_index_at(
                            reverse_source, start, FUSION_STEPS, reverse_active);
                        if (sector < 0 || sector >= FUSION2_SECTORS ||
                            !sh_sector[sector].valid) {
                            set_error(error, 573);
                        } else if (reverse_primitive >= sh_sector[sector].count) {
                            set_error(error, 574);
                        } else {
                            x = block_values[
                                sh_sector[sector].local_base + reverse_primitive];
                        }
                    }
                    __syncwarp();

                    const std::uint32_t y =
                        oneesan::twocell::cuda_component::apply_closed_component_warp(
                            sh_forward_label[warp], forward_source,
                            ns, W, forward_i, x, mod, error);
                    __syncwarp();
                    if (lane < ns && sector >= 0 && sector < FUSION2_SECTORS)
                        block_values[
                            sh_sector[sector].local_base + reverse_primitive] = y;
                    __syncwarp();
                }
            }
            __syncthreads();
        }

        for (int q = 0; q < FUSION2_SECTORS; ++q) {
            const auto sec = sh_sector[q];
            if (!sec.valid) continue;
            for (Rank p = threadIdx.x; p < sec.count; p += blockDim.x)
                values[sec.global_base + p] = block_values[sec.local_base + p];
        }
        __syncthreads();
    }
}

std::vector<std::uint32_t> reverse_fusion2_reference(
    const std::vector<std::uint32_t>& input,
    int W,
    int start,
    const RankTables& rt,
    const StationaryRankTables& st,
    std::uint32_t mod
) {
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
    const int source_active = start + 2;
    const int middle_active = start + 1;
    const int final_active = start;
    const int first_pair = source_active + 1;
    const int second_pair = middle_active + 1;

    std::vector<std::uint32_t> mid(input.size()), out(input.size());
    for (const Key& s : reverse_q_basis(W, first_pair, words)) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, source_active, rt, st);
        const std::uint32_t x = input[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : K_reverse_basis(s, W, first_pair)) {
            if (c != 1) std::exit(575);
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, middle_active, rt, st);
            mid[static_cast<std::size_t>(dr)] = add_ref_mod(
                mid[static_cast<std::size_t>(dr)], x, mod);
        }
    }
    for (const Key& s : reverse_q_basis(W, second_pair, words)) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, middle_active, rt, st);
        const std::uint32_t x = mid[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : K_reverse_basis(s, W, second_pair)) {
            if (c != 1) std::exit(576);
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, final_active, rt, st);
            out[static_cast<std::size_t>(dr)] = add_ref_mod(
                out[static_cast<std::size_t>(dr)], x, mod);
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth || mod < 3) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const ReversePrimitiveLut host_lut = build_reverse_primitive_lut(W, rt);

    if (plan_only) {
        std::cout << "reverse-fusion2-register-sector-plan"
                  << " W=" << W
                  << " label_left_LUT_MiB="
                  << double(host_lut.label_left.size() * sizeof(std::uint32_t)) /
                         double(1ULL << 20)
                  << " reflection_LUT_MiB="
                  << double(host_lut.reflection_meta.size() * sizeof(std::uint32_t)) /
                         double(1ULL << 20)
                  << " reflection_root_scan=0"
                  << " reflected_primitive_rescan=0"
                  << " capacity_equals_forward=1\n";
        return 0;
    }
    if (W > 10) return 3;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "reverse fusion2 device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "reverse fusion2 set device");
    install_tables(rt);
    install_stationary_tables(st);

    std::uint32_t *d_left = nullptr, *d_reflect = nullptr;
    Rank* d_offset = nullptr;
    ck(cudaMalloc(&d_left, host_lut.label_left.size() * sizeof(std::uint32_t)),
       "reverse fusion2 alloc left LUT");
    ck(cudaMalloc(&d_reflect,
                  host_lut.reflection_meta.size() * sizeof(std::uint32_t)),
       "reverse fusion2 alloc reflection LUT");
    ck(cudaMalloc(&d_offset, sizeof(host_lut.offset)),
       "reverse fusion2 alloc offsets");
    ck(cudaMemcpy(d_left, host_lut.label_left.data(),
                  host_lut.label_left.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "reverse fusion2 copy left LUT");
    ck(cudaMemcpy(d_reflect, host_lut.reflection_meta.data(),
                  host_lut.reflection_meta.size() * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "reverse fusion2 copy reflection LUT");
    ck(cudaMemcpy(d_offset, host_lut.offset, sizeof(host_lut.offset),
                  cudaMemcpyHostToDevice), "reverse fusion2 copy offsets");

    cudaFuncAttributes attr{};
    ck(cudaFuncGetAttributes(&attr, two_cell_reverse_fusion2_register_sector_kernel),
       "reverse fusion2 attributes");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "reverse fusion2 props");
    const Rank limit = std::min<Rank>(
        shared_kib * 1024ULL, static_cast<Rank>(prop.sharedMemPerBlockOptin));
    const Rank static_shared = static_cast<Rank>(attr.sharedSizeBytes);
    const Rank states = st.total[W];
    const int outer_bits = W - 5;

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + 701ULL) % (mod - 1ULL)));

    for (int start = 0; start <= W - 5; ++start) {
        const auto reference = reverse_fusion2_reference(
            input, W, start, rt, st, mod);
        std::uint32_t* d_values = nullptr;
        int* d_error = nullptr;
        ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)),
           "reverse fusion2 alloc values");
        ck(cudaMalloc(&d_error, sizeof(int)), "reverse fusion2 alloc error");
        ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "reverse fusion2 copy input");
        ck(cudaMemset(d_error, 0, sizeof(int)), "reverse fusion2 zero error");

        for (int o = 0; o <= outer_bits; ++o) {
            const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
            const Rank dynamic_bytes = n * sizeof(std::uint32_t);
            if (static_shared + dynamic_bytes > limit) {
                std::cerr << "reverse fusion2 small-width bucket did not fit\n";
                return 5;
            }
            ck(cudaFuncSetAttribute(
                   two_cell_reverse_fusion2_register_sector_kernel,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(dynamic_bytes)),
               "reverse fusion2 optin shared");
            const Rank support_count = rt.choose[outer_bits][o];
            const unsigned grid = static_cast<unsigned>(
                std::max<Rank>(1, std::min<Rank>(support_count, 65535)));
            two_cell_reverse_fusion2_register_sector_kernel
                <<<grid, FUSION_THREADS, dynamic_bytes>>>(
                    d_values, d_left, d_reflect, d_offset,
                    W, start, o, support_count, states, mod, d_error);
            ck(cudaGetLastError(), "reverse fusion2 launch");
        }
        ck(cudaDeviceSynchronize(), "reverse fusion2 sync");

        std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
        int error = 0;
        ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost), "reverse fusion2 copy output");
        ck(cudaMemcpy(&error, d_error, sizeof(error),
                      cudaMemcpyDeviceToHost), "reverse fusion2 copy error");
        if (error || output != reference) {
            std::cerr << "FAIL reverse fusion2 arithmetic"
                      << " W=" << W << " start=" << start
                      << " error=" << error << '\n';
            return 6;
        }
        std::cout << "reverse-fusion2 arithmetic=OK"
                  << " W=" << W << " start=" << start << '\n';
        cudaFree(d_values);
        cudaFree(d_error);
    }

    cudaFree(d_left);
    cudaFree(d_reflect);
    cudaFree(d_offset);
    return 0;
}
