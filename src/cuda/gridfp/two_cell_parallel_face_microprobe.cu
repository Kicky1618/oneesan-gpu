#include "../../common/two_cell_component_device.cuh"
#include "../../common/two_cell_recoupling_rank.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace {

using namespace oneesan::twocell;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = WARPS_PER_BLOCK * 32;
constexpr int MAX_COMPONENT = 18;
constexpr int MAX_CANDIDATES = 34;

__constant__ RankTables TC_FACE_TABLES;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(200);
    }
}

__device__ __forceinline__ int prefix_height_bits(PackedWord w, int boundary) {
    const std::uint32_t mask = low_mask(boundary);
    return 1 + 2 * __popc(w.left & mask) - __popc(w.support & mask);
}

__device__ __forceinline__ int partner_first_return_warp_lane(
    PackedWord w,
    int p,
    int* rounds
) {
    const int level = prefix_height_bits(w, p);
    int n = 0;
    for (int q = p + 1; q < w.len; ++q) {
        ++n;
        if (symbol(w, q) == TC_R && prefix_height_bits(w, q + 1) == level) {
            *rounds = n;
            return q;
        }
    }
    *rounds = n;
    return -1;
}

__device__ __forceinline__ bool key_in_list(
    const PackedKey* xs,
    int n,
    PackedKey x
) {
    for (int q = 0; q < n; ++q)
        if (equal(xs[q], x)) return true;
    return false;
}

// All lanes in one warp call this function. It is specialized to a deep
// component and writes the same ordered source convention as the serial direct
// helper: three base coordinates, sparse central predecessor, then exposed-face
// strand cuts. Expensive height/partner work is position-parallel.
__device__ __forceinline__ void warp_deep_component_sources(
    PackedWord label,
    int W,
    int i,
    PackedKey* out,
    int* out_size,
    PackedKey* candidate,
    std::uint8_t* candidate_valid,
    int* max_partner_rounds,
    int* error
) {
    const int lane = threadIdx.x & 31;
    PackedWord collapsed{};
    if (!deep_collapse(label, i, collapsed)) {
        if (lane == 0) atomicCAS(error, 0, 201);
        return;
    }

    if (lane == 0) {
        out[0] = make_state(1, label);
        out[1] = make_state(0, insert_symbol(label, i, TC_N));
        out[2] = make_state(0, insert_symbol(label, i + 1, TC_N));
        *out_size = 3;
        *max_partner_rounds = 0;
        for (int q = 0; q < MAX_CANDIDATES; ++q) candidate_valid[q] = 0;
    }
    __syncwarp();

    PackedWord central = insert_symbol(collapsed, i, TC_N);
    central = insert_symbol(central, i, TC_N);
    const int j = i + 1;
    PackedWord z = insert_symbol(central, j + 1, TC_N);

    // The no-cut predecessor is always the fourth source coordinate.
    if (lane == 0) {
        PackedKey sparse{};
        if (!inverse_E(z, i, sparse) || !in_source_layout(sparse, W, i)) {
            atomicCAS(error, 0, 202);
        } else {
            out[(*out_size)++] = sparse;
        }
    }

    const int boundary = lane <= W ? lane : W;
    const int h = lane <= W ? prefix_height_bits(z, boundary) : 0x3fffffff;
    const int level = prefix_height_bits(z, j);

    const unsigned left_bad = __ballot_sync(
        0xffffffffu, lane < j && h < level);
    const unsigned right_bad = __ballot_sync(
        0xffffffffu, lane >= j + 3 && lane <= W && h < level);
    const int face_left = left_bad
        ? (31 - __clz(left_bad)) + 1
        : 0;
    const int face_right = right_bad
        ? (__ffs(static_cast<int>(right_bad)) - 1) - 1
        : W;

    PackedKey mine{};
    bool valid = false;
    int partner_rounds = 0;
    if (lane < W && symbol(z, lane) == TC_L &&
        lane >= face_left && prefix_height_bits(z, lane) == level) {
        const int q = partner_first_return_warp_lane(z, lane, &partner_rounds);
        if (q >= 0 && q < face_right) {
            PackedWord w = z;
            bool full_valid = true;
            if (q < j) {
                w = set_symbol(w, q, TC_L);
                w = set_symbol(w, j, TC_R);
                w = set_symbol(w, j + 1, TC_R);
            } else if (lane > j + 1) {
                w = set_symbol(w, lane, TC_R);
                w = set_symbol(w, j, TC_L);
                w = set_symbol(w, j + 1, TC_L);
            } else if (lane < j && q > j + 1) {
                w = set_symbol(w, j, TC_R);
                w = set_symbol(w, j + 1, TC_L);
            } else {
                full_valid = false;
            }
            if (full_valid && valid_word(w) && inverse_E(w, i, mine) &&
                in_source_layout(mine, W, i))
                valid = true;
        }
    }

    candidate[lane] = mine;
    candidate_valid[lane] = static_cast<std::uint8_t>(valid);

    // The enclosing strand immediately outside the marked face is a separate
    // inverse-R case. Give it candidate slot 32.
    if (lane == 0) {
        PackedKey extra{};
        bool extra_valid = false;
        if (face_left > 0) {
            const int p = face_left - 1;
            if (symbol(z, p) == TC_L) {
                int rounds = 0;
                const int q = partner_first_return_warp_lane(z, p, &rounds);
                *max_partner_rounds = max(*max_partner_rounds, rounds);
                if (q == face_right && p < j && q > j + 1) {
                    PackedWord w = set_symbol(set_symbol(z, j, TC_R), j + 1, TC_L);
                    if (valid_word(w) && inverse_E(w, i, extra) &&
                        in_source_layout(extra, W, i))
                        extra_valid = true;
                }
            }
        }
        candidate[32] = extra;
        candidate_valid[32] = static_cast<std::uint8_t>(extra_valid);
    }

    // Distinguished root case. Find the unique R that reaches height zero with
    // one ballot; lane zero emits candidate slot 33.
    const unsigned root_mask = __ballot_sync(
        0xffffffffu,
        lane < W && symbol(z, lane) == TC_R &&
            prefix_height_bits(z, lane + 1) == 0);
    if (lane == 0) {
        const int root = root_mask ? (__ffs(static_cast<int>(root_mask)) - 1) : -1;
        PackedKey extra{};
        bool extra_valid = false;
        if (root >= 0 &&
            (level == 0 || (face_left == 0 && face_right < W && root == face_right))) {
            PackedWord w = z;
            if (root < j) {
                w = set_symbol(w, root, TC_L);
                w = set_symbol(w, j, TC_R);
                w = set_symbol(w, j + 1, TC_R);
                if (valid_word(w) && inverse_E(w, i, extra) &&
                    in_source_layout(extra, W, i))
                    extra_valid = true;
            } else if (root > j + 1) {
                w = set_symbol(set_symbol(w, j, TC_R), j + 1, TC_L);
                if (valid_word(w) && inverse_E(w, i, extra) &&
                    in_source_layout(extra, W, i))
                    extra_valid = true;
            }
        }
        candidate[33] = extra;
        candidate_valid[33] = static_cast<std::uint8_t>(extra_valid);
    }

    const int warp_max_rounds = __reduce_max_sync(0xffffffffu, partner_rounds);
    if (lane == 0) {
        *max_partner_rounds = max(*max_partner_rounds, warp_max_rounds);
    }
    __syncwarp();

    // Candidate compaction is tiny (<=34 entries) and contains no partner or
    // height scan. Keeping it scalar avoids complicated multiword key ballots.
    if (lane == 0) {
        int n = *out_size;
        for (int q = 0; q < MAX_CANDIDATES; ++q) {
            if (!candidate_valid[q]) continue;
            const PackedKey k = candidate[q];
            if (key_in_list(out, n, k)) continue;
            if (n >= MAX_COMPONENT) {
                atomicCAS(error, 0, 203);
                break;
            }
            out[n++] = k;
        }
        *out_size = n;
    }
    __syncwarp();
}

__global__ void parallel_face_selfcheck_kernel(
    int W,
    Rank components,
    unsigned long long* deep_checked,
    unsigned long long* serial_preimages,
    unsigned long long* parallel_candidates,
    int* worst_partner_rounds,
    int* error
) {
    __shared__ PackedKey sh_out[WARPS_PER_BLOCK][MAX_COMPONENT];
    __shared__ PackedKey sh_candidate[WARPS_PER_BLOCK][MAX_CANDIDATES];
    __shared__ std::uint8_t sh_valid[WARPS_PER_BLOCK][MAX_CANDIDATES];
    __shared__ int sh_size[WARPS_PER_BLOCK];
    __shared__ int sh_rounds[WARPS_PER_BLOCK];
    __shared__ PackedWord sh_label[WARPS_PER_BLOCK];
    __shared__ int sh_deep[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank first = Rank(blockIdx.x) * WARPS_PER_BLOCK + Rank(warp);
    const Rank stride = Rank(gridDim.x) * WARPS_PER_BLOCK;

    for (Rank component = first; component < components; component += stride) {
        if (lane == 0) {
            const PackedKey label_key = component_label_unrank(W, component, TC_FACE_TABLES);
            sh_label[warp] = PackedWord{
                label_key.support, label_key.left, static_cast<std::uint8_t>(W - 2)};
            PackedWord collapsed{};
            sh_deep[warp] = deep_collapse(sh_label[warp], 0, collapsed) ? 1 : 0;
        }
        __syncwarp();

        // Sweep every interior position; component label itself is reused.
        for (int i = 0; i <= W - 4; ++i) {
            if (lane == 0) {
                PackedWord collapsed{};
                sh_deep[warp] = deep_collapse(sh_label[warp], i, collapsed) ? 1 : 0;
            }
            __syncwarp();
            if (!sh_deep[warp]) continue;

            warp_deep_component_sources(
                sh_label[warp], W, i,
                sh_out[warp], &sh_size[warp],
                sh_candidate[warp], sh_valid[warp],
                &sh_rounds[warp], error);
            __syncwarp();

            if (lane == 0) {
                const auto serial = direct_component_sources(sh_label[warp], W, i);
                if (serial.overflow || serial.size != sh_size[warp]) {
                    atomicCAS(error, 0, 204);
                } else {
                    for (int a = 0; a < serial.size; ++a) {
                        bool found = false;
                        for (int b = 0; b < sh_size[warp]; ++b)
                            found |= equal(serial.value[a], sh_out[warp][b]);
                        if (!found) atomicCAS(error, 0, 205);
                    }
                }
                atomicAdd(deep_checked, 1ULL);
                atomicAdd(serial_preimages,
                          static_cast<unsigned long long>(serial.size - 3));
                int candidates = 0;
                for (int q = 0; q < MAX_CANDIDATES; ++q)
                    candidates += sh_valid[warp][q] != 0;
                atomicAdd(parallel_candidates,
                          static_cast<unsigned long long>(candidates));
                atomicMax(worst_partner_rounds, sh_rounds[warp]);
            }
            __syncwarp();
        }
    }
}

bool has_arg(int argc, char** argv, const char* needle) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == needle) return true;
    return false;
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const unsigned blocks = argc > 2
        ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > kMaxWidth || blocks == 0) return 2;

    const RankTables tables = make_rank_tables();
    const Rank components = component_label_count(W, tables);
    if (plan_only) {
        std::cout << "two-cell-parallel-face-microprobe-plan"
                  << " W=" << W
                  << " components=" << components
                  << " lanes_per_position=1"
                  << " prefix_height_popc=2"
                  << " face_boundary_ballots=2"
                  << " root_ballots=1"
                  << " partner_rounds_max=" << (W - 1)
                  << " candidate_slots=" << MAX_CANDIDATES
                  << " output_slots=" << MAX_COMPONENT
                  << " serial_height_array=0 serial_partner_outer_loop=0"
                  << "\n";
        return 0;
    }

    if (W > 11) {
        std::cerr << "execution mode is intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "face device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "face set device");
    ck(cudaMemcpyToSymbol(TC_FACE_TABLES, &tables, sizeof(tables)), "face copy tables");

    unsigned long long* d_deep = nullptr;
    unsigned long long* d_serial = nullptr;
    unsigned long long* d_candidates = nullptr;
    int* d_rounds = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_deep, sizeof(unsigned long long)), "face alloc deep");
    ck(cudaMalloc(&d_serial, sizeof(unsigned long long)), "face alloc serial");
    ck(cudaMalloc(&d_candidates, sizeof(unsigned long long)), "face alloc candidates");
    ck(cudaMalloc(&d_rounds, sizeof(int)), "face alloc rounds");
    ck(cudaMalloc(&d_error, sizeof(int)), "face alloc error");
    ck(cudaMemset(d_deep, 0, sizeof(unsigned long long)), "face zero deep");
    ck(cudaMemset(d_serial, 0, sizeof(unsigned long long)), "face zero serial");
    ck(cudaMemset(d_candidates, 0, sizeof(unsigned long long)), "face zero candidates");
    ck(cudaMemset(d_rounds, 0, sizeof(int)), "face zero rounds");
    ck(cudaMemset(d_error, 0, sizeof(int)), "face zero error");

    const Rank one_pass_blocks = (components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(std::max<Rank>(
        1, std::min<Rank>(blocks, one_pass_blocks)));
    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "face event begin");
    ck(cudaEventCreate(&end), "face event end");
    ck(cudaEventRecord(begin), "face record begin");
    parallel_face_selfcheck_kernel<<<launch_blocks, THREADS>>>(
        W, components, d_deep, d_serial, d_candidates, d_rounds, d_error);
    ck(cudaGetLastError(), "face launch");
    ck(cudaEventRecord(end), "face record end");
    ck(cudaEventSynchronize(end), "face sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, begin, end), "face elapsed");

    unsigned long long deep = 0;
    unsigned long long serial = 0;
    unsigned long long candidates = 0;
    int rounds = 0;
    int error = 0;
    ck(cudaMemcpy(&deep, d_deep, sizeof(deep), cudaMemcpyDeviceToHost), "face copy deep");
    ck(cudaMemcpy(&serial, d_serial, sizeof(serial), cudaMemcpyDeviceToHost), "face copy serial");
    ck(cudaMemcpy(&candidates, d_candidates, sizeof(candidates), cudaMemcpyDeviceToHost), "face copy candidates");
    ck(cudaMemcpy(&rounds, d_rounds, sizeof(rounds), cudaMemcpyDeviceToHost), "face copy rounds");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "face copy error");
    if (error) {
        std::cerr << "FAIL parallel face error=" << error << '\n';
        return 6;
    }

    std::cout << "two-cell-parallel-face-microprobe"
              << " W=" << W
              << " deep_components=" << deep
              << " central_preimages=" << serial
              << " parallel_candidates=" << candidates
              << " worst_partner_rounds=" << rounds
              << " kernel_ms=" << ms
              << " serial_device_oracle=OK"
              << "\n";
    std::cout << "ALL_OK two_cell_parallel_face_cuda=1 W=" << W << '\n';

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_deep);
    cudaFree(d_serial);
    cudaFree(d_candidates);
    cudaFree(d_rounds);
    cudaFree(d_error);
    return 0;
}
