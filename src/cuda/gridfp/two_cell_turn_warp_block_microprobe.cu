#pragma push_macro("main")
#undef main
#define main two_cell_component_sliding_microprobe_main_unused
#include "two_cell_component_sliding_microprobe.cu"
#pragma pop_macro("main")

#include "two_cell_turn_warp_block.cuh"

namespace {

constexpr int TURN_WARPS = 4;
constexpr int TURN_THREADS = 32 * TURN_WARPS;

__global__ void turn_warp_block_selfcheck_kernel(
    int W,
    Rank components,
    unsigned long long* checked,
    unsigned long long* singular_count,
    unsigned long long* passive_count,
    int* error
) {
    __shared__ PackedKey sh_state[TURN_WARPS][oneesan::twocell::kMaxTurnStates];
    __shared__ int sh_size[TURN_WARPS];
    __shared__ int sh_singular[TURN_WARPS];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank first = Rank(blockIdx.x) * TURN_WARPS + Rank(warp);
    const Rank stride = Rank(gridDim.x) * TURN_WARPS;

    for (Rank cr = first; cr < components; cr += stride) {
        PackedWord label{};
        if (lane == 0) {
            const auto k = oneesan::twocell::component_label_unrank(
                W, cr, TC_RANK_TABLES);
            label = PackedWord{
                k.support, k.left, static_cast<std::uint8_t>(W - 2)};
        }
        label.support = __shfl_sync(0xffffffffu, label.support, 0);
        label.left = __shfl_sync(0xffffffffu, label.left, 0);
        const int len = __shfl_sync(0xffffffffu, int(label.len), 0);
        label.len = static_cast<std::uint8_t>(len);

        oneesan::twocell::cuda_turn::right_turn_closed_block_warp(
            label, W, sh_state[warp], &sh_size[warp], &sh_singular[warp], error);
        __syncwarp();

        if (lane == 0) {
            const auto ref = oneesan::twocell::right_turn_closed_block(label, W);
            if (ref.overflow || ref.size != sh_size[warp] ||
                int(ref.singular) != sh_singular[warp]) {
                set_error(error, 504);
            } else {
                for (int q = 0; q < ref.size; ++q)
                    if (!oneesan::twocell::equal(ref.state[q], sh_state[warp][q]))
                        set_error(error, 505);
            }
            atomicAdd(checked, 1ULL);
            if (sh_singular[warp]) atomicAdd(singular_count, 1ULL);
            else atomicAdd(passive_count,
                           static_cast<unsigned long long>(sh_size[warp] - 2));
        }
        __syncwarp();
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const unsigned blocks = argc > 2
        ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > oneesan::twocell::kMaxWidth || blocks == 0) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const Rank components = oneesan::twocell::component_label_count(W, rt);
    if (plan_only) {
        std::cout << "turn-warp-block-plan W=" << W
                  << " components=" << components
                  << " max_turn_states=" << oneesan::twocell::kMaxTurnStates
                  << " serial_height_loop=0"
                  << " passive_compaction=ballot\n";
        return 0;
    }
    if (W > 11) return 3;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "turn warp device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "turn warp set device");
    ck(cudaMemcpyToSymbol(TC_RANK_TABLES, &rt, sizeof(rt)), "turn warp copy tables");

    unsigned long long *d_checked = nullptr, *d_singular = nullptr, *d_passive = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_checked, sizeof(unsigned long long)), "turn warp alloc checked");
    ck(cudaMalloc(&d_singular, sizeof(unsigned long long)), "turn warp alloc singular");
    ck(cudaMalloc(&d_passive, sizeof(unsigned long long)), "turn warp alloc passive");
    ck(cudaMalloc(&d_error, sizeof(int)), "turn warp alloc error");
    ck(cudaMemset(d_checked, 0, sizeof(unsigned long long)), "turn warp zero checked");
    ck(cudaMemset(d_singular, 0, sizeof(unsigned long long)), "turn warp zero singular");
    ck(cudaMemset(d_passive, 0, sizeof(unsigned long long)), "turn warp zero passive");
    ck(cudaMemset(d_error, 0, sizeof(int)), "turn warp zero error");

    turn_warp_block_selfcheck_kernel<<<blocks, TURN_THREADS>>>(
        W, components, d_checked, d_singular, d_passive, d_error);
    ck(cudaGetLastError(), "turn warp launch");
    ck(cudaDeviceSynchronize(), "turn warp sync");

    unsigned long long checked = 0, singular = 0, passive = 0;
    int error = 0;
    ck(cudaMemcpy(&checked, d_checked, sizeof(checked), cudaMemcpyDeviceToHost),
       "turn warp copy checked");
    ck(cudaMemcpy(&singular, d_singular, sizeof(singular), cudaMemcpyDeviceToHost),
       "turn warp copy singular");
    ck(cudaMemcpy(&passive, d_passive, sizeof(passive), cudaMemcpyDeviceToHost),
       "turn warp copy passive");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "turn warp copy error");
    if (error || checked != components) {
        std::cerr << "FAIL turn warp block error=" << error
                  << " checked=" << checked << '/' << components << '\n';
        return 5;
    }

    std::cout << "turn-warp-block W=" << W
              << " components=" << checked
              << " singular=" << singular
              << " passive_states=" << passive
              << " canonical_order=OK\n";
    cudaFree(d_checked);
    cudaFree(d_singular);
    cudaFree(d_passive);
    cudaFree(d_error);
    return 0;
}
