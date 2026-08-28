#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_traffic_microprobe_main_unused
#include "gridfp_reduced_production_p2p_traffic_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void p2p_owner_lut_equivalence_kernel(
    const std::uint8_t* __restrict__ owner_lut,
    Rank64 outer_count,
    int L,
    int O,
    int ngpu,
    unsigned long long* __restrict__ checked,
    int* error
) {
    unsigned long long local_checked = 0;
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    for (Rank64 outer = first; outer < outer_count; outer += stride) {
        const int expected = weighted_outer_owner_device(
            static_cast<std::uint32_t>(outer), L, O, ngpu);
        const int got = owner_lut[outer];
        if (expected != got) set_error(error, 221);
        ++local_checked;
    }
    if (local_checked) atomicAdd(checked, local_checked);
}

void run_owner_lut_proof(int W, int Kwin, int ngpu, unsigned requested_blocks) {
    ProductionFactorTables tables(W);
    const int L = Kwin + 2;
    const int O = W - L;
    if (O < 0 || O > 20) fail("owner LUT proof outer width");
    const auto owner_lut = build_traffic_owner_lut(tables, Kwin, ngpu);
    const Rank64 outer_count = Rank64(1) << O;
    if (owner_lut.size() != static_cast<std::size_t>(outer_count))
        fail("owner LUT proof size");

    ck(cudaSetDevice(0), "owner LUT proof set device");
    install_tables(tables);

    std::uint8_t* d_lut = nullptr;
    unsigned long long* d_checked = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_lut, owner_lut.size()), "owner LUT proof alloc LUT");
    ck(cudaMemcpy(d_lut, owner_lut.data(), owner_lut.size(), cudaMemcpyHostToDevice),
       "owner LUT proof copy LUT");
    ck(cudaMalloc(&d_checked, sizeof(unsigned long long)),
       "owner LUT proof alloc checked");
    ck(cudaMalloc(&d_error, sizeof(int)), "owner LUT proof alloc error");
    ck(cudaMemset(d_checked, 0, sizeof(unsigned long long)),
       "owner LUT proof zero checked");
    ck(cudaMemset(d_error, 0, sizeof(int)), "owner LUT proof zero error");

    const Rank64 needed_blocks =
        (outer_count + Rank64(THREADS) - 1) / Rank64(THREADS);
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed_blocks)));
    p2p_owner_lut_equivalence_kernel<<<blocks, THREADS>>>(
        d_lut, outer_count, L, O, ngpu, d_checked, d_error);
    ck(cudaGetLastError(), "owner LUT proof launch");
    ck(cudaDeviceSynchronize(), "owner LUT proof sync");

    int error = 0;
    unsigned long long checked = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "owner LUT proof copy error");
    ck(cudaMemcpy(&checked, d_checked, sizeof(checked), cudaMemcpyDeviceToHost),
       "owner LUT proof copy checked");
    if (error) fail("P2P owner LUT mismatch");
    if (checked != outer_count) fail("P2P owner LUT incomplete proof");

    std::cout << "gridfp-p2p-owner-lut-proof"
              << " W=" << W
              << " Kwin=" << Kwin
              << " ngpu=" << ngpu
              << " outer_bits=" << O
              << " owner_lut_bytes=" << owner_lut.size()
              << " checked_entries=" << checked
              << " device_weighted_owner_equivalent=1 exact=OK\n";

    cudaFree(d_error);
    cudaFree(d_checked);
    cudaFree(d_lut);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 13;
    const unsigned blocks = argc > 3
        ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    if (W < 6 || W > RP_MAX_W || Kwin < 1 || Kwin + 2 > W ||
        !blocks || ngpu < 2 || ngpu > 8) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "owner LUT proof device count");
    if (visible < 1) return 3;
    run_owner_lut_proof(W, Kwin, ngpu, blocks);
    std::cout << "ALL_OK gridfp_p2p_owner_lut=1\n";
    return 0;
}
