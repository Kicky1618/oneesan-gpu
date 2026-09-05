#define main cpasync_remote_peer_bandwidth_both_main_unused
#include "cpasync_remote_peer_bandwidth.cu"
#undef main

#include <cstring>

int main(int argc, char** argv) {
    const int ngpu = argc > 1 ? std::atoi(argv[1]) : 8;
    const std::uint32_t values = argc > 2
        ? std::uint32_t(std::strtoul(argv[2], nullptr, 10)) : (1u << 26);
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const int blocks = argc > 4 ? std::atoi(argv[4]) : 256;
    const std::uint32_t rounds = argc > 5
        ? std::uint32_t(std::strtoul(argv[5], nullptr, 10)) : 32u;
    const int launches = argc > 6 ? std::atoi(argv[6]) : 8;
    const int warmup = argc > 7 ? std::atoi(argv[7]) : 2;
    const char* mode_name = argc > 8 ? argv[8] : "direct";
    const bool cpasync = std::strcmp(mode_name, "cpasync") == 0;
    if (!cpasync && std::strcmp(mode_name, "direct") != 0) {
        std::cerr << "mode must be direct or cpasync\n";
        return 2;
    }
    if (ngpu < 2 || ngpu > 8 || values < 2u || (values & (values - 1u)) != 0u ||
        threads <= 0 || threads > 1024 || blocks <= 0 || !rounds ||
        launches <= 0 || warmup < 0)
        return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < ngpu) fail("not enough visible GPUs");

    const std::uint32_t total_threads = std::uint32_t(threads * blocks);
    const std::uint32_t mask = values - 1u;
    const std::size_t out_count = std::size_t(total_threads);
    std::vector<std::uint32_t*> state(std::size_t(ngpu), nullptr);
    std::vector<unsigned long long*> out(std::size_t(ngpu), nullptr);
    std::vector<DeviceRun> run(std::size_t(ngpu));

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "alloc set device");
        ck(cudaMalloc(&state[g], std::size_t(values) * sizeof(std::uint32_t)), "state alloc");
        ck(cudaMalloc(&out[g], out_count * sizeof(unsigned long long)), "out alloc");
        std::vector<std::uint32_t> host(values);
        for (std::uint32_t i = 0; i < values; ++i) host[i] = pattern(g, i);
        ck(cudaMemcpy(state[g], host.data(), host.size() * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "state init");
        ck(cudaStreamCreateWithFlags(&run[g].stream, cudaStreamNonBlocking), "stream create");
        ck(cudaEventCreate(&run[g].start), "start event create");
        ck(cudaEventCreate(&run[g].stop), "stop event create");
    }
    for (int g = 0; g < ngpu; ++g) enable_peer(g, (g + 1) % ngpu);

    const auto ms = run_mode(
        cpasync ? Mode::CpAsync : Mode::Direct,
        state, out, run, ngpu, mask, threads, blocks, rounds, launches, warmup);
    const std::uint64_t bytes_per_gpu =
        std::uint64_t(total_threads) * std::uint64_t(rounds) *
        std::uint64_t(kLoads) * sizeof(std::uint32_t) * std::uint64_t(launches);
    report(mode_name, ms, bytes_per_gpu, ngpu);
    std::cout << "single_mode=1"
              << " peer_ring=1"
              << " values_per_gpu=" << values
              << " working_set_mib="
              << (double(values) * sizeof(std::uint32_t) / double(1u << 20))
              << " threads=" << threads
              << " blocks=" << blocks
              << " rounds=" << rounds
              << " launches=" << launches
              << " loads_per_thread_round=" << kLoads
              << '\n';

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "free set device");
        cudaEventDestroy(run[g].stop);
        cudaEventDestroy(run[g].start);
        cudaStreamDestroy(run[g].stream);
        cudaFree(out[g]);
        cudaFree(state[g]);
    }
    return 0;
}
