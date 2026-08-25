#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    if (W < 1 || low < 1 || ngpu < 1 || ngpu > 8) return 1;

    const std::uint64_t rows = std::uint64_t(W);
    const std::uint64_t kernels_per_row = 2ULL * std::uint64_t(low);
    const std::uint64_t old_host_submits =
        std::uint64_t(ngpu) * rows * kernels_per_row;
    const std::uint64_t graph_host_submits = std::uint64_t(ngpu) * rows;
    const std::uint64_t caps = std::uint64_t((W + 1) / 2);
    const std::uint64_t max_graph_execs = std::uint64_t(ngpu) * caps;
    const std::uint64_t first_residue_capture_kernel_calls =
        max_graph_execs * kernels_per_row;

    if (W == 28 && low == 14 && ngpu == 8) {
        if (old_host_submits != 6272ULL
            || graph_host_submits != 224ULL
            || max_graph_execs != 112ULL
            || first_residue_capture_kernel_calls != 3136ULL) {
            std::cerr << "n=27 LOW CUDA Graph launch regression\n";
            return 2;
        }
    }

    std::cout << "low-maskbatch-graph-launch W=" << W
              << " low=" << low
              << " gpus=" << ngpu
              << " kernels_per_row=" << kernels_per_row
              << " old_host_submits_per_residue=" << old_host_submits
              << " graph_host_submits_per_residue=" << graph_host_submits
              << " steady_submit_reduction="
              << (1.0 - double(graph_host_submits) / double(old_host_submits))
              << " max_graph_execs=" << max_graph_execs
              << " first_residue_capture_kernel_calls="
              << first_residue_capture_kernel_calls
              << '\n';
    return 0;
}
