#include <cstdint>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int L = argc > 2 ? std::atoi(argv[2]) : 14;
    const int H = argc > 3 ? std::atoi(argv[3]) : W - L - 1;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    if (W < 3 || (W & 1) || L < 1 || H < 1 || L + H + 1 != W
        || ngpu < 1 || ngpu > 8) {
        std::cerr << "usage: factor_high_stream_end_sync [W L H ngpu]\n";
        return 2;
    }

    // There is exactly one HIGH host worker per GPU per DP row. v0.61/0.72
    // reduce all per-job waits to one row-end wait in that worker. v0.73 keeps
    // the count but narrows the synchronization scope from the whole device to
    // the worker's cudaStreamPerThread.
    const std::uint64_t rows = std::uint64_t(W);
    const std::uint64_t row_workers = rows * std::uint64_t(ngpu);
    const std::uint64_t old_device_wide_waits = row_workers;
    const std::uint64_t new_stream_waits = row_workers;
    const std::uint64_t removed_device_wide_waits = old_device_wide_waits;

    std::cout << "W=" << W
              << " L=" << L
              << " H=" << H
              << " ngpu=" << ngpu
              << " high_row_workers_per_residue=" << row_workers
              << " v072_device_wide_row_waits=" << old_device_wide_waits
              << " v073_stream_row_waits=" << new_stream_waits
              << " device_wide_waits_removed=" << removed_device_wide_waits
              << '\n';

    if (W == 28 && L == 14 && H == 13 && ngpu == 8) {
        if (row_workers != 224ULL || old_device_wide_waits != 224ULL
            || new_stream_waits != 224ULL
            || removed_device_wide_waits != 224ULL) {
            std::cerr << "n=27 HIGH stream-end sync model regression\n";
            return 3;
        }
    }
    return 0;
}
