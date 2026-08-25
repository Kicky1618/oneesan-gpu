#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

using U64 = std::uint64_t;

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const int max_replicas = argc > 4 ? std::atoi(argv[4]) : 16;
    const int high = W - 1 - low;
    if (W < 4 || low < 1 || high < 1 || high >= 31
        || ngpu < 1 || max_replicas < 1 || max_replicas > 65535)
        return 1;

    const U64 masks = U64(1) << high;
    const U64 masks_per_gpu = (masks + U64(ngpu) - 1) / U64(ngpu);
    const U64 row_caps = U64((W + 1) / 2);
    const U64 stages = U64(low + 1);  // one orbit list + low closure lists
    const U64 desc_bytes = 8;

    // Conservative bound: every owned mask uses max_replicas descriptors in
    // every orbit/closure stage.  Actual resident memory is lower and is
    // printed by the runtime executor after exact planning.
    const U64 bytes_per_cap =
        masks_per_gpu * U64(max_replicas) * stages * desc_bytes;
    const U64 resident_bytes = bytes_per_cap * row_caps;
    const U64 old_h2d_bytes_per_residue = bytes_per_cap * U64(W);
    const U64 old_h2d_calls_per_residue = U64(ngpu) * U64(W) * 2ULL;
    const U64 new_h2d_calls_setup = U64(ngpu) * 2ULL;

    if (W == 28 && low == 14 && ngpu == 8 && max_replicas == 16) {
        if (bytes_per_cap != 1966080ULL
            || resident_bytes != 27525120ULL
            || old_h2d_bytes_per_residue != 55050240ULL
            || old_h2d_calls_per_residue != 448ULL
            || new_h2d_calls_setup != 16ULL) {
            std::cerr << "n=27 LOW resident-row descriptor regression\n";
            return 2;
        }
    }

    std::cout << std::fixed << std::setprecision(6)
              << "low-maskbatch-resident W=" << W
              << " low=" << low << " high=" << high
              << " gpus=" << ngpu
              << " max_replicas=" << max_replicas << '\n'
              << "descriptor_bytes_per_cap_upper=" << bytes_per_cap << '\n'
              << "resident_descriptor_bytes_per_gpu_upper=" << resident_bytes
              << " resident_descriptor_mib_per_gpu_upper="
              << double(resident_bytes) / double(1ULL << 20) << '\n'
              << "old_descriptor_h2d_bytes_per_gpu_per_residue_upper="
              << old_h2d_bytes_per_residue
              << " old_descriptor_h2d_mib_per_gpu_per_residue_upper="
              << double(old_h2d_bytes_per_residue) / double(1ULL << 20) << '\n'
              << "old_descriptor_h2d_calls_per_residue="
              << old_h2d_calls_per_residue
              << " new_descriptor_h2d_calls_setup=" << new_h2d_calls_setup
              << " new_descriptor_h2d_calls_per_residue=0\n";
    return 0;
}
