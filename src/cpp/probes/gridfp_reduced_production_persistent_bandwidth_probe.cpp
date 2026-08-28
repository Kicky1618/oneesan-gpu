#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

using Rank = std::uint64_t;

constexpr Rank STATES = 473397057701ULL;
constexpr Rank ROTATED_VALUES = 473330026916ULL;
constexpr Rank PEER_VALUES = 409769189454ULL;

} // namespace

int main(int argc, char** argv) {
    // HGX B300 public peak defaults. Override these with measured sustainable
    // bandwidth when hardware data is available.
    const double node_hbm_tbps = argc > 1 ? std::atof(argv[1]) : 64.0;
    const double node_nvlink_tbps = argc > 2 ? std::atof(argv[2]) : 14.4;
    if (!(node_hbm_tbps > 0.0) || !(node_nvlink_tbps > 0.0)) return 2;

    const Rank local_values = ROTATED_VALUES - PEER_VALUES;

    // Two-phase local-scratch redistribution:
    // - every rotated value is read once from the state stream;
    // - same-owner values are written once to local state;
    // - cross-owner values are written to and read from source-local scratch;
    // - every cross-owner value is then peer-written into destination state.
    const long double source_hbm_bytes =
        8.0L * static_cast<long double>(local_values) +
        12.0L * static_cast<long double>(PEER_VALUES);
    const long double peer_bytes =
        4.0L * static_cast<long double>(PEER_VALUES);
    const long double destination_peer_hbm_bytes = peer_bytes;
    const long double total_hbm_bytes =
        source_hbm_bytes + destination_peer_hbm_bytes;

    const long double hbm_seconds =
        total_hbm_bytes / (static_cast<long double>(node_hbm_tbps) * 1.0e12L);
    const long double nvlink_seconds =
        peer_bytes / (static_cast<long double>(node_nvlink_tbps) * 1.0e12L);
    const long double lower_bound_seconds = std::max(hbm_seconds, nvlink_seconds);

    const Rank fixed_values = STATES - ROTATED_VALUES;
    const double cross_fraction =
        double(PEER_VALUES) / double(ROTATED_VALUES);

    std::cout << std::fixed << std::setprecision(9)
              << "gridfp-persistent-bandwidth-model"
              << " states=" << STATES
              << " rotated_values=" << ROTATED_VALUES
              << " fixed_values=" << fixed_values
              << " peer_values=" << PEER_VALUES
              << " cross_fraction=" << cross_fraction
              << " source_HBM_TB=" << double(source_hbm_bytes / 1.0e12L)
              << " peer_payload_TB=" << double(peer_bytes / 1.0e12L)
              << " destination_peer_HBM_TB="
              << double(destination_peer_hbm_bytes / 1.0e12L)
              << " total_HBM_TB=" << double(total_hbm_bytes / 1.0e12L)
              << " node_HBM_TBps=" << node_hbm_tbps
              << " node_NVLink_TBps=" << node_nvlink_tbps
              << " HBM_floor_ms=" << double(hbm_seconds * 1000.0L)
              << " NVLink_floor_ms=" << double(nvlink_seconds * 1000.0L)
              << " optimistic_redistribution_floor_ms="
              << double(lower_bound_seconds * 1000.0L)
              << " bottleneck="
              << (nvlink_seconds >= hbm_seconds ? "NVLink" : "HBM")
              << " HBM_NVLink_floor_ratio="
              << double(hbm_seconds / nvlink_seconds)
              << " exact_traffic_model=1\n";

    if (fixed_values != 67030785ULL) return 3;
    if (peer_bytes != 1639076757816.0L) return 4;
    if (total_hbm_bytes != 7064793730960.0L) return 5;
    return 0;
}
