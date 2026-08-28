#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_compiled_schedule_probe_main_unused
#include "gridfp_reduced_production_p2p_compiled_schedule_probe.cpp"
#pragma pop_macro("main")

namespace {

struct PackedRunSoA {
    std::vector<std::uint32_t> low;
    std::vector<std::uint8_t> high;
};

void pack_run_39(std::uint64_t z, std::uint32_t& low, std::uint8_t& high) {
    const int owner = compiled_run_owner(z);
    const Rank local = compiled_run_local(z);
    if (owner < 0 || owner >= 8 || local >= (Rank(1) << 36))
        fail("packed39 range");
    low = std::uint32_t(owner) |
          (std::uint32_t(local & ((Rank(1) << 29) - 1)) << 3);
    high = static_cast<std::uint8_t>((local >> 29) & 0x7fu);
}

std::uint64_t unpack_run_39(std::uint32_t low, std::uint8_t high) {
    const int owner = int(low & 7u);
    const Rank local = Rank(low >> 3) | (Rank(high & 0x7fu) << 29);
    return pack_compiled_run(owner, local);
}

PackedRunSoA pack_schedule_runs(const std::vector<std::uint64_t>& in) {
    PackedRunSoA out;
    out.low.resize(in.size());
    out.high.resize(in.size());
    for (std::size_t i = 0; i < in.size(); ++i)
        pack_run_39(in[i], out.low[i], out.high[i]);
    return out;
}

void verify_packed_schedule(int W, bool reverse, int ngpu) {
    const int K = (W - 2) / 2;
    const auto schedule = compile_schedule(W, K, reverse, ngpu);
    Rank cycles = 0, runs = 0, bytes = 0;
    Rank max_local = 0;
    for (int g = 0; g < ngpu; ++g) {
        const auto& s = schedule[static_cast<std::size_t>(g)];
        const PackedRunSoA packed = pack_schedule_runs(s.run);
        if (packed.low.size() != s.run.size() || packed.high.size() != s.run.size())
            fail("packed39 size mismatch");
        for (std::size_t i = 0; i < s.run.size(); ++i) {
            if (unpack_run_39(packed.low[i], packed.high[i]) != s.run[i])
                fail("packed39 round trip");
            max_local = std::max(max_local, compiled_run_local(s.run[i]));
        }
        cycles += s.header.size();
        runs += s.run.size();
        bytes += s.header.size() * sizeof(CompiledCycleHeader) +
                 s.run.size() * (sizeof(std::uint32_t) + sizeof(std::uint8_t));
    }
    const Rank old_bytes = cycles * sizeof(CompiledCycleHeader) +
                           runs * sizeof(std::uint64_t);
    if (bytes > old_bytes) fail("packed39 metadata grew");
    std::cout << "W=" << W
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " cycles=" << cycles
              << " runs=" << runs
              << " max_local=" << max_local
              << " local_fits_36bit=1"
              << " packed_metadata_bytes=" << bytes
              << " u64_metadata_bytes=" << old_bytes
              << " packed_over_u64=" << (old_bytes ? double(bytes) / double(old_bytes) : 0.0)
              << " run_low32_bytes=4 run_high8_bytes=1"
              << " exact_roundtrip=1\n";
}

void print_w28_packed_theory() {
    constexpr Rank runs = 167763968ULL;
    constexpr Rank cycles = 21566612ULL;
    constexpr Rank run_bytes = runs * 5ULL;
    constexpr Rank header_bytes = cycles * 8ULL;
    constexpr Rank total_bytes = run_bytes + header_bytes;
    static_assert(run_bytes == 838819840ULL);
    static_assert(header_bytes == 172532896ULL);
    static_assert(total_bytes == 1011352736ULL);
    std::cout << "W=28 K=13"
              << " packed_run_MiB=" << double(run_bytes) / double(1ULL << 20)
              << " header_MiB=" << double(header_bytes) / double(1ULL << 20)
              << " one_direction_GiB=" << double(total_bytes) / double(1ULL << 30)
              << " avg_one_direction_MiB_per_8gpu="
              << double(total_bytes) / 8.0 / double(1ULL << 20)
              << " forward_reverse_both_GiB="
              << 2.0 * double(total_bytes) / double(1ULL << 30)
              << " avg_both_MiB_per_8gpu="
              << 2.0 * double(total_bytes) / 8.0 / double(1ULL << 20)
              << " run_record_bits=39 storage_bytes=5"
              << " aligned_soa_loads=u32_plus_u8\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    if (maxW < 8 || maxW > 14 || ngpu < 2 || ngpu > 8) return 2;
    for (int W = 8; W <= maxW; W += 2)
        for (bool reverse : {false, true})
            verify_packed_schedule(W, reverse, ngpu);
    print_w28_packed_theory();
    std::cout << "ALL_OK production_p2p_packed_schedule=1\n";
    return 0;
}
