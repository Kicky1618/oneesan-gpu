#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_compact_segment_probe_main_unused
#include "gridfp_reduced_production_p2p_compact_segment_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <map>

namespace {

static constexpr int P2P_PC_CLASSES = 14;

int pc_class_from_count(Rank pc) {
    for (int c = 0; c < P2P_PC_CLASSES; ++c)
        if (catalan(c + 1) == pc) return c;
    fail("segment-major primitive count class");
}

struct SegmentMajorNetworkGroup {
    std::vector<std::uint32_t> run_begin; // one per segment + sentinel
    PackedRunSoA source;                  // one remote predecessor per segment
    PackedRunSoA local;                   // destination-local runs, concatenated
};

struct SegmentMajorLocalGroup {
    std::vector<std::uint32_t> run_begin; // one per cycle + sentinel
    PackedRunSoA local;
};

struct SegmentMajorGpu {
    std::vector<std::array<SegmentMajorNetworkGroup, P2P_PC_CLASSES>> network; // batch
    std::array<SegmentMajorLocalGroup, P2P_PC_CLASSES> local;
    Rank scratch_states = 0;
};

struct SegmentMajorSchedule {
    std::vector<SegmentMajorGpu> gpu;
    Rank network_segments = 0;
    Rank local_cycles = 0;
    Rank local_run_records = 0;
};

void push_packed(PackedRunSoA& dst, std::uint32_t low, std::uint8_t high) {
    dst.low.push_back(low);
    dst.high.push_back(high);
}

void finish_group_sentinels(SegmentMajorSchedule& out, int batches) {
    for (auto& gpu : out.gpu) {
        for (int b = 0; b < batches; ++b) {
            for (auto& group : gpu.network[static_cast<std::size_t>(b)]) {
                group.run_begin.push_back(static_cast<std::uint32_t>(group.local.low.size()));
                if (group.local.low.size() != group.local.high.size() ||
                    group.source.low.size() != group.source.high.size() ||
                    group.source.low.size() + 1 != group.run_begin.size())
                    fail("segment-major network group shape");
            }
        }
        for (auto& group : gpu.local) {
            group.run_begin.push_back(static_cast<std::uint32_t>(group.local.low.size()));
            if (group.local.low.size() != group.local.high.size() || group.run_begin.empty())
                fail("segment-major local group shape");
        }
    }
}

SegmentMajorSchedule compile_segment_major(
    const CompactSegmentSchedule& compact,
    int ngpu,
    int batches
) {
    SegmentMajorSchedule out;
    out.gpu.resize(static_cast<std::size_t>(ngpu));
    for (auto& g : out.gpu) g.network.resize(static_cast<std::size_t>(batches));

    // Local-only cycles: each packed run is stored exactly once on its owner.
    for (int g = 0; g < ngpu; ++g) {
        for (std::uint32_t ci32 : compact.local_cycle[static_cast<std::size_t>(g)]) {
            const Rank ci = ci32;
            const auto h = compact.header.at(static_cast<std::size_t>(ci));
            const int len = compiled_header_len(h);
            const Rank pc = compiled_header_pc(h);
            const int cls = pc_class_from_count(pc);
            auto& group = out.gpu[static_cast<std::size_t>(g)].local[static_cast<std::size_t>(cls)];
            if (group.local.low.size() > std::numeric_limits<std::uint32_t>::max())
                fail("segment-major local run offset overflow");
            group.run_begin.push_back(static_cast<std::uint32_t>(group.local.low.size()));
            for (int i = 0; i < len; ++i) {
                const Rank r = Rank(h.run_begin) + i;
                if (packed_owner(compact, r) != g) fail("segment-major local owner");
                push_packed(group.local, compact.run.low[static_cast<std::size_t>(r)],
                            compact.run.high[static_cast<std::size_t>(r)]);
                ++out.local_run_records;
            }
            ++out.local_cycles;
        }
    }

    // Network cycles are already split into owner-local segments by the compact
    // descriptor. Store only one remote predecessor record and the local run
    // sequence; cycle ids and scratch offsets disappear from runtime metadata.
    for (int b = 0; b < batches; ++b) {
        for (int g = 0; g < ngpu; ++g) {
            for (std::uint64_t z : compact.segment[static_cast<std::size_t>(b)][static_cast<std::size_t>(g)]) {
                const Rank ci = segment_desc_cycle(z);
                const int start = segment_desc_start(z);
                const auto h = compact.header.at(static_cast<std::size_t>(ci));
                const int len = compiled_header_len(h);
                const Rank pc = compiled_header_pc(h);
                const int cls = pc_class_from_count(pc);
                const Rank rb = h.run_begin;
                const int pred = (start + len - 1) % len;
                if (packed_owner(compact, rb + start) != g ||
                    packed_owner(compact, rb + pred) == g)
                    fail("segment-major boundary decode");

                auto& group = out.gpu[static_cast<std::size_t>(g)]
                                 .network[static_cast<std::size_t>(b)][static_cast<std::size_t>(cls)];
                if (group.local.low.size() > std::numeric_limits<std::uint32_t>::max())
                    fail("segment-major network run offset overflow");
                group.run_begin.push_back(static_cast<std::uint32_t>(group.local.low.size()));
                push_packed(group.source, compact.run.low[static_cast<std::size_t>(rb + pred)],
                            compact.run.high[static_cast<std::size_t>(rb + pred)]);

                int seg_len = 1;
                while (seg_len < len && packed_owner(
                           compact, rb + ((start + seg_len) % len)) == g)
                    ++seg_len;
                if (seg_len >= len) fail("segment-major network segment swallowed cycle");
                for (int j = 0; j < seg_len; ++j) {
                    const Rank r = rb + ((start + j) % len);
                    if (packed_owner(compact, r) != g) fail("segment-major local sequence owner");
                    push_packed(group.local, compact.run.low[static_cast<std::size_t>(r)],
                                compact.run.high[static_cast<std::size_t>(r)]);
                    ++out.local_run_records;
                }
                ++out.network_segments;
            }
        }
    }

    finish_group_sentinels(out, batches);

    // Scratch offsets are implicit inside each (batch,gpu,pc-class): segment i
    // occupies [class_base+i*pc, class_base+(i+1)*pc). Only tiny class bases are
    // required at runtime.
    for (auto& gpu : out.gpu) {
        Rank worst = 0;
        for (int b = 0; b < batches; ++b) {
            Rank total = 0;
            for (int cls = 0; cls < P2P_PC_CLASSES; ++cls) {
                const auto& group = gpu.network[static_cast<std::size_t>(b)][static_cast<std::size_t>(cls)];
                const Rank nseg = group.source.low.size();
                total += nseg * catalan(cls + 1);
            }
            worst = std::max(worst, total);
        }
        gpu.scratch_states = worst;
    }
    return out;
}

std::uint64_t packed_key(std::uint32_t low, std::uint8_t high) {
    return unpack_run_39(low, high);
}

void verify_segment_major_execution(
    const CompactSegmentSchedule& compact,
    const SegmentMajorSchedule& major,
    int ngpu,
    int batches
) {
    std::map<std::uint64_t, Rank> initial, got, want;
    Rank serial = 1;
    for (Rank r = 0; r < compact.run.low.size(); ++r) {
        const auto key = packed_key(compact.run.low[static_cast<std::size_t>(r)],
                                    compact.run.high[static_cast<std::size_t>(r)]);
        if (!initial.emplace(key, serial++).second) fail("segment-major duplicate run key");
    }
    got = initial;
    for (Rank ci = 0; ci < compact.header.size(); ++ci) {
        const auto h = compact.header[static_cast<std::size_t>(ci)];
        const int len = compiled_header_len(h);
        for (int i = 0; i < len; ++i) {
            const Rank d = Rank(h.run_begin) + i;
            const Rank s = Rank(h.run_begin) + ((i + len - 1) % len);
            want[packed_key(compact.run.low[static_cast<std::size_t>(d)],
                            compact.run.high[static_cast<std::size_t>(d)])] =
                initial.at(packed_key(compact.run.low[static_cast<std::size_t>(s)],
                                      compact.run.high[static_cast<std::size_t>(s)]));
        }
    }

    // Local cycles.
    for (int g = 0; g < ngpu; ++g) {
        for (int cls = 0; cls < P2P_PC_CLASSES; ++cls) {
            const auto& group = major.gpu[static_cast<std::size_t>(g)].local[static_cast<std::size_t>(cls)];
            for (std::size_t si = 0; si + 1 < group.run_begin.size(); ++si) {
                const std::uint32_t begin = group.run_begin[si], end = group.run_begin[si + 1];
                if (end <= begin) fail("segment-major empty local cycle");
                std::vector<std::uint64_t> key;
                for (std::uint32_t r = begin; r < end; ++r)
                    key.push_back(packed_key(group.local.low[r], group.local.high[r]));
                Rank temp = got.at(key.back());
                for (int i = int(key.size()) - 1; i > 0; --i) got[key[static_cast<std::size_t>(i)]] = got.at(key[static_cast<std::size_t>(i - 1)]);
                got[key[0]] = temp;
            }
        }
    }

    // Network cycles, batch-atomic. Scratch is addressed implicitly by pc class.
    for (int b = 0; b < batches; ++b) {
        std::vector<std::vector<Rank>> scratch(static_cast<std::size_t>(ngpu));
        for (int g = 0; g < ngpu; ++g)
            scratch[static_cast<std::size_t>(g)].resize(
                static_cast<std::size_t>(major.gpu[static_cast<std::size_t>(g)].scratch_states));

        for (int g = 0; g < ngpu; ++g) {
            Rank class_base = 0;
            for (int cls = 0; cls < P2P_PC_CLASSES; ++cls) {
                const Rank pc = catalan(cls + 1);
                const auto& group = major.gpu[static_cast<std::size_t>(g)]
                                      .network[static_cast<std::size_t>(b)][static_cast<std::size_t>(cls)];
                for (std::size_t si = 0; si < group.source.low.size(); ++si) {
                    const Rank off = class_base + Rank(si) * pc;
                    scratch[static_cast<std::size_t>(g)][static_cast<std::size_t>(off)] =
                        got.at(packed_key(group.source.low[si], group.source.high[si]));
                }
                class_base += Rank(group.source.low.size()) * pc;
            }
        }

        for (int g = 0; g < ngpu; ++g) {
            Rank class_base = 0;
            for (int cls = 0; cls < P2P_PC_CLASSES; ++cls) {
                const Rank pc = catalan(cls + 1);
                const auto& group = major.gpu[static_cast<std::size_t>(g)]
                                      .network[static_cast<std::size_t>(b)][static_cast<std::size_t>(cls)];
                for (std::size_t si = 0; si < group.source.low.size(); ++si) {
                    const std::uint32_t begin = group.run_begin[si], end = group.run_begin[si + 1];
                    if (end <= begin) fail("segment-major empty network segment");
                    std::vector<std::uint64_t> key;
                    for (std::uint32_t r = begin; r < end; ++r)
                        key.push_back(packed_key(group.local.low[r], group.local.high[r]));
                    for (int i = int(key.size()) - 1; i > 0; --i)
                        got[key[static_cast<std::size_t>(i)]] = got.at(key[static_cast<std::size_t>(i - 1)]);
                    const Rank off = class_base + Rank(si) * pc;
                    got[key[0]] = scratch[static_cast<std::size_t>(g)][static_cast<std::size_t>(off)];
                }
                class_base += Rank(group.source.low.size()) * pc;
            }
        }
    }
    if (got != want) fail("segment-major execution mismatch");
}

void verify_segment_major(int W, bool reverse, int ngpu, int batches) {
    const int K = (W - 2) / 2;
    const auto compact = compile_compact_segments(W, K, reverse, ngpu, batches);
    const auto major = compile_segment_major(compact, ngpu, batches);
    verify_segment_major_execution(compact, major, ngpu, batches);

    Rank run_bytes = 0, header_bytes = 0, max_scratch = 0;
    for (const auto& gpu : major.gpu) {
        max_scratch = std::max(max_scratch, gpu.scratch_states);
        for (const auto& group : gpu.local) {
            run_bytes += group.local.low.size() * 5ULL;
            header_bytes += group.run_begin.size() * sizeof(std::uint32_t);
        }
        for (const auto& batch : gpu.network) for (const auto& group : batch) {
            run_bytes += group.local.low.size() * 5ULL;
            run_bytes += group.source.low.size() * 5ULL;
            header_bytes += group.run_begin.size() * sizeof(std::uint32_t);
        }
    }
    if (major.network_segments != compact.network_segments)
        fail("segment-major network segment count");
    if (major.local_run_records != compact.run.low.size())
        fail("segment-major local run coverage");

    std::cout << "W=" << W
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " network_segments=" << major.network_segments
              << " local_cycles=" << major.local_cycles
              << " local_run_records=" << major.local_run_records
              << " run_metadata_bytes=" << run_bytes
              << " header_bytes=" << header_bytes
              << " max_scratch_states=" << max_scratch
              << " pc_classes=14 scratch_offset_per_segment_bytes=0"
              << " cycle_id_per_segment_bytes=0 segment_len_per_segment_bytes=0"
              << " scalar_execution_exact=1\n";
}

void print_w28_segment_major_theory() {
    constexpr int W = 28, K = 13, ngpu = 8;
    std::array<std::array<Rank,14>,8> hist{};
    for (std::uint32_t mask = 0; mask < (1u << K); ++mask) {
        const int r = __builtin_popcount(mask);
        const int owner = weighted_owner(mask, K + 2, K, ngpu);
        ++hist[static_cast<std::size_t>(owner)][static_cast<std::size_t>(r)];
    }
    Rank moved_runs = 0;
    for (int so = 0; so < ngpu; ++so) for (int a = 0; a <= K; ++a) {
        const Rank ca = hist[static_cast<std::size_t>(so)][static_cast<std::size_t>(a)];
        for (int d = 0; d < ngpu; ++d) if (d != so) for (int b = 0; b <= K; ++b) {
            const Rank cb = hist[static_cast<std::size_t>(d)][static_cast<std::size_t>(b)];
            const int base = a + b;
            moved_runs += ca * cb * ((base & 1) ? 2 : 3);
        }
    }
    constexpr Rank nonfixed_runs = 167763968ULL;
    const Rank dominant_bytes = nonfixed_runs * 5ULL + moved_runs * 9ULL;
    std::cout << "W=28 K=13"
              << " network_segments=" << moved_runs
              << " nonfixed_runs=" << nonfixed_runs
              << " dominant_segment_major_GiB=" << double(dominant_bytes) / double(1ULL << 30)
              << " dominant_avg_MiB_per_gpu=" << double(dominant_bytes) / 8.0 / double(1ULL << 20)
              << " formula=5*nonfixed_runs+9*network_segments"
              << " global_cycle_replication=0\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 8;
    const int batches = argc > 3 ? std::atoi(argv[3]) : 8;
    if (maxW < 8 || maxW > 14 || ngpu < 2 || ngpu > 8 || batches < 1 || batches > 64)
        return 2;
    for (int W = 8; W <= maxW; W += 2)
        for (bool reverse : {false, true})
            verify_segment_major(W, reverse, ngpu, batches);
    print_w28_segment_major_theory();
    std::cout << "ALL_OK production_p2p_segment_major_schedule=1\n";
    return 0;
}
