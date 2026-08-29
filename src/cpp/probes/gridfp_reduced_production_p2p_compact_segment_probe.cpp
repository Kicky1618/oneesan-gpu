#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_packed_schedule_probe_main_unused
#include "gridfp_reduced_production_p2p_packed_schedule_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <set>

namespace {

static constexpr int SEG_CYCLE_BITS = 25;
static constexpr int SEG_START_BITS = 5;
static constexpr int SEG_SCRATCH_SHIFT = SEG_CYCLE_BITS + SEG_START_BITS;
static constexpr std::uint64_t SEG_CYCLE_MASK = (std::uint64_t(1) << SEG_CYCLE_BITS) - 1u;
static constexpr std::uint64_t SEG_START_MASK = (std::uint64_t(1) << SEG_START_BITS) - 1u;
static constexpr std::uint64_t SEG_SCRATCH_LIMIT = std::uint64_t(1) << (64 - SEG_SCRATCH_SHIFT);

std::uint32_t compact_segment_hash(std::uint32_t support, bool blocked, bool reverse) {
    std::uint32_t x = support ^ 0x9e3779b9u;
    x ^= blocked ? 0x27d4eb2du : 0u;
    x ^= reverse ? 0x165667b1u : 0u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

std::uint64_t pack_segment_desc(Rank cycle, int start, Rank scratch) {
    if (cycle >= (Rank(1) << SEG_CYCLE_BITS) || start < 0 || start >= (1 << SEG_START_BITS) ||
        scratch >= SEG_SCRATCH_LIMIT)
        fail("compact segment descriptor packing range");
    return std::uint64_t(cycle) |
           (std::uint64_t(start) << SEG_CYCLE_BITS) |
           (std::uint64_t(scratch) << SEG_SCRATCH_SHIFT);
}

Rank segment_desc_cycle(std::uint64_t z) { return Rank(z & SEG_CYCLE_MASK); }
int segment_desc_start(std::uint64_t z) {
    return int((z >> SEG_CYCLE_BITS) & SEG_START_MASK);
}
Rank segment_desc_scratch(std::uint64_t z) { return Rank(z >> SEG_SCRATCH_SHIFT); }

struct CompactSegmentSchedule {
    std::vector<CompiledCycleHeader> header;
    PackedRunSoA run;
    std::vector<std::vector<std::vector<std::uint64_t>>> segment; // [batch][gpu]
    std::vector<std::vector<std::uint32_t>> local_cycle;         // [gpu]
    std::vector<Rank> scratch_states;
    Rank network_segments = 0;
    __uint128_t network_states = 0;
};

int packed_owner(const CompactSegmentSchedule& s, Rank run) {
    return int(s.run.low.at(static_cast<std::size_t>(run)) & 7u);
}

Rank packed_local(const CompactSegmentSchedule& s, Rank run) {
    const std::uint32_t low = s.run.low.at(static_cast<std::size_t>(run));
    const std::uint8_t high = s.run.high.at(static_cast<std::size_t>(run));
    return compiled_run_local(unpack_run_39(low, high));
}

CompactSegmentSchedule compile_compact_segments(
    int W, int K, bool reverse, int ngpu, int batches
) {
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    const OwnerPlan owner_plan = make_owner_plan(W, K, ngpu);
    const auto actual = actual_run_map(W, K, reverse, ngpu);

    CompactSegmentSchedule out;
    out.segment.resize(static_cast<std::size_t>(batches));
    for (auto& b : out.segment) b.resize(static_cast<std::size_t>(ngpu));
    out.local_cycle.resize(static_cast<std::size_t>(ngpu));
    out.scratch_states.assign(static_cast<std::size_t>(ngpu), 0);
    std::vector<std::vector<Rank>> batch_scratch(
        static_cast<std::size_t>(batches), std::vector<Rank>(static_cast<std::size_t>(ngpu), 0));

    std::set<RunKey> seen;
    for (const auto& [root, root_rec] : actual) {
        if (seen.count(root)) continue;
        std::vector<RunKey> cycle;
        RunKey cur = root;
        do {
            if (!actual.count(cur)) fail("compact segment cycle escaped run map");
            if (!seen.insert(cur).second) fail("compact segment cycle overlap");
            cycle.push_back(cur);
            cur = shifted_next_key(cur, W, q, K, K, reverse);
            if (cycle.size() > static_cast<std::size_t>(RP_MAX_W))
                fail("compact segment cycle too long");
        } while (!(cur.support == root.support && cur.blocked == root.blocked));
        if (cycle.size() <= 1) continue;

        if (out.header.size() >= (std::size_t(1) << SEG_CYCLE_BITS))
            fail("compact segment cycle id overflow");
        const Rank cycle_id = out.header.size();
        const Rank run_begin = out.run.low.size();
        const Rank pc = root_rec.pc;
        out.header.push_back(make_compiled_header(run_begin, pc, int(cycle.size())));

        std::vector<int> owners(cycle.size());
        for (std::size_t i = 0; i < cycle.size(); ++i) {
            const auto& rec = actual.at(cycle[i]);
            if (rec.pc != pc) fail("compact segment pc changed in cycle");
            owners[i] = rec.owner;
            std::uint32_t low = 0;
            std::uint8_t high = 0;
            pack_run_39(pack_compiled_run(rec.owner, rec.base), low, high);
            out.run.low.push_back(low);
            out.run.high.push_back(high);
        }

        std::vector<int> starts;
        for (int i = 0; i < static_cast<int>(cycle.size()); ++i) {
            const int p = (i + static_cast<int>(cycle.size()) - 1) % int(cycle.size());
            if (owners[static_cast<std::size_t>(i)] != owners[static_cast<std::size_t>(p)])
                starts.push_back(i);
        }
        if (starts.empty()) {
            const int owner = owners[0];
            out.local_cycle[static_cast<std::size_t>(owner)].push_back(
                static_cast<std::uint32_t>(cycle_id));
            continue;
        }

        const int batch = int(compact_segment_hash(root.support, root.blocked, reverse) %
                              std::uint32_t(batches));
        for (int start : starts) {
            const int owner = owners[static_cast<std::size_t>(start)];
            Rank& off = batch_scratch[static_cast<std::size_t>(batch)][static_cast<std::size_t>(owner)];
            out.segment[static_cast<std::size_t>(batch)][static_cast<std::size_t>(owner)].push_back(
                pack_segment_desc(cycle_id, start, off));
            off += pc;
            if (off >= SEG_SCRATCH_LIMIT) fail("compact segment scratch offset overflow");
            out.network_states += pc;
            ++out.network_segments;
        }
    }
    if (seen.size() != actual.size()) fail("compact segment run coverage");
    for (int b = 0; b < batches; ++b)
        for (int g = 0; g < ngpu; ++g)
            out.scratch_states[static_cast<std::size_t>(g)] = std::max(
                out.scratch_states[static_cast<std::size_t>(g)],
                batch_scratch[static_cast<std::size_t>(b)][static_cast<std::size_t>(g)]);
    return out;
}

void verify_compact_execution(
    const CompactSegmentSchedule& s, int ngpu, int batches
) {
    const Rank nruns = s.run.low.size();
    std::vector<Rank> old(static_cast<std::size_t>(nruns));
    std::vector<Rank> got(static_cast<std::size_t>(nruns));
    std::vector<Rank> want(static_cast<std::size_t>(nruns));
    for (Rank r = 0; r < nruns; ++r) old[static_cast<std::size_t>(r)] = got[static_cast<std::size_t>(r)] = r + 1;

    for (Rank ci = 0; ci < s.header.size(); ++ci) {
        const auto h = s.header[static_cast<std::size_t>(ci)];
        const int len = compiled_header_len(h);
        const Rank rb = h.run_begin;
        for (int i = 0; i < len; ++i)
            want[static_cast<std::size_t>(rb + i)] = old[static_cast<std::size_t>(rb + (i + len - 1) % len)];
    }

    // Local-only cycles need no scratch or peer traffic.
    for (int g = 0; g < ngpu; ++g) {
        for (std::uint32_t ci32 : s.local_cycle[static_cast<std::size_t>(g)]) {
            const Rank ci = ci32;
            const auto h = s.header.at(static_cast<std::size_t>(ci));
            const int len = compiled_header_len(h);
            const Rank rb = h.run_begin;
            for (int i = 0; i < len; ++i)
                if (packed_owner(s, rb + i) != g) fail("compact local cycle owner mismatch");
            Rank temp = got[static_cast<std::size_t>(rb + len - 1)];
            for (int i = len - 1; i > 0; --i)
                got[static_cast<std::size_t>(rb + i)] = got[static_cast<std::size_t>(rb + i - 1)];
            got[static_cast<std::size_t>(rb)] = temp;
        }
    }

    for (int b = 0; b < batches; ++b) {
        std::vector<std::vector<Rank>> scratch(static_cast<std::size_t>(ngpu));
        for (int g = 0; g < ngpu; ++g)
            scratch[static_cast<std::size_t>(g)].resize(
                static_cast<std::size_t>(s.scratch_states[static_cast<std::size_t>(g)]));

        // Gather every predecessor before modifying any network cycle in this batch.
        for (int g = 0; g < ngpu; ++g) {
            for (std::uint64_t z : s.segment[static_cast<std::size_t>(b)][static_cast<std::size_t>(g)]) {
                const Rank ci = segment_desc_cycle(z);
                const int start = segment_desc_start(z);
                const Rank so = segment_desc_scratch(z);
                const auto h = s.header.at(static_cast<std::size_t>(ci));
                const int len = compiled_header_len(h);
                const Rank rb = h.run_begin;
                if (start >= len || packed_owner(s, rb + start) != g)
                    fail("compact segment start decode");
                const int pred = (start + len - 1) % len;
                if (packed_owner(s, rb + pred) == g)
                    fail("compact segment predecessor not remote");
                if (so >= scratch[static_cast<std::size_t>(g)].size())
                    fail("compact scratch range");
                scratch[static_cast<std::size_t>(g)][static_cast<std::size_t>(so)] =
                    got[static_cast<std::size_t>(rb + pred)];
            }
        }

        // Decode segment length from owner changes in the shared packed run list.
        for (int g = 0; g < ngpu; ++g) {
            for (std::uint64_t z : s.segment[static_cast<std::size_t>(b)][static_cast<std::size_t>(g)]) {
                const Rank ci = segment_desc_cycle(z);
                const int start = segment_desc_start(z);
                const Rank so = segment_desc_scratch(z);
                const auto h = s.header.at(static_cast<std::size_t>(ci));
                const int len = compiled_header_len(h);
                const Rank rb = h.run_begin;
                int seg_len = 1;
                while (seg_len < len &&
                       packed_owner(s, rb + ((start + seg_len) % len)) == g)
                    ++seg_len;
                if (seg_len >= len) fail("compact network segment swallowed cycle");
                for (int j = seg_len - 1; j > 0; --j) {
                    const int dst = (start + j) % len;
                    const int src = (start + j - 1) % len;
                    got[static_cast<std::size_t>(rb + dst)] = got[static_cast<std::size_t>(rb + src)];
                }
                got[static_cast<std::size_t>(rb + start)] =
                    scratch[static_cast<std::size_t>(g)][static_cast<std::size_t>(so)];
            }
        }
    }
    if (got != want) fail("compact segmented execution mismatch");
}

void verify_compact_segment_schedule(int W, bool reverse, int ngpu, int batches) {
    const int K = (W - 2) / 2;
    const auto s = compile_compact_segments(W, K, reverse, ngpu, batches);
    verify_compact_execution(s, ngpu, batches);

    Rank segment_bytes = 0, local_bytes = 0;
    for (int b = 0; b < batches; ++b)
        for (int g = 0; g < ngpu; ++g)
            segment_bytes += s.segment[static_cast<std::size_t>(b)][static_cast<std::size_t>(g)].size() * sizeof(std::uint64_t);
    for (int g = 0; g < ngpu; ++g)
        local_bytes += s.local_cycle[static_cast<std::size_t>(g)].size() * sizeof(std::uint32_t);
    const Rank packed_bytes = s.header.size() * sizeof(CompiledCycleHeader) +
                              s.run.low.size() * 5ULL;
    Rank max_scratch = 0;
    for (Rank z : s.scratch_states) max_scratch = std::max(max_scratch, z);

    std::cout << "W=" << W
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " cycles=" << s.header.size()
              << " runs=" << s.run.low.size()
              << " network_segments=" << s.network_segments
              << " packed_cycle_run_bytes=" << packed_bytes
              << " segment_descriptor_bytes=" << segment_bytes
              << " local_cycle_id_bytes=" << local_bytes
              << " max_scratch_states=" << max_scratch
              << " descriptor_cycle_bits=25 descriptor_start_bits=5 descriptor_scratch_bits=34"
              << " segment_len_inferred_from_owner_runs=1 scalar_execution_exact=1\n";
}

void print_w28_compact_segment_theory(int batches) {
    constexpr int W = 28, K = 13, ngpu = 8;
    // For K=13 the two 15-site windows cover all 28 sites, so there are no
    // common outer bits. Aggregate the 13-bit exclusive supports by owner and
    // popcount, then count cross-owner run boundaries without enumerating states.
    std::array<std::array<Rank,14>,8> hist{};
    for (std::uint32_t mask = 0; mask < (1u << K); ++mask) {
        const int r = __builtin_popcount(mask);
        const int owner = weighted_owner(mask, K + 2, K, ngpu);
        ++hist[static_cast<std::size_t>(owner)][static_cast<std::size_t>(r)];
    }
    Rank moved_runs = 0;
    Rank moved_states = 0;
    for (int so = 0; so < ngpu; ++so) for (int a = 0; a <= K; ++a) {
        const Rank ca = hist[static_cast<std::size_t>(so)][static_cast<std::size_t>(a)];
        if (!ca) continue;
        for (int d = 0; d < ngpu; ++d) if (d != so) for (int b = 0; b <= K; ++b) {
            const Rank cb = hist[static_cast<std::size_t>(d)][static_cast<std::size_t>(b)];
            if (!cb) continue;
            const int base = a + b;
            const Rank run_weight = (base & 1) ? 2 : 3;
            const Rank state_weight = (base & 1)
                ? catalan((base + 1) / 2) + catalan((base + 3) / 2)
                : 3 * catalan((base + 2) / 2);
            moved_runs += ca * cb * run_weight;
            moved_states += ca * cb * state_weight;
        }
    }
    if (moved_states != 409769189454ULL) fail("W28 compact segment payload constant");
    constexpr Rank packed_cycle_run_bytes = 1011352736ULL;
    const Rank segment_bytes = moved_runs * sizeof(std::uint64_t);
    std::cout << "W=28 K=13 ngpu=8"
              << " batches=" << batches
              << " network_segments=" << moved_runs
              << " exact_network_states=" << moved_states
              << " exact_network_TiB=" << double(moved_states) * 4.0 / double(1ULL << 40)
              << " packed_cycle_run_GiB=" << double(packed_cycle_run_bytes) / double(1ULL << 30)
              << " compact_segment_GiB=" << double(segment_bytes) / double(1ULL << 30)
              << " descriptor_bytes=8"
              << " old_segment_rank_array_eliminated=1\n";
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
            verify_compact_segment_schedule(W, reverse, ngpu, batches);
    print_w28_compact_segment_theory(batches);
    std::cout << "ALL_OK production_p2p_compact_segment_metadata=1\n";
    return 0;
}
