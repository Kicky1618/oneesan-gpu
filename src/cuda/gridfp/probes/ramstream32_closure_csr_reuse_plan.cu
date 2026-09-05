#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main csr_reuse_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

struct CsrHash128 {
    uint64_t a = 0, b = 0;
    bool operator==(const CsrHash128& o) const { return a == o.a && b == o.b; }
};
struct CsrHash128Hash {
    size_t operator()(const CsrHash128& x) const {
        uint64_t z = x.a ^ (x.b + 0x9e3779b97f4a7c15ull + (x.a << 6) + (x.a >> 2));
        return size_t(z ^ (z >> 32));
    }
};
static inline uint64_t csr_mix(uint64_t h, uint32_t x) {
    h ^= uint64_t(x) + 0x9e3779b9u + (h << 6) + (h >> 2);
    h *= 0x100000001b3ull;
    return h;
}
static CsrHash128 csr_hash_record(
    const BucketFusedDst& r,
    const std::vector<uint32_t>& local,
    const std::vector<uint32_t>& cross
) {
    uint32_t lc = r.counts & 0xffffu, cc = r.counts >> 16;
    uint64_t a = 0xcbf29ce484222325ull, b = 0x84222325cbf29ce4ull;
    a = csr_mix(a, lc); a = csr_mix(a, cc);
    b = csr_mix(b, cc); b = csr_mix(b, lc);
    for (uint32_t e = r.local_begin; e < r.local_begin + lc; ++e) {
        a = csr_mix(a, local[e]);
        b = csr_mix(b, local[e] ^ 0xa5a5a5a5u);
    }
    a = csr_mix(a, 0xffffffffu); b = csr_mix(b, 0x13579bdfu);
    for (uint32_t e = r.cross_begin; e < r.cross_begin + cc; ++e) {
        a = csr_mix(a, cross[e] ^ 0x5a5a5a5au);
        b = csr_mix(b, cross[e]);
    }
    return {a, b};
}
static bool csr_same_record(
    const BucketFusedDst& a, const BucketFusedDst& b,
    const std::vector<uint32_t>& local,
    const std::vector<uint32_t>& cross
) {
    uint32_t alc = a.counts & 0xffffu, acc = a.counts >> 16;
    uint32_t blc = b.counts & 0xffffu, bcc = b.counts >> 16;
    if (alc != blc || acc != bcc) return false;
    for (uint32_t i = 0; i < alc; ++i)
        if (local[a.local_begin + i] != local[b.local_begin + i]) return false;
    for (uint32_t i = 0; i < acc; ++i)
        if (cross[a.cross_begin + i] != cross[b.cross_begin + i]) return false;
    return true;
}

struct CsrReuseStats {
    uint64_t records = 0, unique_patterns = 0;
    uint64_t source_entries = 0, unique_source_entries = 0;
    uint64_t hash_collisions = 0;
    uint32_t max_local = 0, max_cross = 0;
};
static CsrReuseStats csr_analyze(
    const char* tag,
    const std::vector<BucketFusedDst>& rec,
    const std::vector<uint32_t>& local,
    const std::vector<uint32_t>& cross
) {
    CsrReuseStats s; s.records = rec.size();
    std::unordered_map<CsrHash128, uint32_t, CsrHash128Hash> rep;
    rep.reserve(rec.size());
    uint64_t counted_local = 0, counted_cross = 0;
    for (uint32_t q = 0; q < rec.size(); ++q) {
        const auto& r = rec[q];
        uint32_t lc = r.counts & 0xffffu, cc = r.counts >> 16;
        if (uint64_t(r.local_begin) + lc > local.size() || uint64_t(r.cross_begin) + cc > cross.size()) {
            std::cerr << "closure CSR slice overflow " << tag << " q=" << q << '\n';
            std::exit(430);
        }
        counted_local += lc; counted_cross += cc;
        s.max_local = std::max(s.max_local, lc); s.max_cross = std::max(s.max_cross, cc);
        CsrHash128 h = csr_hash_record(r, local, cross);
        auto [it, inserted] = rep.emplace(h, q);
        if (inserted) {
            ++s.unique_patterns;
            s.unique_source_entries += uint64_t(lc) + cc;
        } else if (!csr_same_record(r, rec[it->second], local, cross)) {
            ++s.hash_collisions;
            std::cerr << "closure CSR 128-bit hash collision " << tag << " q=" << q
                      << " rep=" << it->second << '\n';
            std::exit(431);
        }
    }
    s.source_entries = local.size() + cross.size();
    if (counted_local != local.size() || counted_cross != cross.size()) {
        std::cerr << "closure CSR source arrays are not a disjoint record partition " << tag
                  << " counted_local=" << counted_local << " actual_local=" << local.size()
                  << " counted_cross=" << counted_cross << " actual_cross=" << cross.size() << '\n';
        std::exit(432);
    }
    uint64_t old_bytes = s.records * 16ull + s.source_entries * 4ull;
    uint64_t thin_bytes = s.records * 12ull + s.source_entries * 4ull;
    uint64_t dict_bytes = s.records * 4ull + s.unique_patterns * 12ull + s.unique_source_entries * 4ull;
    auto mib = [](uint64_t x) { return double(x) / double(1 << 20); };
    double reuse = s.records ? 1.0 - double(s.unique_patterns) / double(s.records) : 0.0;
    double src_reuse = s.source_entries ? 1.0 - double(s.unique_source_entries) / double(s.source_entries) : 0.0;
    std::cout << std::setprecision(10)
              << "closure-csr-reuse side=" << tag
              << " records=" << s.records
              << " unique_patterns=" << s.unique_patterns
              << " record_reuse=" << reuse
              << " source_entries=" << s.source_entries
              << " unique_source_entries=" << s.unique_source_entries
              << " source_reuse=" << src_reuse
              << " max_local=" << s.max_local
              << " max_cross=" << s.max_cross
              << " old_mib=" << mib(old_bytes)
              << " thin12_mib=" << mib(thin_bytes)
              << " dict4_mib=" << mib(dict_bytes)
              << " dict_vs_thin_ratio=" << (thin_bytes ? double(dict_bytes) / double(thin_bytes) : 0.0)
              << '\n';
    return s;
}

int main() {
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectGatherHost ordinary = build_gpu_direct_gather(layout, lowdesc, loworbit, highdirect);
    GpuDirectCrossGatherHost cross = build_gpu_direct_cross_gather(storage, layout, lowdesc, loworbit, highdirect);
    GpuDirectFusedHost fused = build_gpu_direct_fused_checked(layout, ordinary, cross);
    BucketOwnerHost owner = build_bucket_owners(G_FACTOR, storage);
    BucketFusedHost bf = build_bucket_fused(storage, layout, owner, ordinary, cross, fused);

    ReverseLowDescHost rlow = build_reverse_low_descriptors(storage, layout);
    ReverseHighDescHost rhigh = build_reverse_high_descriptors(storage, layout);
    ReverseOrbitHost rlo = build_reverse_orbit(storage, layout, true);
    ReverseOrbitHost rhi = build_reverse_orbit(storage, layout, false);
    ReverseBucketAtomicHost rb = build_reverse_bucket_atomic(storage, layout, owner, rlow, rhigh, rlo, rhi);
    ReverseBucketFusedHost rf = build_reverse_bucket_fused_checked(layout, owner, rb);

    auto fl = csr_analyze("forward-low", bf.low_dst, bf.low_local_src, bf.low_cross_op);
    auto fh = csr_analyze("forward-high", bf.high_dst, bf.high_local_src, bf.high_cross_op);
    auto rl = csr_analyze("reverse-low", rf.low_dst, rf.low_local_src, rf.low_cross_op);
    auto rh = csr_analyze("reverse-high", rf.high_dst, rf.high_local_src, rf.high_cross_op);
    uint64_t rec = fl.records + fh.records + rl.records + rh.records;
    uint64_t uniq = fl.unique_patterns + fh.unique_patterns + rl.unique_patterns + rh.unique_patterns;
    uint64_t src = fl.source_entries + fh.source_entries + rl.source_entries + rh.source_entries;
    uint64_t usrc = fl.unique_source_entries + fh.unique_source_entries + rl.unique_source_entries + rh.unique_source_entries;
    std::cout << "closure-csr-reuse-plan OK W=" << TARGET_W
              << " records=" << rec << " unique_patterns=" << uniq
              << " record_reuse=" << (rec ? 1.0 - double(uniq) / double(rec) : 0.0)
              << " source_entries=" << src << " unique_source_entries=" << usrc
              << " source_reuse=" << (src ? 1.0 - double(usrc) / double(src) : 0.0)
              << '\n';
    return 0;
}
