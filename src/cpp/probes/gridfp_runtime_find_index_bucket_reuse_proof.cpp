#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

int hash_bucket(std::uint64_t x, int hash_buckets, std::uint64_t& calls) {
    ++calls;
    x ^= x >> 7;
    x ^= x >> 14;
    return int(x & std::uint64_t(hash_buckets - 1));
}

std::uint64_t low_mask(int hash_buckets) {
    return hash_buckets == 64 ? ~0ULL : ((1ULL << hash_buckets) - 1ULL);
}

void remember_bucket(std::uint64_t& occupancy, int hash_buckets, int bucket, bool reuse) {
    if (!reuse || hash_buckets == 64) return;
    const int shift = hash_buckets;
    const std::uint64_t meta_mask = 0x3fULL << shift;
    occupancy = (occupancy & ~meta_mask) |
                (std::uint64_t(bucket + 1) << shift);
}

int record_bucket(std::uint64_t k, std::uint64_t occupancy, int hash_buckets,
                  bool reuse, std::uint64_t& hash_calls) {
    if (reuse && hash_buckets < 64) {
        const int code = int((occupancy >> hash_buckets) & 0x3fULL);
        if (code) return code - 1;
    }
    return hash_bucket(k, hash_buckets, hash_calls);
}

struct Stats {
    std::uint64_t components = 0;
    std::uint64_t records = 0;
    std::uint64_t find_hashes = 0;
    std::uint64_t record_hashes_baseline = 0;
    std::uint64_t record_hashes_reuse = 0;
    std::uint64_t memo_hits = 0;
    std::uint64_t hit_queries = 0;
};

Stats prove_config(int hash_buckets, std::uint64_t seed) {
    constexpr std::uint64_t COMPONENTS = 200000;
    constexpr int MAX_PAIRS = 20;
    std::mt19937_64 rng(seed);
    Stats st{};

    for (std::uint64_t component = 0; component < COMPONENTS; ++component) {
        std::uint64_t occ_reuse = 0;
        std::uint64_t occ_base = 0;
        const int n = 1 + int(rng() % MAX_PAIRS);
        std::vector<std::uint64_t> keys;
        keys.reserve(n);

        for (int i = 0; i < n; ++i) {
            std::uint64_t k = rng();
            while ([&] {
                for (auto x : keys) if (x == k) return true;
                return false;
            }()) k = rng();
            keys.push_back(k);

            // Initial seed record has no preceding find. Every later insertion
            // is preceded by a miss; only that miss updates the memo.
            if (i > 0) {
                const std::uint64_t before_low = occ_reuse & low_mask(hash_buckets);
                const int bfind = hash_bucket(k, hash_buckets, st.find_hashes);
                remember_bucket(occ_reuse, hash_buckets, bfind, true);
                if ((occ_reuse & low_mask(hash_buckets)) != before_low) std::exit(2);
            }

            const int b0 = record_bucket(k, occ_base, hash_buckets, false,
                                         st.record_hashes_baseline);
            const std::uint64_t before_low = occ_reuse & low_mask(hash_buckets);
            const auto before_hashes = st.record_hashes_reuse;
            const int b1 = record_bucket(k, occ_reuse, hash_buckets, true,
                                         st.record_hashes_reuse);
            if (b0 != b1) std::exit(3);
            if (i > 0 && hash_buckets < 64 && st.record_hashes_reuse == before_hashes)
                ++st.memo_hits;

            // record() consumes the memo but does not rewrite it. It only sets
            // the actual occupancy bit for this bucket.
            occ_reuse |= 1ULL << b1;
            occ_base |= 1ULL << b0;
            if ((occ_reuse & low_mask(hash_buckets)) !=
                (occ_base & low_mask(hash_buckets))) std::exit(4);
            if ((before_low & ~(1ULL << b1)) !=
                ((occ_reuse & low_mask(hash_buckets)) & ~(1ULL << b1))) std::exit(5);
            ++st.records;

            // Hits hash and return without touching memo metadata. Exercise
            // interleavings to prove stale miss metadata is harmless because a
            // later miss always overwrites it before any later record.
            if (i > 0 && (rng() & 3ULL) == 0) {
                const std::uint64_t meta_before =
                    hash_buckets == 64 ? 0 : (occ_reuse >> hash_buckets) & 0x3fULL;
                const auto hit = keys[std::size_t(rng() % keys.size())];
                (void)hash_bucket(hit, hash_buckets, st.find_hashes);
                const std::uint64_t meta_after =
                    hash_buckets == 64 ? 0 : (occ_reuse >> hash_buckets) & 0x3fULL;
                if (meta_before != meta_after) std::exit(11);
                ++st.hit_queries;
            }
        }
        ++st.components;
    }
    return st;
}

} // namespace

int main() {
    for (int hash_buckets : {16, 32, 64}) {
        const Stats st = prove_config(
            hash_buckets, 0x6275636b65747265ULL ^ std::uint64_t(hash_buckets));
        if (!st.records || st.record_hashes_baseline != st.records || !st.hit_queries)
            return 6;
        if (hash_buckets < 64) {
            if (!st.memo_hits) return 7;
            if (!(st.record_hashes_reuse < st.record_hashes_baseline)) return 8;
        } else {
            if (st.memo_hits != 0) return 9;
            if (st.record_hashes_reuse != st.record_hashes_baseline) return 10;
        }
        std::cout << "hash_buckets=" << hash_buckets
                  << " components=" << st.components
                  << " records=" << st.records
                  << " find_hashes=" << st.find_hashes
                  << " record_hashes_baseline=" << st.record_hashes_baseline
                  << " record_hashes_reuse=" << st.record_hashes_reuse
                  << " memo_hits=" << st.memo_hits
                  << " hit_queries=" << st.hit_queries
                  << " hit_memo_writes=0"
                  << " shared_bytes_added=0 occupancy_low_bits_exact=1 record_bucket_exact=1\n";
    }
    std::cout << "gridfp-runtime-find-index-bucket-reuse-proof OK"
              << " hash_bucket_configs=3 max_pairs=20"
              << " memo_storage=unused_occupancy_high_bits"
              << " memo_update=miss_only hit_memo_writes=0"
              << " hash64_fallback=rehash shared_bytes_added=0 exact=1\n";
    return 0;
}
