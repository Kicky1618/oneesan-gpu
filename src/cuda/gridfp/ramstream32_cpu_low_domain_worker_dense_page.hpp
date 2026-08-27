#pragma once

#include "ramstream32_cpu_low_domain_page_global.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

// Research-only v5.30 dense page-ID substrate for exact worker-boundary search.
//
// v5.29 removed candidate-local unordered_maps, but persistent page reference
// counts still use 64-bit page IDs in a hash table. Every page that can appear
// in the search is already present in one of the ordered worker boundaries, so
// we enumerate that finite universe once, map it to dense uint32_t IDs, and
// use ordinary vectors for persistent refcounts and candidate deltas.

struct CpuLowWorkerDenseBoundarySignature {
    std::vector<uint32_t> pages_2m;
    std::vector<uint32_t> pages_4k;
};

struct CpuLowWorkerDensePageIndex {
    std::vector<uint64_t> universe_2m;
    std::vector<uint64_t> universe_4k;
    std::vector<CpuLowWorkerDenseBoundarySignature> boundary;
    double build_s = 0.0;

    // Logical payload bytes, preserving the original v5.30 accounting.
    size_t bytes() const {
        size_t z = universe_2m.size() * sizeof(uint64_t)
            + universe_4k.size() * sizeof(uint64_t)
            + boundary.size() * sizeof(CpuLowWorkerDenseBoundarySignature);
        for (const auto& s : boundary)
            z += (s.pages_2m.size() + s.pages_4k.size()) * sizeof(uint32_t);
        return z;
    }

    // Vector-owned capacity bytes. This is a better lower bound for retained
    // heap storage than bytes(), especially after sort/unique compaction.
    size_t reserved_bytes() const {
        size_t z = universe_2m.capacity() * sizeof(uint64_t)
            + universe_4k.capacity() * sizeof(uint64_t)
            + boundary.capacity() * sizeof(CpuLowWorkerDenseBoundarySignature);
        for (const auto& s : boundary)
            z += (s.pages_2m.capacity() + s.pages_4k.capacity()) * sizeof(uint32_t);
        return z;
    }
};

struct CpuLowWorkerDenseDelta {
    std::vector<std::pair<uint32_t,int>> entries;
};

static uint32_t cpu_low_worker_dense_id(
    const std::vector<uint64_t>& universe, uint64_t page
) {
    auto it = std::lower_bound(universe.begin(), universe.end(), page);
    if (it == universe.end() || *it != page) {
        std::cerr << "cpu LOW dense page universe lookup failed\n";
        std::exit(214);
    }
    size_t id = size_t(it - universe.begin());
    if (id > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "cpu LOW dense page universe exceeds uint32 IDs\n";
        std::exit(215);
    }
    return uint32_t(id);
}

static void cpu_low_worker_dense_unique_compact(std::vector<uint64_t>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    v.shrink_to_fit();
}

static void cpu_low_worker_dense_encode_signature(
    CpuLowWorkerDenseBoundarySignature& dst,
    const CpuLowDomainGlobalPageSignature& src,
    const std::vector<uint64_t>& universe_2m,
    const std::vector<uint64_t>& universe_4k
) {
    dst.pages_2m.reserve(src.pages_2m.size());
    dst.pages_4k.reserve(src.pages_4k.size());
    for (uint64_t p : src.pages_2m)
        dst.pages_2m.push_back(cpu_low_worker_dense_id(universe_2m, p));
    for (uint64_t p : src.pages_4k)
        dst.pages_4k.push_back(cpu_low_worker_dense_id(universe_4k, p));
    if (!std::is_sorted(dst.pages_2m.begin(), dst.pages_2m.end())
        || !std::is_sorted(dst.pages_4k.begin(), dst.pages_4k.end())) {
        std::cerr << "cpu LOW dense boundary signature lost sort order\n";
        std::exit(216);
    }
}

static CpuLowWorkerDensePageIndex cpu_low_build_worker_dense_page_index_from_raw(
    const std::vector<CpuLowDomainGlobalPageSignature>& raw
) {
    CpuLowWorkerDensePageIndex out;
    auto t0 = std::chrono::steady_clock::now();
    out.boundary.resize(raw.size());

    for (const auto& s : raw) {
        out.universe_2m.insert(
            out.universe_2m.end(), s.pages_2m.begin(), s.pages_2m.end());
        out.universe_4k.insert(
            out.universe_4k.end(), s.pages_4k.begin(), s.pages_4k.end());
    }
    cpu_low_worker_dense_unique_compact(out.universe_2m);
    cpu_low_worker_dense_unique_compact(out.universe_4k);

    for (size_t k = 0; k < raw.size(); ++k)
        cpu_low_worker_dense_encode_signature(
            out.boundary[k], raw[k], out.universe_2m, out.universe_4k);

    out.build_s = ram_seconds_since(t0);
    return out;
}

static CpuLowWorkerDensePageIndex cpu_low_build_worker_dense_page_index(
    const std::vector<CpuLowStaticJobCost>& ordered,
    const StorageLayout& layout,
    const StorageFactorHost& storage,
    const CpuLowDomainPageMaskIndex& mask_index
) {
    std::vector<CpuLowDomainGlobalPageSignature> raw(ordered.size() + 1);
    for (size_t boundary = 1; boundary < ordered.size(); ++boundary) {
        raw[boundary] = cpu_low_domain_boundary_page_signature(
            layout, storage, mask_index, ordered[boundary].mask);
    }
    return cpu_low_build_worker_dense_page_index_from_raw(raw);
}

struct CpuLowWorkerDensePageBitmap {
    std::vector<uint64_t> main_2m;
    std::vector<uint64_t> block_2m;
    std::vector<uint64_t> main_4k;
    std::vector<uint64_t> block_4k;
};

static size_t cpu_low_worker_dense_bitmap_words(Code elems, uint64_t page_bytes) {
    if (!page_bytes
        || elems > Code(std::numeric_limits<uint64_t>::max() / sizeof(Count))) {
        std::cerr << "cpu LOW dense bitmap byte size overflow\n";
        std::exit(322);
    }
    uint64_t bytes = uint64_t(elems) * sizeof(Count);
    uint64_t pages = bytes ? 1 + (bytes - 1) / page_bytes : 0;
    uint64_t words = pages / 64 + uint64_t((pages & 63u) != 0);
    if (words > std::numeric_limits<size_t>::max()) {
        std::cerr << "cpu LOW dense bitmap word size overflow\n";
        std::exit(323);
    }
    return size_t(words);
}

static void cpu_low_worker_dense_bitmap_mark(
    std::vector<uint64_t>& bits, uint64_t page
) {
    uint64_t word = page >> 6;
    if (word >= bits.size()) {
        std::cerr << "cpu LOW dense bitmap page out of range\n";
        std::exit(324);
    }
    bits[size_t(word)] |= uint64_t(1) << (page & 63u);
}

static uint64_t cpu_low_worker_dense_bitmap_popcount(
    const std::vector<uint64_t>& bits
) {
    uint64_t z = 0;
    for (uint64_t w : bits) z += uint64_t(__builtin_popcountll(w));
    return z;
}

static void cpu_low_worker_dense_bitmap_append(
    std::vector<uint64_t>& universe,
    const std::vector<uint64_t>& bits,
    uint64_t tag
) {
    for (size_t wi = 0; wi < bits.size(); ++wi) {
        uint64_t w = bits[wi];
        while (w) {
            unsigned bit = unsigned(__builtin_ctzll(w));
            universe.push_back(tag | (uint64_t(wi) * 64 + bit));
            w &= w - 1;
        }
    }
}

static void cpu_low_worker_dense_bitmap_materialize(
    std::vector<uint64_t>& universe,
    const std::vector<uint64_t>& main_bits,
    const std::vector<uint64_t>& block_bits
) {
    constexpr uint64_t ARRAY_TAG = uint64_t(1) << 63;
    uint64_t count = cpu_low_worker_dense_bitmap_popcount(main_bits)
        + cpu_low_worker_dense_bitmap_popcount(block_bits);
    if (count > uint64_t(std::numeric_limits<uint32_t>::max()) + 1ull
        || count > std::numeric_limits<size_t>::max()) {
        std::cerr << "cpu LOW dense bitmap universe exceeds uint32 IDs\n";
        std::exit(325);
    }
    universe.reserve(size_t(count));
    cpu_low_worker_dense_bitmap_append(universe, main_bits, 0);
    cpu_low_worker_dense_bitmap_append(universe, block_bits, ARRAY_TAG);
    if (universe.size() != size_t(count)
        || !std::is_sorted(universe.begin(), universe.end())) {
        std::cerr << "cpu LOW dense bitmap universe materialization mismatch\n";
        std::exit(326);
    }
}

// Low-memory exact-workspace builder. The legacy v5.30 builder above retains
// every raw boundary signature while also materializing a duplicate 64-bit
// universe. The first streaming implementation removed retained signatures but
// still accumulated every boundary page membership as duplicate uint64_t IDs.
//
// Pass 1 now records page presence in fixed-size bitmaps keyed by authoritative
// array page number. For n=27 the 4 KiB bitmaps are bounded by the ~1.9 TiB
// authoritative arrays themselves (about 61 MiB total), independent of how
// many boundary memberships repeat. The sorted dense universes are then
// materialized once and the bitmaps are released before pass 2 recomputes one
// boundary at a time and immediately encodes it to uint32 dense IDs.
static CpuLowWorkerDensePageIndex cpu_low_build_worker_dense_page_index_streaming(
    const std::vector<CpuLowStaticJobCost>& ordered,
    const StorageLayout& layout,
    const StorageFactorHost& storage,
    const CpuLowDomainPageMaskIndex& mask_index
) {
    constexpr uint64_t PAGE4K = 4096ull;
    constexpr uint64_t PAGE2M = 2ull << 20;
    constexpr uint64_t ARRAY_TAG = uint64_t(1) << 63;

    CpuLowWorkerDensePageIndex out;
    auto t0 = std::chrono::steady_clock::now();

    {
        CpuLowWorkerDensePageBitmap bitmap;
        bitmap.main_2m.assign(
            cpu_low_worker_dense_bitmap_words(layout.main_size, PAGE2M), 0);
        bitmap.block_2m.assign(
            cpu_low_worker_dense_bitmap_words(layout.block_size, PAGE2M), 0);
        bitmap.main_4k.assign(
            cpu_low_worker_dense_bitmap_words(layout.main_size, PAGE4K), 0);
        bitmap.block_4k.assign(
            cpu_low_worker_dense_bitmap_words(layout.block_size, PAGE4K), 0);

        for (size_t boundary = 1; boundary < ordered.size(); ++boundary) {
            auto raw = cpu_low_domain_boundary_page_signature(
                layout, storage, mask_index, ordered[boundary].mask);
            for (uint64_t p : raw.pages_2m) {
                bool block = (p & ARRAY_TAG) != 0;
                uint64_t page = p & ~ARRAY_TAG;
                cpu_low_worker_dense_bitmap_mark(
                    block ? bitmap.block_2m : bitmap.main_2m, page);
            }
            for (uint64_t p : raw.pages_4k) {
                bool block = (p & ARRAY_TAG) != 0;
                uint64_t page = p & ~ARRAY_TAG;
                cpu_low_worker_dense_bitmap_mark(
                    block ? bitmap.block_4k : bitmap.main_4k, page);
            }
        }

        cpu_low_worker_dense_bitmap_materialize(
            out.universe_2m, bitmap.main_2m, bitmap.block_2m);
        cpu_low_worker_dense_bitmap_materialize(
            out.universe_4k, bitmap.main_4k, bitmap.block_4k);
    }

    // The bitmap storage is gone before the retained per-boundary dense-ID
    // vectors start growing, keeping the two construction peaks disjoint.
    out.boundary.resize(ordered.size() + 1);
    for (size_t boundary = 1; boundary < ordered.size(); ++boundary) {
        auto raw = cpu_low_domain_boundary_page_signature(
            layout, storage, mask_index, ordered[boundary].mask);
        cpu_low_worker_dense_encode_signature(
            out.boundary[boundary], raw, out.universe_2m, out.universe_4k);
    }

    out.build_s = ram_seconds_since(t0);
    return out;
}

static uint64_t cpu_low_worker_dense_ref_unique(
    const std::vector<uint32_t>& refs
) {
    uint64_t z = 0;
    for (uint32_t x : refs) z += x != 0;
    return z;
}

static void cpu_low_worker_dense_ref_add(
    std::vector<uint32_t>& refs,
    const std::vector<uint32_t>& pages,
    int delta
) {
    for (uint32_t id : pages) {
        if (id >= refs.size()) {
            std::cerr << "cpu LOW dense page ref ID out of range\n";
            std::exit(217);
        }
        uint32_t old = refs[id];
        if (delta < 0) {
            if (!old) {
                std::cerr << "cpu LOW dense page ref underflow\n";
                std::exit(218);
            }
            refs[id] = old - 1;
        } else {
            if (old == std::numeric_limits<uint32_t>::max()) {
                std::cerr << "cpu LOW dense page ref overflow\n";
                std::exit(219);
            }
            refs[id] = old + 1;
        }
    }
}

static void cpu_low_worker_dense_delta_add(
    CpuLowWorkerDenseDelta& d,
    const std::vector<uint32_t>& pages,
    int delta
) {
    for (uint32_t id : pages) d.entries.push_back({id, delta});
}

static void cpu_low_worker_dense_delta_normalize(CpuLowWorkerDenseDelta& d) {
    std::sort(d.entries.begin(), d.entries.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    size_t w = 0;
    for (size_t r = 0; r < d.entries.size();) {
        uint32_t id = d.entries[r].first;
        int sum = 0;
        do {
            sum += d.entries[r].second;
            ++r;
        } while (r < d.entries.size() && d.entries[r].first == id);
        if (sum) d.entries[w++] = {id, sum};
    }
    d.entries.resize(w);
}

static uint64_t cpu_low_worker_dense_unique_after_delta(
    const std::vector<uint32_t>& refs,
    uint64_t current_unique,
    const CpuLowWorkerDenseDelta& d
) {
    uint64_t z = current_unique;
    for (const auto& kv : d.entries) {
        if (kv.first >= refs.size()) {
            std::cerr << "cpu LOW dense candidate ID out of range\n";
            std::exit(220);
        }
        int64_t old = refs[kv.first];
        int64_t now = old + kv.second;
        if (now < 0 || now > std::numeric_limits<uint32_t>::max()) {
            std::cerr << "cpu LOW dense candidate ref range violation\n";
            std::exit(221);
        }
        if (old == 0 && now > 0) ++z;
        else if (old > 0 && now == 0) --z;
    }
    return z;
}

static void cpu_low_worker_dense_apply_delta(
    std::vector<uint32_t>& refs,
    uint64_t& unique,
    const CpuLowWorkerDenseDelta& d
) {
    unique = cpu_low_worker_dense_unique_after_delta(refs, unique, d);
    for (const auto& kv : d.entries) {
        int64_t now = int64_t(refs[kv.first]) + kv.second;
        if (now < 0 || now > std::numeric_limits<uint32_t>::max()) {
            std::cerr << "cpu LOW dense delta apply range violation\n";
            std::exit(222);
        }
        refs[kv.first] = uint32_t(now);
    }
}
