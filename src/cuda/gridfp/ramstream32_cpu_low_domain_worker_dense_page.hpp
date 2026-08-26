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
// counts still use 64-bit page IDs in a hash table.  Every page that can appear
// in the search is already present in one of the ordered worker boundaries, so
// we can enumerate that finite universe once, map it to dense uint32_t IDs, and
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

    size_t bytes() const {
        size_t z = universe_2m.size() * sizeof(uint64_t)
            + universe_4k.size() * sizeof(uint64_t)
            + boundary.size() * sizeof(CpuLowWorkerDenseBoundarySignature);
        for (const auto& s : boundary)
            z += (s.pages_2m.size() + s.pages_4k.size()) * sizeof(uint32_t);
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
    std::sort(out.universe_2m.begin(), out.universe_2m.end());
    out.universe_2m.erase(
        std::unique(out.universe_2m.begin(), out.universe_2m.end()),
        out.universe_2m.end());
    std::sort(out.universe_4k.begin(), out.universe_4k.end());
    out.universe_4k.erase(
        std::unique(out.universe_4k.begin(), out.universe_4k.end()),
        out.universe_4k.end());

    for (size_t k = 0; k < raw.size(); ++k) {
        auto& d = out.boundary[k];
        d.pages_2m.reserve(raw[k].pages_2m.size());
        d.pages_4k.reserve(raw[k].pages_4k.size());
        for (uint64_t p : raw[k].pages_2m)
            d.pages_2m.push_back(cpu_low_worker_dense_id(out.universe_2m, p));
        for (uint64_t p : raw[k].pages_4k)
            d.pages_4k.push_back(cpu_low_worker_dense_id(out.universe_4k, p));
        if (!std::is_sorted(d.pages_2m.begin(), d.pages_2m.end())
            || !std::is_sorted(d.pages_4k.begin(), d.pages_4k.end())) {
            std::cerr << "cpu LOW dense boundary signature lost sort order\n";
            std::exit(216);
        }
    }
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
