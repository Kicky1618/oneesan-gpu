#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Query a sparse sample of resident pages without moving them. This is a
// diagnostic only: move_pages(pid=0, nodes=nullptr, flags=0) asks the kernel for
// each page's NUMA node. Negative per-page status values are reported rather
// than treated as fatal because lazy anonymous mappings can legitimately have
// not-yet-instantiated samples.
struct RamstreamNumaSampleResult {
    size_t samples = 0;
    size_t success = 0;
    size_t requested_spacing_bytes = 0;
    size_t actual_spacing_bytes = 0;
    int syscall_errno = 0;
    std::map<int, size_t> nodes;
    std::map<int, size_t> errors;
};

static inline size_t ramstream_round_up(size_t x, size_t a) {
    return a ? ((x + a - 1) / a) * a : x;
}

static RamstreamNumaSampleResult ramstream_query_numa_pages(
    void* base, size_t bytes, size_t requested_spacing_bytes,
    size_t max_samples = 32768
) {
    RamstreamNumaSampleResult out;
    out.requested_spacing_bytes = requested_spacing_bytes;
    if (!base || !bytes || !requested_spacing_bytes || !max_samples) return out;

#if defined(__linux__) && defined(SYS_move_pages)
    long ps = ::sysconf(_SC_PAGESIZE);
    size_t page = ps > 0 ? size_t(ps) : size_t(4096);
    size_t spacing = ramstream_round_up(std::max(page, requested_spacing_bytes), page);
    if (bytes / spacing + 1 > max_samples) {
        size_t min_spacing = ramstream_round_up(
            (bytes + max_samples - 1) / max_samples, page);
        spacing = std::max(spacing, min_spacing);
    }
    out.actual_spacing_bytes = spacing;

    std::vector<void*> pages;
    pages.reserve(std::min(max_samples, bytes / spacing + 2));
    uintptr_t start = reinterpret_cast<uintptr_t>(base);
    uintptr_t aligned = (start + page - 1) & ~(uintptr_t(page) - 1);
    uintptr_t end = start + bytes;
    for (uintptr_t p = aligned; p < end && pages.size() < max_samples; p += spacing)
        pages.push_back(reinterpret_cast<void*>(p));
    if (pages.empty()) return out;

    std::vector<int> status(pages.size(), -999999);
    errno = 0;
    long rc = ::syscall(
        SYS_move_pages, 0, static_cast<unsigned long>(pages.size()),
        pages.data(), nullptr, status.data(), 0);
    if (rc < 0) {
        out.syscall_errno = errno;
        return out;
    }

    out.samples = pages.size();
    for (int s : status) {
        if (s >= 0) {
            ++out.success;
            ++out.nodes[s];
        } else {
            ++out.errors[s];
        }
    }
#else
    (void)max_samples;
    out.syscall_errno = ENOSYS;
#endif
    return out;
}

static void ramstream_print_numa_sample(
    const char* tag, const char* array_name, void* base, size_t bytes,
    double requested_spacing_mib
) {
    if (!(requested_spacing_mib > 0.0)) return;
    long double req = static_cast<long double>(requested_spacing_mib) * (1ULL << 20);
    size_t requested_bytes = req >= static_cast<long double>(SIZE_MAX)
        ? SIZE_MAX : size_t(req);
    if (!requested_bytes) requested_bytes = 4096;

    RamstreamNumaSampleResult r = ramstream_query_numa_pages(
        base, bytes, requested_bytes);
    std::cerr << "numa_sample tag=" << tag
              << " array=" << array_name
              << " bytes=" << bytes
              << " requested_spacing_mib=" << requested_spacing_mib
              << " actual_spacing_mib="
              << double(r.actual_spacing_bytes) / double(1ULL << 20)
              << " samples=" << r.samples
              << " success=" << r.success;
    if (r.syscall_errno)
        std::cerr << " syscall_errno=" << r.syscall_errno
                  << " syscall_error=" << std::strerror(r.syscall_errno);
    for (const auto& [node, count] : r.nodes)
        std::cerr << " N" << node << '=' << count;
    for (const auto& [error, count] : r.errors)
        std::cerr << " E" << error << '=' << count;
    std::cerr << '\n';
}
