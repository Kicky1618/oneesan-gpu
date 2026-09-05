#pragma once

#include <pthread.h>
#include <sched.h>

#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static std::vector<int> cpu_parse_cpu_list(const char* name, const char* text) {
    std::vector<int> cpus;
    if (!text || !*text) return cpus;

    const char* p = text;
    while (*p) {
        while (*p && (std::isspace(static_cast<unsigned char>(*p)) || *p == ',')) ++p;
        if (!*p) break;

        char* end = nullptr;
        long a = std::strtol(p, &end, 10);
        if (end == p || a < 0 || a >= CPU_SETSIZE) {
            std::cerr << "invalid " << name << " near: " << p << '\n';
            std::exit(131);
        }
        p = end;
        long b = a;
        if (*p == '-') {
            ++p;
            b = std::strtol(p, &end, 10);
            if (end == p || b < a || b >= CPU_SETSIZE) {
                std::cerr << "invalid " << name << " range near: " << p << '\n';
                std::exit(131);
            }
            p = end;
        }

        while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
        if (*p && *p != ',') {
            std::cerr << "invalid " << name << " separator near: " << p << '\n';
            std::exit(131);
        }

        for (long cpu = a; cpu <= b; ++cpu) cpus.push_back(int(cpu));
        if (*p == ',') ++p;
    }

    if (cpus.empty()) {
        std::cerr << name << " produced an empty CPU set\n";
        std::exit(131);
    }
    return cpus;
}

static std::vector<int> cpu_high_parse_cpu_list(const char* text) {
    return cpu_parse_cpu_list("CPU_HIGH_CPU_LIST", text);
}
static std::vector<int> cpu_low_parse_cpu_list(const char* text) {
    return cpu_parse_cpu_list("CPU_LOW_CPU_LIST", text);
}

static const std::vector<int>& cpu_high_affinity_cpus() {
    static const std::vector<int> cpus = [] {
        auto out = cpu_high_parse_cpu_list(std::getenv("CPU_HIGH_CPU_LIST"));
        if (!out.empty()) {
            std::cerr << "cpu_high_affinity cpus=" << out.size()
                      << " first=" << out.front()
                      << " last=" << out.back() << '\n';
        }
        return out;
    }();
    return cpus;
}

static const std::vector<int>& cpu_low_affinity_cpus() {
    static const std::vector<int> cpus = [] {
        auto out = cpu_low_parse_cpu_list(std::getenv("CPU_LOW_CPU_LIST"));
        if (!out.empty()) {
            std::cerr << "cpu_low_affinity cpus=" << out.size()
                      << " first=" << out.front()
                      << " last=" << out.back() << '\n';
        }
        return out;
    }();
    return cpus;
}

static void cpu_bind_worker(
    const char* tag, int worker_index, const std::vector<int>& cpus
) {
    if (cpus.empty()) return;
    int cpu = cpus[size_t(worker_index) % cpus.size()];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        std::cerr << "pthread_setaffinity_np failed tag=" << tag
                  << " worker=" << worker_index
                  << " cpu=" << cpu << " error=" << std::strerror(rc) << '\n';
        std::exit(132);
    }
}

static void cpu_high_bind_worker(int worker_index) {
    cpu_bind_worker("high", worker_index, cpu_high_affinity_cpus());
}
static void cpu_low_bind_worker(int worker_index) {
    cpu_bind_worker("low", worker_index, cpu_low_affinity_cpus());
}
