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

static std::vector<int> cpu_high_parse_cpu_list(const char* text) {
    std::vector<int> cpus;
    if (!text || !*text) return cpus;

    const char* p = text;
    while (*p) {
        while (*p && (std::isspace(static_cast<unsigned char>(*p)) || *p == ',')) ++p;
        if (!*p) break;

        char* end = nullptr;
        long a = std::strtol(p, &end, 10);
        if (end == p || a < 0 || a >= CPU_SETSIZE) {
            std::cerr << "invalid CPU_HIGH_CPU_LIST near: " << p << '\n';
            std::exit(131);
        }
        p = end;
        long b = a;
        if (*p == '-') {
            ++p;
            b = std::strtol(p, &end, 10);
            if (end == p || b < a || b >= CPU_SETSIZE) {
                std::cerr << "invalid CPU_HIGH_CPU_LIST range near: " << p << '\n';
                std::exit(131);
            }
            p = end;
        }

        while (*p && std::isspace(static_cast<unsigned char>(*p))) ++p;
        if (*p && *p != ',') {
            std::cerr << "invalid CPU_HIGH_CPU_LIST separator near: " << p << '\n';
            std::exit(131);
        }

        for (long cpu = a; cpu <= b; ++cpu) cpus.push_back(int(cpu));
        if (*p == ',') ++p;
    }

    if (cpus.empty()) {
        std::cerr << "CPU_HIGH_CPU_LIST produced an empty CPU set\n";
        std::exit(131);
    }
    return cpus;
}

static const std::vector<int>& cpu_high_affinity_cpus() {
    static const std::vector<int> cpus = [] {
        const char* text = std::getenv("CPU_HIGH_CPU_LIST");
        auto out = cpu_high_parse_cpu_list(text);
        if (!out.empty()) {
            std::cerr << "cpu_high_affinity cpus=" << out.size()
                      << " first=" << out.front()
                      << " last=" << out.back() << '\n';
        }
        return out;
    }();
    return cpus;
}

static void cpu_high_bind_worker(int worker_index) {
    const auto& cpus = cpu_high_affinity_cpus();
    if (cpus.empty()) return;

    int cpu = cpus[size_t(worker_index) % cpus.size()];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    if (rc != 0) {
        std::cerr << "pthread_setaffinity_np failed worker=" << worker_index
                  << " cpu=" << cpu << " error=" << std::strerror(rc) << '\n';
        std::exit(132);
    }
}
