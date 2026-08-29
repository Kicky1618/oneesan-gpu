#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

static uint32_t effective_batch(uint32_t total, uint32_t blocks, uint32_t max_batch, uint32_t waves) {
    if (waves == 0) return max_batch;
    const uint64_t denom = uint64_t(blocks) * uint64_t(waves);
    const uint32_t cap = uint32_t((uint64_t(total) + denom - 1u) / denom);
    uint32_t b = 1;
    if (max_batch >= 2 && cap >= 2) b = 2;
    if (max_batch >= 4 && cap >= 4) b = 4;
    if (max_batch >= 8 && cap >= 8) b = 8;
    if (max_batch >= 16 && cap >= 16) b = 16;
    return b;
}

struct Cta {
    uint32_t lease_base = 0;
    uint32_t lease_pos = 0;
    uint32_t batch = 1;
    bool done = false;
};

// CPU model of p10dc_orbitcta_flat_dynamic_pipe2_next_k(). Queue atomics are
// serialized by the caller in an arbitrary CTA interleaving. This is enough to
// prove the only property the queue scheduler owns: each global orbit id is
// handed to exactly one CTA, including ids whose prepare later yields valid=0.
static uint32_t next_k(Cta& c, uint32_t total, uint32_t& queue) {
    if (c.lease_pos < c.batch) {
        const uint32_t k = c.lease_base + c.lease_pos++;
        return k < total ? k : UINT32_MAX;
    }
    c.lease_base = queue;
    queue += c.batch;
    c.lease_pos = 0;
    if (c.lease_base >= total) return UINT32_MAX;
    c.lease_pos = 1;
    return c.lease_base;
}

static void run_case(uint32_t total, uint32_t blocks, uint32_t max_batch, uint32_t waves, uint64_t seed) {
    const uint32_t batch = effective_batch(total, blocks, max_batch, waves);
    std::vector<Cta> ctas(blocks);
    for (auto& c : ctas) c.batch = batch;
    std::vector<uint32_t> visit(total, 0), execute(total, 0);

    // Some orbit descriptors are intentionally non-executable after prepare.
    // A correct pipe must still consume the id and continue its lease.
    std::vector<uint8_t> valid(total, 1);
    for (uint32_t k = 0; k < total; ++k) {
        const uint64_t x = (uint64_t(k) * 0x9e3779b97f4a7c15ULL) ^ seed;
        valid[k] = uint8_t(((x >> (k & 31u)) & 7u) != 0u);  // ~1/8 invalid.
    }

    uint32_t queue = 0;
    std::mt19937_64 rng(seed ^ (uint64_t(total) << 32) ^ blocks ^ (uint64_t(batch) << 16));
    uint64_t steps = 0;
    while (true) {
        std::vector<uint32_t> live;
        for (uint32_t i = 0; i < blocks; ++i) if (!ctas[i].done) live.push_back(i);
        if (live.empty()) break;
        std::shuffle(live.begin(), live.end(), rng);
        for (uint32_t i : live) {
            Cta& c = ctas[i];
            const uint32_t k = next_k(c, total, queue);
            if (k == UINT32_MAX) {
                c.done = true;
                continue;
            }
            if (k >= total) std::abort();
            ++visit[k];
            if (valid[k]) ++execute[k];
            if (++steps > uint64_t(total + blocks) * 8u + 1024u) {
                std::cerr << "nonterminating schedule\n";
                std::exit(2);
            }
        }
    }

    for (uint32_t k = 0; k < total; ++k) {
        if (visit[k] != 1u) {
            std::cerr << "coverage mismatch total=" << total << " blocks=" << blocks
                      << " max_batch=" << max_batch << " waves=" << waves
                      << " effective=" << batch << " k=" << k << " visit=" << visit[k] << '\n';
            std::exit(3);
        }
        const uint32_t expect_exec = valid[k] ? 1u : 0u;
        if (execute[k] != expect_exec) {
            std::cerr << "valid-skip mismatch k=" << k << " valid=" << int(valid[k])
                      << " execute=" << execute[k] << '\n';
            std::exit(4);
        }
    }
}

int main() {
    constexpr std::array<uint32_t, 5> batches{1, 2, 4, 8, 16};
    constexpr std::array<uint32_t, 4> waves{0, 1, 2, 4};
    uint64_t cases = 0;
    for (uint32_t total = 0; total <= 513; ++total) {
        for (uint32_t blocks = 1; blocks <= 32; ++blocks) {
            for (uint32_t b : batches) {
                for (uint32_t w : waves) {
                    if (w != 0 && b == 1) continue;
                    run_case(total, blocks, b, w,
                             0xd1b54a32d192ed03ULL ^ uint64_t(cases) * 0x94d049bb133111ebULL);
                    ++cases;
                }
            }
        }
    }
    std::cout << "gridfp_dynamic_pipe2_schedule_probe OK cases=" << cases
              << " invalid_orbits=covered_not_executed adaptive_waves=0,1,2,4\n";
}
