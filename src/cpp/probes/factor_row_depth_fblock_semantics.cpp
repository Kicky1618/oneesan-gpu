#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using Count = std::uint32_t;
using Map = std::unordered_map<MateID, Count>;
static constexpr Count MOD = 4294967291u;

static void add(Map& a, MateID m, Count x) {
    if (!x) return;
    auto it = a.find(m);
    if (it == a.end()) {
        a.emplace(m, x);
        return;
    }
    const std::uint64_t z = std::uint64_t(it->second) + x;
    const Count y = Count(z >= MOD ? z - MOD : z);
    if (y) it->second = y;
    else a.erase(it);
}

static void step(Map& main, Map& block, int W, int p) {
    Map nm = main; // excluded MAIN branch
    Map nb;
    nm.reserve(main.size() * 2 + block.size() + 1);
    nb.reserve(main.size() / 2 + 1);
    for (const auto& kv : main) {
        const IncludeResult z = include_horizontal(kv.first, W, p);
        if (!z.valid) continue;
        if (z.blocked) add(nb, z.mate, kv.second);
        else add(nm, z.mate, kv.second);
    }
    for (const auto& kv : block)
        add(nm, blocked_exclude(kv.first, p), kv.second);
    main.swap(nm);
    block.swap(nb);
}

static int main_block_height(MateID m, int W, int low) {
    int h = 1;
    for (int p = W - 1; p >= low + 1; --p) {
        const MateValue v = mget(m, p);
        if (v == R) --h;
        else if (v == L) ++h;
    }
    const int he = h;
    const MateValue c = mget(m, low);
    if (c == R) --h;
    else if (c == L) ++h;
    return std::max(he, h);
}

static int blocked_block_height(MateID m, int W, int low) {
    int h = 1;
    for (int p = W - 2; p >= low; --p) {
        const MateValue v = mget(m, p);
        if (v == R) --h;
        else if (v == L) ++h;
    }
    return h;
}

static void filter_gather(Map& main, int W, int low, int cap) {
    for (auto it = main.begin(); it != main.end(); ) {
        if (main_block_height(it->first, W, low) > cap) it = main.erase(it);
        else ++it;
    }
}

static void filter_scatter(Map& main, Map& block, int W, int low, int cap) {
    for (auto it = main.begin(); it != main.end(); ) {
        if (main_block_height(it->first, W, low) > cap) it = main.erase(it);
        else ++it;
    }
    for (auto it = block.begin(); it != block.end(); ) {
        if (blocked_block_height(it->first, W, low) > cap) it = block.erase(it);
        else ++it;
    }
}

static void verify(int W, int low) {
    Map ref_main, ref_block, got_main, got_block;
    const MateID init = MateID(R) << (2 * (W - 1));
    ref_main.emplace(init, 1);
    got_main = ref_main;

    for (int row = 1; row <= W; ++row) {
        const int gather_cap = std::max(1, row - 1);
        filter_gather(got_main, W, low, gather_cap);
        got_block.clear(); // v0.13/v0.14 lazy row-boundary BLOCKED input
        if (got_main != ref_main || !ref_block.empty()) {
            std::cerr << "FBlock gather removed live input W=" << W
                      << " row=" << row << '\n';
            std::exit(10);
        }

        for (int p = W - 1; p >= low + 1; --p) {
            step(ref_main, ref_block, W, p);
            step(got_main, got_block, W, p);
        }
        filter_scatter(got_main, got_block, W, low, row);
        if (got_main != ref_main || got_block != ref_block) {
            std::cerr << "FBlock scatter removed live output W=" << W
                      << " row=" << row << '\n';
            std::exit(11);
        }

        for (int p = low; p >= 1; --p) {
            step(ref_main, ref_block, W, p);
            step(got_main, got_block, W, p);
        }
        if (got_main != ref_main || got_block != ref_block || !ref_block.empty()) {
            std::cerr << "row mismatch after FBlock I/O filtering W=" << W
                      << " row=" << row << '\n';
            std::exit(12);
        }
    }
    std::cout << "row-depth-fblock-semantics OK W=" << W << " low=" << low
              << " final_nonzero=" << ref_main.size() << '\n';
}

int main(int argc, char** argv) {
    const int max_w = argc > 1 ? std::atoi(argv[1]) : 12;
    if (max_w < 6 || max_w > 14) return 1;
    for (int W = 6; W <= max_w; W += 2) verify(W, W / 2);
    return 0;
}
