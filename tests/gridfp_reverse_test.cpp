#include "../src/common/gridfp_reverse.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

using namespace oneesan::gridfp;
static std::vector<MateID> states(int width) {
    std::vector<MateID> result;
    auto visit = [&](auto&& self, int pos, int height, MateID m) -> void {
        if (height > pos + 1) return;
        if (pos < 0) { if (!height) result.push_back(m); return; }
        self(self, pos - 1, height, m);
        if (height) self(self, pos - 1, height - 1, mset(m, pos, R));
        self(self, pos - 1, height + 1, mset(m, pos, L));
    };
    visit(visit, width - 1, 1, 0);
    return result;
}

int main() {
    // Regression against an invalid suffix-height pruning rule: LLLRRRRN
    // has peak four but one more row maps it into the final-row support.
    MateID witness=43348;
    bool is_blocked=false;
    for(int p=7;p>=1;--p){
        if(is_blocked){witness=blocked_exclude(witness,p);is_blocked=false;}
        else if(p==4||p==2){
            auto z=include_horizontal(witness,8,p);
            if(!z.valid)return 7;
            witness=z.mate;is_blocked=z.blocked;
        }
    }
    MateID terminal_support=include_horizontal(MateID(R)<<14,8,6).mate;
    if(is_blocked||witness!=terminal_support||witness!=25600)return 8;
    uint64_t checked = 0, edges = 0;
    for (int width = 3; width <= 12; ++width) {
        const auto main = states(width), blocked = states(width - 1);
        std::unordered_map<MateID,std::vector<MateID>> boundary;
        for(MateID m:main){auto z=include_horizontal(m,width,1);if(z.valid){if(z.blocked)return 9;boundary[z.mate].push_back(m);}}
        for(MateID t:main){
            std::vector<MateID> got;
            reverse_boundary_main_predecessors(t,width,[&](MateID m){got.push_back(m);});
            auto want=boundary[t];std::sort(got.begin(),got.end());std::sort(want.begin(),want.end());
            if(got!=want){std::cerr<<"Boundary mismatch width="<<width<<" target="<<t<<'\n';return 10;}
            ++checked;edges+=got.size();
        }
        for (int p = 2; p < width; ++p) {
            // Independent oracle: enumerate the FORWARD transition for every
            // legal source, then invert that relation including multiplicity.
            std::unordered_map<MateID, std::vector<MateID>> expected;
            for (MateID m : main) {
                auto z = include_horizontal(m, width, p);
                if (z.valid && z.blocked) expected[z.mate].push_back(m);
            }
            for (MateID b : blocked) {
                std::vector<MateID> sparse, scalar;
                reverse_block_predecessors<true>(b, width, p, [&](MateID m){ sparse.push_back(m); });
                reverse_block_predecessors<false>(b, width, p, [&](MateID m){ scalar.push_back(m); });
                if (sparse != scalar) return 1; // Also preserve summation order.
                auto want = expected[b];
                std::sort(sparse.begin(), sparse.end());
                std::sort(want.begin(), want.end());
                if (sparse != want) {
                    std::cerr << "Mismatch width=" << width << " p=" << p << " b=" << b << '\n';
                    return 2;
                }
                ++checked;
                edges += sparse.size();
            }
        }
    }
    // Width 28 boundary cases: long runs of empty positions, both directions.
    for (int r = 0; r < 27; ++r) {
        MateID b = mset(0, r, R);
        for (int p = 2; p < 28; ++p) {
            std::vector<MateID> a, bscan;
            reverse_block_predecessors<true>(b, 28, p, [&](MateID m){ a.push_back(m); });
            reverse_block_predecessors<false>(b, 28, p, [&](MateID m){ bscan.push_back(m); });
            if (a != bscan) return 3;
            for (MateID m : a) {
                auto z = include_horizontal(m, 28, p);
                if (!z.valid || !z.blocked || z.mate != b) return 4;
            }
        }
    }
    uint64_t ways[28][30]{};
    ways[0][0] = 1;
    for (int w = 1; w <= 27; ++w)
        for (int h = 0; h <= 27; ++h)
            ways[w][h] = ways[w-1][h] + (h ? ways[w-1][h-1] : 0) + ways[w-1][h+1];
    std::mt19937_64 rng(20260905);
    for (int sample = 0; sample < 10000; ++sample) {
        uint64_t rank = rng() % ways[27][1];
        MateID b = 0;
        int h = 1;
        for (int pos = 26; pos >= 0; --pos) {
            if (rank < ways[pos][h]) continue;
            rank -= ways[pos][h];
            if (h) {
                if (rank < ways[pos][h-1]) { b = mset(b, pos, R); --h; continue; }
                rank -= ways[pos][h-1];
            }
            b = mset(b, pos, L); ++h;
        }
        for (int p = 2; p < 28; ++p) {
            std::vector<MateID> a, scalar;
            reverse_block_predecessors<true>(b, 28, p, [&](MateID m){ a.push_back(m); });
            reverse_block_predecessors<false>(b, 28, p, [&](MateID m){ scalar.push_back(m); });
            if (a != scalar) return 5;
            for (MateID m : a) {
                auto z = include_horizontal(m, 28, p);
                if (!z.valid || !z.blocked || z.mate != b) return 6;
            }
        }
    }
    std::cout << "PASS " << checked << " targets, " << edges
              << " predecessor edges; widths 3..12, width-28 boundaries and 260000 sampled targets\n";
}
