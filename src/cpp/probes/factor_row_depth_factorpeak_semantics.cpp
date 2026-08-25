#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using U64 = std::uint64_t;

struct SegmentPeak {
    int end_h = 0;
    int peak = 0;
};

static std::vector<MateID> enumerate_states(int width) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        if (pos < 0) {
            if (h == 0) out.push_back(m);
            return;
        }
        if (h < 0 || h > pos + 1) return;
        self(self, pos - 1, h, m);
        if (h > 0)
            self(self, pos - 1, h - 1, m | (MateID(R) << (2 * pos)));
        self(self, pos - 1, h + 1, m | (MateID(L) << (2 * pos)));
    };
    rec(rec, width - 1, 1, 0);
    return out;
}

static SegmentPeak segment_peak(MateID m, int hi, int lo, int start_h) {
    int h = start_h;
    int peak = h;
    for (int p = hi; p >= lo; --p) {
        const MateValue v = mget(m, p);
        if (v == R) --h;
        else if (v == L) {
            ++h;
            peak = std::max(peak, h);
        }
    }
    return {h, peak};
}

static int full_peak(MateID m, int width) {
    return segment_peak(m, width - 1, 0, 1).peak;
}

static int factor_peak_main(MateID m, int W, int low) {
    const int high = W - 1 - low;
    const SegmentPeak hp = segment_peak(m, W - 1, low + 1, 1);
    if (W - 1 - (low + 1) + 1 != high) std::exit(20);
    int hs = hp.end_h;
    int center_peak = hp.peak;
    const MateValue c = mget(m, low);
    if (c == R) --hs;
    else if (c == L) {
        ++hs;
        center_peak = std::max(center_peak, hs);
    }
    const SegmentPeak lp = segment_peak(m, low - 1, 0, hs);
    return std::max(center_peak, lp.peak);
}

static int factor_peak_block(MateID m, int W, int low) {
    const int high = W - 1 - low;
    const SegmentPeak hp = segment_peak(m, W - 2, low, 1);
    if (W - 2 - low + 1 != high) std::exit(21);
    const SegmentPeak lp = segment_peak(m, low - 1, 0, hp.end_h);
    return std::max(hp.peak, lp.peak);
}

static std::uint32_t low_code(MateID m, int low) {
    return std::uint32_t(m & ((MateID(1) << (2 * low)) - 1));
}

static void verify_width(int W, int low) {
    const auto main_states = enumerate_states(W);
    const auto block_states = enumerate_states(W - 1);

    // A LOW code can only close from one starting height: its net L-R delta
    // fixes hs uniquely. This is why one byte per stored LOW code is sufficient
    // even though the peak is measured relative to hs.
    std::unordered_map<std::uint32_t, int> low_start;
    low_start.reserve(main_states.size() / 4 + 1);

    for (MateID m : main_states) {
        const int a = full_peak(m, W);
        const int b = factor_peak_main(m, W, low);
        if (a != b) {
            std::cerr << "MAIN factor peak mismatch W=" << W
                      << " low=" << low << " full=" << a << " factor=" << b << '\n';
            std::exit(10);
        }

        const SegmentPeak hp = segment_peak(m, W - 1, low + 1, 1);
        int hs = hp.end_h;
        const MateValue c = mget(m, low);
        if (c == R) --hs;
        else if (c == L) ++hs;
        const std::uint32_t lc = low_code(m, low);
        const auto it = low_start.find(lc);
        if (it == low_start.end()) low_start.emplace(lc, hs);
        else if (it->second != hs) {
            std::cerr << "LOW code has non-unique start height W=" << W << '\n';
            std::exit(11);
        }
    }

    for (MateID m : block_states) {
        const int a = full_peak(m, W - 1);
        const int b = factor_peak_block(m, W, low);
        if (a != b) {
            std::cerr << "BLOCK factor peak mismatch W=" << W
                      << " low=" << low << " full=" << a << " factor=" << b << '\n';
            std::exit(12);
        }
    }

    std::cout << "row-depth-factorpeak-semantics OK W=" << W
              << " low=" << low
              << " main=" << main_states.size()
              << " blocked=" << block_states.size()
              << " low_codes_seen=" << low_start.size() << '\n';
}

int main(int argc, char** argv) {
    const int max_w = argc > 1 ? std::atoi(argv[1]) : 12;
    if (max_w < 6 || max_w > 14) return 1;
    for (int W = 6; W <= max_w; W += 2) verify_width(W, W / 2);
    return 0;
}
