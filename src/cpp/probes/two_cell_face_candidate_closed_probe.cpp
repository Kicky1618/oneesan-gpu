#pragma push_macro("main")
#undef main
#define main two_cell_component_device_probe_main_unused
#include "two_cell_component_device_probe.cpp"
#pragma pop_macro("main")

namespace {

using oneesan::twocell::PackedKey;
using oneesan::twocell::PackedWord;
using oneesan::twocell::Rank;
using oneesan::twocell::Symbol;
using oneesan::twocell::TC_L;
using oneesan::twocell::TC_N;
using oneesan::twocell::TC_R;

struct RawFaceList {
    PackedKey value[oneesan::twocell::kMaxWidth]{};
    int size = 0;
    int raw_candidates = 0;
    bool ok = true;
};

RawFaceList raw_face_sources(PackedWord label, int W, int i) {
    using namespace oneesan::twocell;
    RawFaceList out{};
    PackedWord collapsed{};
    if (!deep_collapse(label, i, collapsed)) {
        out.ok = false;
        return out;
    }

    out.value[out.size++] = make_state(1, label);
    out.value[out.size++] = make_state(0, insert_symbol(label, i, TC_N));
    out.value[out.size++] = make_state(0, insert_symbol(label, i + 1, TC_N));

    PackedWord central = insert_symbol(collapsed, i, TC_N);
    central = insert_symbol(central, i, TC_N);
    const int j = i + 1;
    PackedWord z = insert_symbol(central, j + 1, TC_N);
    PackedKey sparse{};
    if (!inverse_E(z, i, sparse) || !valid_word(z) ||
        !in_source_layout(sparse, W, i)) {
        out.ok = false;
        return out;
    }
    out.value[out.size++] = sparse;

    int h[oneesan::twocell::kMaxWidth + 2]{};
    h[0] = 1;
    for (int p = 0; p < W; ++p) {
        h[p + 1] = h[p];
        const Symbol c = symbol(z, p);
        if (c == TC_L) ++h[p + 1];
        else if (c == TC_R) --h[p + 1];
    }
    const int level = h[j];
    int face_left = j;
    while (face_left > 0 && h[face_left - 1] >= level) --face_left;
    int face_right = j + 2;
    while (face_right < W && h[face_right + 1] >= level) ++face_right;

    auto push_raw = [&](PackedWord w) {
        ++out.raw_candidates;
        if (!valid_word(w)) {
            out.ok = false;
            return;
        }
        PackedKey k{};
        if (!inverse_E(w, i, k) || !in_source_layout(k, W, i)) {
            out.ok = false;
            return;
        }
        for (int q = 0; q < out.size; ++q)
            if (equal(out.value[q], k)) {
                out.ok = false; // filtering would have been required
                return;
            }
        if (out.size >= oneesan::twocell::kMaxWidth) {
            out.ok = false;
            return;
        }
        out.value[out.size++] = k;
    };

    for (int p = 0; p < W; ++p) {
        if (symbol(z, p) != TC_L) continue;
        const int q = partner(z, p);
        if (q < 0 || p < face_left || q >= face_right || h[p] != level) continue;
        PackedWord w = z;
        bool candidate = true;
        if (q < j) {
            w = set_symbol(w, q, TC_L);
            w = set_symbol(w, j, TC_R);
            w = set_symbol(w, j + 1, TC_R);
        } else if (p > j + 1) {
            w = set_symbol(w, p, TC_R);
            w = set_symbol(w, j, TC_L);
            w = set_symbol(w, j + 1, TC_L);
        } else if (p < j && q > j + 1) {
            w = set_symbol(w, j, TC_R);
            w = set_symbol(w, j + 1, TC_L);
        } else {
            candidate = false;
        }
        if (candidate) push_raw(w);
    }

    if (face_left > 0) {
        const int p = face_left - 1;
        if (symbol(z, p) == TC_L) {
            const int q = partner(z, p);
            if (q == face_right && p < j && q > j + 1) {
                PackedWord w = set_symbol(set_symbol(z, j, TC_R), j + 1, TC_L);
                push_raw(w);
            }
        }
    }

    const int root = root_position(z);
    if (root >= 0 &&
        (level == 0 || (face_left == 0 && face_right < W && root == face_right))) {
        if (root < j) {
            PackedWord w = set_symbol(z, root, TC_L);
            w = set_symbol(w, j, TC_R);
            w = set_symbol(w, j + 1, TC_R);
            push_raw(w);
        } else if (root > j + 1) {
            PackedWord w = set_symbol(set_symbol(z, j, TC_R), j + 1, TC_L);
            push_raw(w);
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 5 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank deep = 0;
        Rank raw_candidates = 0;
        Rank ordered = 0;
        Rank max_raw = 0;
        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                const PackedWord label = device_word(u);
                PackedWord collapsed{};
                if (!oneesan::twocell::deep_collapse(label, i, collapsed)) continue;
                const RawFaceList raw = raw_face_sources(label, W, i);
                if (!raw.ok)
                    fail("raw face filtering required W=" + std::to_string(W) +
                         " i=" + std::to_string(i));
                const auto serial = oneesan::twocell::direct_component_sources(label, W, i);
                if (serial.overflow || serial.size != raw.size)
                    fail("raw face size mismatch");
                for (int q = 0; q < serial.size; ++q)
                    if (!oneesan::twocell::equal(serial.value[q], raw.value[q]))
                        fail("raw face order mismatch W=" + std::to_string(W));
                ++deep;
                ++ordered;
                raw_candidates += raw.raw_candidates;
                max_raw = std::max<Rank>(max_raw, raw.raw_candidates);
            }
        }
        std::cout << "W=" << W
                  << " deep_components=" << deep
                  << " raw_face_candidates=" << raw_candidates
                  << " max_raw_candidates=" << max_raw
                  << " all_full_words_valid=1"
                  << " all_sources_valid=1"
                  << " duplicate_candidates=0"
                  << " canonical_order=" << ordered
                  << " filtering_required=0 OK\n";
    }

    std::cout << "W=28_plan max_sources=17"
              << " candidate_valid_word_scans=0"
              << " source_valid_word_scans=0"
              << " candidate_dedup_searches=0"
              << " warp_compaction=ballot+popc\n";
    std::cout << "ALL_OK closed_face_candidate_generation=1\n";
    return 0;
}
