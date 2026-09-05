#pragma push_macro("main")
#undef main
#define main two_cell_direct_component_probe_main_unused
#include "two_cell_direct_component_probe.cpp"
#pragma pop_macro("main")

namespace {

int packed_prefix_height(const PackedWord& w, int boundary) {
    const std::uint32_t mask = low_mask(boundary);
    const int occupied = __builtin_popcount(w.support & mask);
    const int left = __builtin_popcount(w.left & mask);
    return 1 + 2 * left - occupied;
}

int packed_partner_first_return(const PackedWord& w, int p, int& rounds) {
    if (packed_symbol(w, p) != L) return -2;
    const int level = packed_prefix_height(w, p);
    rounds = 0;
    for (int q = p + 1; q < w.len; ++q) {
        ++rounds;
        if (packed_symbol(w, q) == R && packed_prefix_height(w, q + 1) == level)
            return q;
    }
    return -3;
}

int packed_root_by_height(const PackedWord& w) {
    for (int p = 0; p < w.len; ++p)
        if (packed_symbol(w, p) == R && packed_prefix_height(w, p + 1) == 0)
            return p;
    return -1;
}

SmallUnique<PackedWord, 32> packed_inverse_R_parallel_face(
    const PackedKey& raw,
    int j,
    int W,
    int& max_partner_rounds,
    int& candidate_lanes
) {
    SmallUnique<PackedWord, 32> out;
    if (raw.type != 'A') fail("parallel face raw type");
    if (packed_symbol(raw.w, j) != N) fail("parallel face local not N");

    PackedWord z = packed_insert(raw.w, j + 1, N);
    out.insert(z);

    const int level = packed_prefix_height(z, j);
    int face_left = 0;
    for (int q = 0; q < j; ++q)
        if (packed_prefix_height(z, q) < level) face_left = q + 1;

    int face_right = W;
    for (int q = j + 3; q <= W; ++q) {
        if (packed_prefix_height(z, q) < level) {
            face_right = q - 1;
            break;
        }
    }

    for (int p = 0; p < W; ++p) {
        if (packed_symbol(z, p) != L || p < face_left ||
            packed_prefix_height(z, p) != level)
            continue;

        int rounds = 0;
        const int q = packed_partner_first_return(z, p, rounds);
        max_partner_rounds = std::max(max_partner_rounds, rounds);
        if (q < 0 || q >= face_right) continue;
        ++candidate_lanes;

        PackedWord w = z;
        bool candidate = true;
        if (q < j) {
            w = packed_set(w, q, L);
            w = packed_set(w, j, R);
            w = packed_set(w, j + 1, R);
        } else if (p > j + 1) {
            w = packed_set(w, p, R);
            w = packed_set(w, j, L);
            w = packed_set(w, j + 1, L);
        } else if (p < j && q > j + 1) {
            w = packed_set(w, j, R);
            w = packed_set(w, j + 1, L);
        } else {
            candidate = false;
        }
        if (candidate && packed_valid(w)) out.insert(w);
    }

    if (face_left > 0) {
        const int p = face_left - 1;
        if (packed_symbol(z, p) == L) {
            int rounds = 0;
            const int q = packed_partner_first_return(z, p, rounds);
            max_partner_rounds = std::max(max_partner_rounds, rounds);
            if (q == face_right && p < j && q > j + 1) {
                PackedWord w = packed_set(packed_set(z, j, R), j + 1, L);
                if (packed_valid(w)) out.insert(w);
            }
        }
    }

    const int root = packed_root_by_height(z);
    if (root >= 0 &&
        (level == 0 || (face_left == 0 && face_right < W && root == face_right))) {
        if (root < j) {
            PackedWord w = packed_set(z, root, L);
            w = packed_set(w, j, R);
            w = packed_set(w, j + 1, R);
            if (packed_valid(w)) out.insert(w);
        } else if (root > j + 1) {
            PackedWord w = packed_set(packed_set(z, j, R), j + 1, L);
            if (packed_valid(w)) out.insert(w);
        }
    }
    return out;
}

SmallUnique<PackedKey, 32> packed_parallel_central_inverse(
    const PackedKey& central,
    int W,
    int i,
    int& max_partner_rounds,
    int& candidate_lanes
) {
    if (central.type != 'A') fail("parallel central type");
    const int j = i + 1;
    if (packed_symbol(central.w, j) != N)
        fail("parallel central active not N");

    SmallUnique<PackedKey, 32> out;
    const auto full = packed_inverse_R_parallel_face(
        central, j, W, max_partner_rounds, candidate_lanes);
    for (int q = 0; q < full.size; ++q) {
        PackedKey src;
        if (packed_inverse_E(full.value[q], i, src) &&
            packed_in_source_layout(src, W, i))
            out.insert(src);
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
        Rank deep_components = 0;
        Rank central_preimages = 0;
        Rank candidate_lanes = 0;
        Rank max_preimages = 0;
        int max_partner_rounds = 0;

        for (int i = 0; i <= W - 4; ++i) {
            for (const Word& u : words[W - 2]) {
                PackedWord collapsed;
                if (!packed_deep_collapse(pack_word(u), i, collapsed)) continue;

                PackedWord central_word = packed_insert(collapsed, i, N);
                central_word = packed_insert(central_word, i, N);
                const PackedKey central{'A', central_word};

                const auto expected = packed_inverse_K(central, W, i);
                int rounds = 0;
                int candidates = 0;
                const auto got = packed_parallel_central_inverse(
                    central, W, i, rounds, candidates);

                std::set<Key> a, b;
                for (int q = 0; q < expected.size; ++q)
                    a.insert(unpack_key(expected.value[q]));
                for (int q = 0; q < got.size; ++q)
                    b.insert(unpack_key(got.value[q]));
                if (a != b)
                    fail("parallel central inverse mismatch W=" + std::to_string(W) +
                         " i=" + std::to_string(i));

                max_partner_rounds = std::max(max_partner_rounds, rounds);
                candidate_lanes += candidates;
                central_preimages += got.size;
                max_preimages = std::max<Rank>(max_preimages, got.size);
                ++deep_components;
            }
        }

        std::cout << "W=" << W
                  << " deep_components=" << deep_components
                  << " central_preimages=" << central_preimages
                  << " avg_preimages="
                  << (deep_components ? double(central_preimages) / double(deep_components) : 0.0)
                  << " max_preimages=" << max_preimages
                  << " avg_candidate_lanes="
                  << (deep_components ? double(candidate_lanes) / double(deep_components) : 0.0)
                  << " max_partner_rounds=" << max_partner_rounds
                  << " height_array=0 partner_stack=0 exact=OK\n";
    }

    std::cout << "W=28_plan lanes_per_position=1"
              << " prefix_height=2popc"
              << " face_boundary=ballot"
              << " partner_search_rounds<=27"
              << " central_preimages_max=14"
              << "\n";
    std::cout << "ALL_OK parallel_marked_face_inverse=1\n";
    return 0;
}
